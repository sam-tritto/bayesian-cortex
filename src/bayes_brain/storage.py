import abc
import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

class BaseStorage(abc.ABC):
    """Abstract base class defining the storage backend interface for BayesBrain."""

    @abc.abstractmethod
    def get_tool_params(self, context_key: str, tool_name: str) -> Tuple[float, float]:
        """
        Retrieve the (alpha, beta) posterior parameters for a tool under a given context.
        Defaults to (1.0, 1.0) if not found.
        """
        pass

    @abc.abstractmethod
    def update_tool_params(
        self, context_key: str, tool_name: str, alpha: float, beta: float
    ) -> None:
        """
        Directly set the (alpha, beta) parameters for a tool under a given context.
        """
        pass

    @abc.abstractmethod
    def decay_and_update(
        self, context_key: str, tool_name: str, decay_factor: float, reward: float
    ) -> Tuple[float, float]:
        """
        Atomically decay the current parameters and add the reward.
        alpha_new = alpha_old * decay_factor + reward
        beta_new = beta_old * decay_factor + (1 - reward)
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Close any resources associated with the storage backend."""
        pass

    @abc.abstractmethod
    def load_metadata(self, key: str) -> Optional[str]:
        """Retrieve stored metadata for a given key, or None if not found."""
        pass

    @abc.abstractmethod
    def save_metadata(self, key: str, value: str) -> None:
        """Store metadata key-value pair."""
        pass

    def load_all_vectors(self) -> Dict[str, List[float]]:
        """
        Retrieve all stored context vectors from the backend.
        Fallback implementation uses metadata for backwards compatibility.
        """
        try:
            serialized = self.load_metadata("vector_context_store")
            if serialized:
                data = json.loads(serialized)
                return {k: list(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def save_vector(self, context_key: str, vector: Sequence[float]) -> None:
        """
        Store a single context vector incrementally.
        Fallback implementation updates the entire metadata JSON string.
        """
        try:
            serialized = self.load_metadata("vector_context_store")
            if serialized:
                data = json.loads(serialized)
            else:
                data = {}
            data[context_key] = list(vector)
            self.save_metadata("vector_context_store", json.dumps(data))
        except Exception:
            pass


class InMemoryStorage(BaseStorage):
    """
    In-memory thread-safe implementation of BaseStorage.
    Perfect for unit testing and ephemeral sessions.
    """

    def __init__(self) -> None:
        self._data: dict[Tuple[str, str], Tuple[float, float]] = {}
        self._metadata: dict[str, str] = {}
        self._vectors: dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def get_tool_params(self, context_key: str, tool_name: str) -> Tuple[float, float]:
        with self._lock:
            return self._data.get((context_key, tool_name), (1.0, 1.0))

    def update_tool_params(
        self, context_key: str, tool_name: str, alpha: float, beta: float
    ) -> None:
        with self._lock:
            self._data[(context_key, tool_name)] = (alpha, beta)

    def decay_and_update(
        self, context_key: str, tool_name: str, decay_factor: float, reward: float
    ) -> Tuple[float, float]:
        with self._lock:
            alpha, beta = self._data.get((context_key, tool_name), (1.0, 1.0))
            new_alpha = alpha * decay_factor + reward
            new_beta = beta * decay_factor + (1.0 - reward)
            self._data[(context_key, tool_name)] = (new_alpha, new_beta)
            return new_alpha, new_beta

    def close(self) -> None:
        pass

    def load_metadata(self, key: str) -> Optional[str]:
        with self._lock:
            return self._metadata.get(key)

    def save_metadata(self, key: str, value: str) -> None:
        with self._lock:
            self._metadata[key] = value

    def load_all_vectors(self) -> Dict[str, List[float]]:
        with self._lock:
            if not self._vectors:
                # Backwards compatibility migration check
                serialized = self._metadata.get("vector_context_store")
                if serialized:
                    try:
                        data = json.loads(serialized)
                        self._vectors = {k: list(v) for k, v in data.items()}
                    except Exception:
                        pass
            return dict(self._vectors)

    def save_vector(self, context_key: str, vector: Sequence[float]) -> None:
        with self._lock:
            self._vectors[context_key] = list(vector)


class SQLiteStorage(BaseStorage):
    """
    SQLite-backed storage for persistent local storage with thread safety.
    Guarantees atomic updates by utilizing BEGIN IMMEDIATE transactions.
    """

    def __init__(self, db_path: str = "bayes_brain.db") -> None:
        self.db_path = db_path
        # Initialize the database tables if they do not exist
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tool_params (
                        context_key TEXT,
                        tool_name TEXT,
                        alpha REAL,
                        beta REAL,
                        PRIMARY KEY (context_key, tool_name)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        val TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS context_vectors (
                        context_key TEXT PRIMARY KEY,
                        vector TEXT
                    )
                    """
                )
        finally:
            conn.close()
        
        self._local = threading.local()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = self._connect()
        return self._local.conn

    def get_tool_params(self, context_key: str, tool_name: str) -> Tuple[float, float]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT alpha, beta FROM tool_params WHERE context_key = ? AND tool_name = ?",
            (context_key, tool_name),
        )
        row = cursor.fetchone()
        if row is not None:
            return float(row[0]), float(row[1])
        return 1.0, 1.0

    def update_tool_params(
        self, context_key: str, tool_name: str, alpha: float, beta: float
    ) -> None:
        conn = self._get_conn()
        with conn:
            conn.execute(
                """
                INSERT INTO tool_params (context_key, tool_name, alpha, beta)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(context_key, tool_name) DO UPDATE SET
                    alpha = excluded.alpha,
                    beta = excluded.beta
                """,
                (context_key, tool_name, alpha, beta),
            )

    def decay_and_update(
        self, context_key: str, tool_name: str, decay_factor: float, reward: float
    ) -> Tuple[float, float]:
        conn = self._get_conn()
        # Use BEGIN IMMEDIATE to lock the database and ensure atomicity in multi-threaded contexts
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                "SELECT alpha, beta FROM tool_params WHERE context_key = ? AND tool_name = ?",
                (context_key, tool_name),
            )
            row = cursor.fetchone()
            if row is not None:
                alpha, beta = float(row[0]), float(row[1])
            else:
                alpha, beta = 1.0, 1.0

            new_alpha = alpha * decay_factor + reward
            new_beta = beta * decay_factor + (1.0 - reward)

            cursor.execute(
                """
                INSERT INTO tool_params (context_key, tool_name, alpha, beta)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(context_key, tool_name) DO UPDATE SET
                    alpha = excluded.alpha,
                    beta = excluded.beta
                """,
                (context_key, tool_name, new_alpha, new_beta),
            )
            conn.commit()
            return new_alpha, new_beta
        except Exception as e:
            conn.rollback()
            raise e

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            delattr(self._local, "conn")

    def load_metadata(self, key: str) -> Optional[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT val FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row is not None else None

    def save_metadata(self, key: str, value: str) -> None:
        conn = self._get_conn()
        with conn:
            conn.execute(
                """
                INSERT INTO metadata (key, val) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET val = excluded.val
                """,
                (key, value),
            )

    def load_all_vectors(self) -> Dict[str, List[float]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT context_key, vector FROM context_vectors")
        rows = cursor.fetchall()

        if not rows:
            # Migration check: see if there's legacy metadata
            serialized = self.load_metadata("vector_context_store")
            if serialized:
                try:
                    data = json.loads(serialized)
                    with conn:
                        for k, v in data.items():
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO context_vectors (context_key, vector)
                                VALUES (?, ?)
                                """,
                                (k, json.dumps(v)),
                            )
                    # Query again
                    cursor.execute("SELECT context_key, vector FROM context_vectors")
                    rows = cursor.fetchall()
                except Exception:
                    pass

        res = {}
        for row in rows:
            res[row[0]] = json.loads(row[1])
        return res

    def save_vector(self, context_key: str, vector: Sequence[float]) -> None:
        conn = self._get_conn()
        with conn:
            conn.execute(
                """
                INSERT INTO context_vectors (context_key, vector)
                VALUES (?, ?)
                ON CONFLICT(context_key) DO UPDATE SET vector = excluded.vector
                """,
                (context_key, json.dumps(list(vector))),
            )


class RedisStorage(BaseStorage):
    """
    Redis-backed storage backend for distributed setups.
    Uses Redis hashes and a Lua script to perform atomic multiply-and-add decay updates.
    """

    LUA_DECAY_UPDATE = """
    local key = KEYS[1]
    local field_alpha = ARGV[1]
    local field_beta = ARGV[2]
    local decay = tonumber(ARGV[3])
    local reward = tonumber(ARGV[4])
    local reward_fail = 1.0 - reward

    local alpha = redis.call('HGET', key, field_alpha)
    local beta = redis.call('HGET', key, field_beta)

    if not alpha then alpha = 1.0 else alpha = tonumber(alpha) end
    if not beta then beta = 1.0 else beta = tonumber(beta) end

    local new_alpha = alpha * decay + reward
    local new_beta = beta * decay + reward_fail

    redis.call('HSET', key, field_alpha, tostring(new_alpha), field_beta, tostring(new_beta))
    return {tostring(new_alpha), tostring(new_beta)}
    """

    def __init__(self, redis_client: Any, prefix: str = "bayes_brain:") -> None:
        """
        Initialize with a pre-configured redis-py Client.
        """
        self.client = redis_client
        self.prefix = prefix
        self._script = self.client.register_script(self.LUA_DECAY_UPDATE)

    def _get_key(self, context_key: str) -> str:
        return f"{self.prefix}{context_key}"

    def get_tool_params(self, context_key: str, tool_name: str) -> Tuple[float, float]:
        key = self._get_key(context_key)
        alpha_val = self.client.hget(key, f"{tool_name}:alpha")
        beta_val = self.client.hget(key, f"{tool_name}:beta")

        alpha = float(alpha_val) if alpha_val is not None else 1.0
        beta = float(beta_val) if beta_val is not None else 1.0
        return alpha, beta

    def update_tool_params(
        self, context_key: str, tool_name: str, alpha: float, beta: float
    ) -> None:
        key = self._get_key(context_key)
        self.client.hset(
            key,
            mapping={
                f"{tool_name}:alpha": str(alpha),
                f"{tool_name}:beta": str(beta),
            },
        )

    def decay_and_update(
        self, context_key: str, tool_name: str, decay_factor: float, reward: float
    ) -> Tuple[float, float]:
        key = self._get_key(context_key)
        res = self._script(
            keys=[key],
            args=[
                f"{tool_name}:alpha",
                f"{tool_name}:beta",
                str(decay_factor),
                str(reward),
            ],
        )
        return float(res[0]), float(res[1])

    def close(self) -> None:
        # We don't close the client as it is passed from outside
        pass

    def load_metadata(self, key: str) -> Optional[str]:
        val = self.client.get(f"{self.prefix}metadata:{key}")
        if val is None:
            return None
        return val.decode("utf-8") if isinstance(val, bytes) else str(val)

    def save_metadata(self, key: str, value: str) -> None:
        self.client.set(f"{self.prefix}metadata:{key}", value)

    def load_all_vectors(self) -> Dict[str, List[float]]:
        vectors_hash = self.client.hgetall(f"{self.prefix}context_vectors")
        if not vectors_hash:
            # Check for legacy metadata to migrate
            serialized = self.load_metadata("vector_context_store")
            if serialized:
                try:
                    data = json.loads(serialized)
                    mapping = {k: json.dumps(v) for k, v in data.items()}
                    if mapping:
                        self.client.hset(f"{self.prefix}context_vectors", mapping=mapping)
                    return data
                except Exception:
                    pass
        res = {}
        for k, v in vectors_hash.items():
            key_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            val_str = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            res[key_str] = json.loads(val_str)
        return res

    def save_vector(self, context_key: str, vector: Sequence[float]) -> None:
        self.client.hset(
            f"{self.prefix}context_vectors",
            key=context_key,
            value=json.dumps(list(vector))
        )
