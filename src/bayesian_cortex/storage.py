import abc
import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class BaseStorage(abc.ABC):
    """Abstract base class defining the storage backend interface for BayesianCortex."""

    @abc.abstractmethod
    def get_candidate_params(
        self, context_key: str, candidate_name: str
    ) -> Tuple[float, float]:
        """
        Retrieve the (alpha, beta) posterior parameters for a candidate under a given context.
        Defaults to (1.0, 1.0) if not found.
        """
        pass

    @abc.abstractmethod
    def has_candidate_params(self, context_key: str, candidate_name: str) -> bool:
        """
        Return True if the candidate has ever been observed under the given context
        (i.e. a record exists in storage), False otherwise.

        This must NOT be inferred from the parameter values alone. After a failure
        with decay_factor=1.0 the parameters legitimately remain at the floor (1.0, 1.0),
        which is identical to the cold-start default. Only an explicit existence check
        can distinguish the two states.
        """
        pass

    @abc.abstractmethod
    def update_candidate_params(
        self, context_key: str, candidate_name: str, alpha: float, beta: float
    ) -> None:
        """
        Directly set the (alpha, beta) parameters for a candidate under a given context.
        """
        pass

    @abc.abstractmethod
    def decay_and_update(
        self, context_key: str, candidate_name: str, decay_factor: float, reward: float
    ) -> Tuple[float, float]:
        """
        Atomically decay the current parameters and add the reward, ensuring
        they do not drop below the flat prior baseline of 1.0:
        alpha_new = max(1.0, alpha_old * decay_factor + reward)
        beta_new = max(1.0, beta_old * decay_factor + (1 - reward))

        Note: Clamping alpha and beta to a lower bound of 1.0 is critical for standard Thompson Sampling.
        If they were allowed to drop below 1.0 (e.g. under aggressive decay), the Beta probability density
        function (PDF) would turn into a bimodal U-shape. This would force random draws to spike violently
        to either absolute 0 or absolute 1, causing completely erratic routing behavior. Clamping protects
        the stability of the distribution's mode.
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

    @abc.abstractmethod
    def get_linear_params(
        self, candidate_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieve (precision, reward_vector) for a candidate.
        Returns (None, None) if not found.
        """
        pass

    @abc.abstractmethod
    def decay_and_update_linear(
        self,
        candidate_name: str,
        decay_factor: float,
        reward: float,
        x_augmented: np.ndarray,
        lambda_val: float,
        prior_p: float,
        diagonal: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Atomically decay the current parameters for a candidate and add the new observation,
        returning the updated (precision, reward_vector).
        """
        pass

    def get_candidate_params_batch(
        self, keys: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """
        Batch retrieve the (alpha, beta) posterior parameters for a list of context and candidate keys.
        """
        return {key: self.get_candidate_params(key[0], key[1]) for key in keys}

    def update_candidate_params_batch(
        self, params: Dict[Tuple[str, str], Tuple[float, float]]
    ) -> None:
        """
        Batch set the (alpha, beta) parameters.
        """
        for (ctx, cand), (alpha, beta) in params.items():
            self.update_candidate_params(ctx, cand, alpha, beta)

    def decay_and_update_batch(
        self, updates: List[Tuple[str, str, float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Batch decay and update parameters in order.
        Each update is (context_key, candidate_name, decay_factor, reward).
        """
        return [
            self.decay_and_update(ctx, cand, decay, reward)
            for ctx, cand, decay, reward in updates
        ]

    def get_linear_params_batch(
        self, candidate_names: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Batch retrieve (precision, reward_vector) for multiple candidates.
        """
        results = {}
        for c in candidate_names:
            val = self.get_linear_params(c)
            if val[0] is not None and val[1] is not None:
                results[c] = val
        return results

    def decay_and_update_linear_batch(
        self, updates: List[Tuple[str, float, float, np.ndarray, float, float, bool]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Batch decay and update linear parameters in order.
        Each update is (candidate_name, decay_factor, reward, x_augmented, lambda_val, prior_p, diagonal).
        """
        return [
            self.decay_and_update_linear(cand, decay, reward, x_aug, lamb, prior, diag)
            for cand, decay, reward, x_aug, lamb, prior, diag in updates
        ]

    def save_vectors(self, vectors: Dict[str, Sequence[float]]) -> None:
        """
        Batch store context vectors incrementally.
        """
        for key, vector in vectors.items():
            self.save_vector(key, vector)

    @abc.abstractmethod
    def log_selection(
        self, trace_id: str, context_key: str, candidate_name: str
    ) -> None:
        """Log a candidate selection event."""
        pass

    @abc.abstractmethod
    def log_feedback(self, trace_id: str, reward: float) -> None:
        """Log reward feedback for a selection event."""
        pass

    @abc.abstractmethod
    def get_selection_logs(self) -> List[Dict[str, Any]]:
        """Retrieve all selection logs, ordered by timestamp ascending."""
        return []


class InMemoryStorage(BaseStorage):
    """
    In-memory thread-safe implementation of BaseStorage.
    Perfect for unit testing and ephemeral sessions.
    """

    def __init__(self) -> None:
        self._data: dict[Tuple[str, str], Tuple[float, float]] = {}
        self._metadata: dict[str, str] = {}
        self._vectors: dict[str, List[float]] = {}
        self._linear_data: dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._selection_logs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_candidate_params(
        self, context_key: str, candidate_name: str
    ) -> Tuple[float, float]:
        with self._lock:
            return self._data.get((context_key, candidate_name), (1.0, 1.0))

    def has_candidate_params(self, context_key: str, candidate_name: str) -> bool:
        with self._lock:
            return (context_key, candidate_name) in self._data

    def update_candidate_params(
        self, context_key: str, candidate_name: str, alpha: float, beta: float
    ) -> None:
        with self._lock:
            self._data[(context_key, candidate_name)] = (alpha, beta)

    def decay_and_update(
        self, context_key: str, candidate_name: str, decay_factor: float, reward: float
    ) -> Tuple[float, float]:
        with self._lock:
            alpha, beta = self._data.get((context_key, candidate_name), (1.0, 1.0))
            new_alpha = max(1.0, alpha * decay_factor + reward)
            new_beta = max(1.0, beta * decay_factor + (1.0 - reward))
            self._data[(context_key, candidate_name)] = (new_alpha, new_beta)
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

    def get_linear_params(
        self, candidate_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            val = self._linear_data.get(candidate_name)
            if val is not None:
                return np.copy(val[0]), np.copy(val[1])
            return None, None

    def decay_and_update_linear(
        self,
        candidate_name: str,
        decay_factor: float,
        reward: float,
        x_augmented: np.ndarray,
        lambda_val: float,
        prior_p: float,
        diagonal: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d = len(x_augmented)
        with self._lock:
            val = self._linear_data.get(candidate_name)
            if val is not None:
                precision, reward_vector = val
            else:
                precision = (
                    lambda_val * np.ones(d, dtype=np.float32)
                    if diagonal
                    else lambda_val * np.eye(d, dtype=np.float32)
                )
                reward_vector = np.zeros(d, dtype=np.float32)
                reward_vector[-1] = lambda_val * prior_p

            prior_reward_vector = np.zeros(d, dtype=np.float32)
            prior_reward_vector[-1] = lambda_val * prior_p

            if diagonal:
                new_precision = (
                    decay_factor * precision
                    + (1.0 - decay_factor) * lambda_val * np.ones(d, dtype=np.float32)
                    + (x_augmented**2)
                )
            else:
                new_precision = (
                    decay_factor * precision
                    + (1.0 - decay_factor) * lambda_val * np.eye(d, dtype=np.float32)
                    + np.outer(x_augmented, x_augmented)
                )

            new_reward_vector = (
                decay_factor * reward_vector
                + (1.0 - decay_factor) * prior_reward_vector
                + reward * x_augmented
            )

            self._linear_data[candidate_name] = (new_precision, new_reward_vector)
            return np.copy(new_precision), np.copy(new_reward_vector)

    def get_candidate_params_batch(
        self, keys: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        with self._lock:
            return {key: self._data.get(key, (1.0, 1.0)) for key in keys}

    def update_candidate_params_batch(
        self, params: Dict[Tuple[str, str], Tuple[float, float]]
    ) -> None:
        with self._lock:
            for key, val in params.items():
                self._data[key] = val

    def decay_and_update_batch(
        self, updates: List[Tuple[str, str, float, float]]
    ) -> List[Tuple[float, float]]:
        with self._lock:
            results = []
            for context_key, candidate_name, decay_factor, reward in updates:
                alpha, beta = self._data.get((context_key, candidate_name), (1.0, 1.0))
                new_alpha = max(1.0, alpha * decay_factor + reward)
                new_beta = max(1.0, beta * decay_factor + (1.0 - reward))
                self._data[(context_key, candidate_name)] = (new_alpha, new_beta)
                results.append((new_alpha, new_beta))
            return results

    def get_linear_params_batch(
        self, candidate_names: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            results = {}
            for c in candidate_names:
                val = self._linear_data.get(c)
                if val is not None:
                    results[c] = (np.copy(val[0]), np.copy(val[1]))
            return results

    def decay_and_update_linear_batch(
        self, updates: List[Tuple[str, float, float, np.ndarray, float, float, bool]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            results = []
            for (
                candidate_name,
                decay_factor,
                reward,
                x_augmented,
                lambda_val,
                prior_p,
                diagonal,
            ) in updates:
                d = len(x_augmented)
                val = self._linear_data.get(candidate_name)
                if val is not None:
                    precision, reward_vector = val
                else:
                    precision = (
                        lambda_val * np.ones(d, dtype=np.float32)
                        if diagonal
                        else lambda_val * np.eye(d, dtype=np.float32)
                    )
                    reward_vector = np.zeros(d, dtype=np.float32)
                    reward_vector[-1] = lambda_val * prior_p

                prior_reward_vector = np.zeros(d, dtype=np.float32)
                prior_reward_vector[-1] = lambda_val * prior_p

                if diagonal:
                    new_precision = (
                        decay_factor * precision
                        + (1.0 - decay_factor)
                        * lambda_val
                        * np.ones(d, dtype=np.float32)
                        + (x_augmented**2)
                    )
                else:
                    new_precision = (
                        decay_factor * precision
                        + (1.0 - decay_factor)
                        * lambda_val
                        * np.eye(d, dtype=np.float32)
                        + np.outer(x_augmented, x_augmented)
                    )

                new_reward_vector = (
                    decay_factor * reward_vector
                    + (1.0 - decay_factor) * prior_reward_vector
                    + reward * x_augmented
                )

                self._linear_data[candidate_name] = (new_precision, new_reward_vector)
                results.append((np.copy(new_precision), np.copy(new_reward_vector)))
            return results

    def save_vectors(self, vectors: Dict[str, Sequence[float]]) -> None:
        with self._lock:
            for key, vector in vectors.items():
                self._vectors[key] = list(vector)

    def log_selection(
        self, trace_id: str, context_key: str, candidate_name: str
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if trace_id not in self._selection_logs:
                self._selection_logs[trace_id] = {
                    "trace_id": trace_id,
                    "timestamp": timestamp,
                    "context_key": context_key,
                    "candidate_name": candidate_name,
                    "reward": None,
                }

    def log_feedback(self, trace_id: str, reward: float) -> None:
        with self._lock:
            if trace_id in self._selection_logs:
                self._selection_logs[trace_id]["reward"] = reward

    def get_selection_logs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return sorted(self._selection_logs.values(), key=lambda x: x["timestamp"])


def _retry_on_lock(max_retries: int = 5, initial_delay: float = 0.05, max_delay: float = 1.0):
    """
    Decorator to retry synchronous database operations that fail due to locking.
    Uses exponential backoff with jitter.
    """
    import functools
    import random
    import time

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if attempt == max_retries or "locked" not in str(e).lower():
                        raise
                    sleep_time = delay * (0.5 + random.random())
                    time.sleep(sleep_time)
                    delay = min(delay * 2, max_delay)
        return wrapper
    return decorator


class SQLiteStorage(BaseStorage):
    """
    SQLite-backed storage for persistent local storage with thread safety.
    Guarantees atomic updates by utilizing BEGIN IMMEDIATE transactions.
    """

    def __init__(self, db_path: str = "bayesian_cortex.db") -> None:
        self.db_path = db_path
        self._init_db()
        self._local = threading.local()

    @_retry_on_lock()
    def _init_db(self) -> None:
        # Initialize the database tables if they do not exist
        conn = self._connect()
        try:
            with conn:
                self._migrate_database(conn)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS candidate_params (
                        context_key TEXT,
                        candidate_name TEXT,
                        alpha REAL,
                        beta REAL,
                        PRIMARY KEY (context_key, candidate_name)
                    )
                    """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        val TEXT
                    )
                    """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS context_vectors (
                        context_key TEXT PRIMARY KEY,
                        vector TEXT
                    )
                    """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS linear_bandit_params (
                        candidate_name TEXT PRIMARY KEY,
                        precision_matrix TEXT,
                        reward_vector TEXT
                    )
                    """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS selection_log (
                        trace_id TEXT PRIMARY KEY,
                        timestamp TEXT,
                        context_key TEXT,
                        candidate_name TEXT,
                        reward REAL
                    )
                    """)
        finally:
            conn.close()

    def _migrate_database(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        # Rename table tool_params -> candidate_params
        cursor.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='tool_params';"
        )
        if cursor.fetchone()[0] > 0:
            cursor.execute("ALTER TABLE tool_params RENAME TO candidate_params;")
            cursor.execute(
                "ALTER TABLE candidate_params RENAME COLUMN tool_name TO candidate_name;"
            )

        # Check linear_bandit_params column tool_name -> candidate_name
        cursor.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='linear_bandit_params';"
        )
        if cursor.fetchone()[0] > 0:
            cursor.execute("PRAGMA table_info(linear_bandit_params);")
            columns = [row[1] for row in cursor.fetchall()]
            if "tool_name" in columns:
                cursor.execute(
                    "ALTER TABLE linear_bandit_params RENAME COLUMN tool_name TO candidate_name;"
                )

        # Check selection_log column tool_name -> candidate_name
        cursor.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='selection_log';"
        )
        if cursor.fetchone()[0] > 0:
            cursor.execute("PRAGMA table_info(selection_log);")
            columns = [row[1] for row in cursor.fetchall()]
            if "tool_name" in columns:
                cursor.execute(
                    "ALTER TABLE selection_log RENAME COLUMN tool_name TO candidate_name;"
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = self._connect()
        return self._local.conn

    def get_candidate_params(
        self, context_key: str, candidate_name: str
    ) -> Tuple[float, float]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT alpha, beta FROM candidate_params WHERE context_key = ? AND candidate_name = ?",
            (context_key, candidate_name),
        )
        row = cursor.fetchone()
        if row is not None:
            return float(row[0]), float(row[1])
        return 1.0, 1.0

    def has_candidate_params(self, context_key: str, candidate_name: str) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM candidate_params WHERE context_key = ? AND candidate_name = ? LIMIT 1",
            (context_key, candidate_name),
        )
        return cursor.fetchone() is not None

    @_retry_on_lock()
    def update_candidate_params(
        self, context_key: str, candidate_name: str, alpha: float, beta: float
    ) -> None:
        conn = self._get_conn()
        with conn:
            conn.execute(
                """
                INSERT INTO candidate_params (context_key, candidate_name, alpha, beta)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(context_key, candidate_name) DO UPDATE SET
                    alpha = excluded.alpha,
                    beta = excluded.beta
                """,
                (context_key, candidate_name, alpha, beta),
            )

    @_retry_on_lock()
    def decay_and_update(
        self, context_key: str, candidate_name: str, decay_factor: float, reward: float
    ) -> Tuple[float, float]:
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                "SELECT alpha, beta FROM candidate_params WHERE context_key = ? AND candidate_name = ?",
                (context_key, candidate_name),
            )
            row = cursor.fetchone()
            if row is not None:
                alpha, beta = float(row[0]), float(row[1])
            else:
                alpha, beta = 1.0, 1.0

            new_alpha = max(1.0, alpha * decay_factor + reward)
            new_beta = max(1.0, beta * decay_factor + (1.0 - reward))

            cursor.execute(
                """
                INSERT INTO candidate_params (context_key, candidate_name, alpha, beta)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(context_key, candidate_name) DO UPDATE SET
                    alpha = excluded.alpha,
                    beta = excluded.beta
                """,
                (context_key, candidate_name, new_alpha, new_beta),
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

    @_retry_on_lock()
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

    @_retry_on_lock()
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

    @_retry_on_lock()
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

    def get_linear_params(
        self, candidate_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT precision_matrix, reward_vector FROM linear_bandit_params WHERE candidate_name = ?",
            (candidate_name,),
        )
        row = cursor.fetchone()
        if row is not None:
            precision = np.array(json.loads(row[0]), dtype=np.float32)
            reward_vector = np.array(json.loads(row[1]), dtype=np.float32)
            return precision, reward_vector
        return None, None

    @_retry_on_lock()
    def decay_and_update_linear(
        self,
        candidate_name: str,
        decay_factor: float,
        reward: float,
        x_augmented: np.ndarray,
        lambda_val: float,
        prior_p: float,
        diagonal: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d = len(x_augmented)
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                "SELECT precision_matrix, reward_vector FROM linear_bandit_params WHERE candidate_name = ?",
                (candidate_name,),
            )
            row = cursor.fetchone()
            if row is not None:
                precision = np.array(json.loads(row[0]), dtype=np.float32)
                reward_vector = np.array(json.loads(row[1]), dtype=np.float32)
            else:
                precision = (
                    lambda_val * np.ones(d, dtype=np.float32)
                    if diagonal
                    else lambda_val * np.eye(d, dtype=np.float32)
                )
                reward_vector = np.zeros(d, dtype=np.float32)
                reward_vector[-1] = lambda_val * prior_p

            prior_reward_vector = np.zeros(d, dtype=np.float32)
            prior_reward_vector[-1] = lambda_val * prior_p

            if diagonal:
                new_precision = (
                    decay_factor * precision
                    + (1.0 - decay_factor) * lambda_val * np.ones(d, dtype=np.float32)
                    + (x_augmented**2)
                )
            else:
                new_precision = (
                    decay_factor * precision
                    + (1.0 - decay_factor) * lambda_val * np.eye(d, dtype=np.float32)
                    + np.outer(x_augmented, x_augmented)
                )

            new_reward_vector = (
                decay_factor * reward_vector
                + (1.0 - decay_factor) * prior_reward_vector
                + reward * x_augmented
            )

            cursor.execute(
                """
                INSERT INTO linear_bandit_params (candidate_name, precision_matrix, reward_vector)
                VALUES (?, ?, ?)
                ON CONFLICT(candidate_name) DO UPDATE SET
                    precision_matrix = excluded.precision_matrix,
                    reward_vector = excluded.reward_vector
                """,
                (
                    candidate_name,
                    json.dumps(new_precision.tolist()),
                    json.dumps(new_reward_vector.tolist()),
                ),
            )
            conn.commit()
            return new_precision, new_reward_vector
        except Exception as e:
            conn.rollback()
            raise e

    def get_candidate_params_batch(
        self, keys: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        if not keys:
            return {}
        results = dict.fromkeys(keys, (1.0, 1.0))
        conn = self._get_conn()
        cursor = conn.cursor()
        chunk_size = 200
        for i in range(0, len(keys), chunk_size):
            chunk = keys[i : i + chunk_size]
            clauses = []
            params = []
            for c_key, cand_name in chunk:
                clauses.append("(context_key = ? AND candidate_name = ?)")
                params.extend([c_key, cand_name])
            query = (
                "SELECT context_key, candidate_name, alpha, beta FROM candidate_params WHERE "
                + " OR ".join(clauses)
            )
            cursor.execute(query, params)
            for row in cursor.fetchall():
                results[(row[0], row[1])] = (float(row[2]), float(row[3]))
        return results

    @_retry_on_lock()
    def update_candidate_params_batch(
        self, params: Dict[Tuple[str, str], Tuple[float, float]]
    ) -> None:
        if not params:
            return
        conn = self._get_conn()
        with conn:
            conn.executemany(
                """
                INSERT INTO candidate_params (context_key, candidate_name, alpha, beta)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(context_key, candidate_name) DO UPDATE SET
                    alpha = excluded.alpha,
                    beta = excluded.beta
                """,
                [
                    (ctx, cand, alpha, beta)
                    for (ctx, cand), (alpha, beta) in params.items()
                ],
            )

    @_retry_on_lock()
    def decay_and_update_batch(
        self, updates: List[Tuple[str, str, float, float]]
    ) -> List[Tuple[float, float]]:
        if not updates:
            return []

        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")

            keys = [(ctx, cand) for ctx, cand, _, _ in updates]
            current_vals = {}
            chunk_size = 200
            for i in range(0, len(keys), chunk_size):
                chunk = keys[i : i + chunk_size]
                clauses = []
                params = []
                for c_key, cand_name in chunk:
                    clauses.append("(context_key = ? AND candidate_name = ?)")
                    params.extend([c_key, cand_name])
                query = (
                    "SELECT context_key, candidate_name, alpha, beta FROM candidate_params WHERE "
                    + " OR ".join(clauses)
                )
                cursor.execute(query, params)
                for row in cursor.fetchall():
                    current_vals[(row[0], row[1])] = (float(row[2]), float(row[3]))

            updated_params = []
            for ctx, cand, decay_factor, reward in updates:
                alpha, beta = current_vals.get((ctx, cand), (1.0, 1.0))
                new_alpha = max(1.0, alpha * decay_factor + reward)
                new_beta = max(1.0, beta * decay_factor + (1.0 - reward))
                current_vals[(ctx, cand)] = (new_alpha, new_beta)
                updated_params.append((ctx, cand, new_alpha, new_beta))

            cursor.executemany(
                """
                INSERT INTO candidate_params (context_key, candidate_name, alpha, beta)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(context_key, candidate_name) DO UPDATE SET
                    alpha = excluded.alpha,
                    beta = excluded.beta
                """,
                [
                    (ctx, cand, val[0], val[1])
                    for (ctx, cand), val in current_vals.items()
                ],
            )

            conn.commit()
            return [(item[2], item[3]) for item in updated_params]
        except Exception as e:
            conn.rollback()
            raise e

    def get_linear_params_batch(
        self, candidate_names: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if not candidate_names:
            return {}
        conn = self._get_conn()
        cursor = conn.cursor()
        placeholders = ",".join(["?"] * len(candidate_names))
        cursor.execute(
            f"SELECT candidate_name, precision_matrix, reward_vector FROM linear_bandit_params WHERE candidate_name IN ({placeholders})",
            candidate_names,
        )
        results = {}
        for row in cursor.fetchall():
            precision = np.array(json.loads(row[1]), dtype=np.float32)
            reward_vector = np.array(json.loads(row[2]), dtype=np.float32)
            results[row[0]] = (precision, reward_vector)
        return results

    def decay_and_update_linear_batch(
        self, updates: List[Tuple[str, float, float, np.ndarray, float, float, bool]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not updates:
            return []

        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")

            candidate_names = list(set([item[0] for item in updates]))
            current_vals = {}
            if candidate_names:
                placeholders = ",".join(["?"] * len(candidate_names))
                query = f"SELECT candidate_name, precision_matrix, reward_vector FROM linear_bandit_params WHERE candidate_name IN ({placeholders})"
                cursor.execute(query, candidate_names)
                for row in cursor.fetchall():
                    precision = np.array(json.loads(row[1]), dtype=np.float32)
                    reward_vector = np.array(json.loads(row[2]), dtype=np.float32)
                    current_vals[row[0]] = (precision, reward_vector)

            results = []
            for (
                candidate_name,
                decay_factor,
                reward,
                x_augmented,
                lambda_val,
                prior_p,
                diagonal,
            ) in updates:
                d = len(x_augmented)
                val = current_vals.get(candidate_name)
                if val is not None:
                    precision, reward_vector = val
                else:
                    precision = (
                        lambda_val * np.ones(d, dtype=np.float32)
                        if diagonal
                        else lambda_val * np.eye(d, dtype=np.float32)
                    )
                    reward_vector = np.zeros(d, dtype=np.float32)
                    reward_vector[-1] = lambda_val * prior_p

                prior_reward_vector = np.zeros(d, dtype=np.float32)
                prior_reward_vector[-1] = lambda_val * prior_p

                if diagonal:
                    new_precision = (
                        decay_factor * precision
                        + (1.0 - decay_factor)
                        * lambda_val
                        * np.ones(d, dtype=np.float32)
                        + (x_augmented**2)
                    )
                else:
                    new_precision = (
                        decay_factor * precision
                        + (1.0 - decay_factor)
                        * lambda_val
                        * np.eye(d, dtype=np.float32)
                        + np.outer(x_augmented, x_augmented)
                    )

                new_reward_vector = (
                    decay_factor * reward_vector
                    + (1.0 - decay_factor) * prior_reward_vector
                    + reward * x_augmented
                )

                current_vals[candidate_name] = (new_precision, new_reward_vector)
                results.append((np.copy(new_precision), np.copy(new_reward_vector)))

            db_updates = []
            for c_name, (prec, rew) in current_vals.items():
                db_updates.append(
                    (c_name, json.dumps(prec.tolist()), json.dumps(rew.tolist()))
                )

            cursor.executemany(
                """
                INSERT INTO linear_bandit_params (candidate_name, precision_matrix, reward_vector)
                VALUES (?, ?, ?)
                ON CONFLICT(candidate_name) DO UPDATE SET
                    precision_matrix = excluded.precision_matrix,
                    reward_vector = excluded.reward_vector
                """,
                db_updates,
            )

            conn.commit()
            return results
        except Exception as e:
            conn.rollback()
            raise e

    def save_vectors(self, vectors: Dict[str, Sequence[float]]) -> None:
        if not vectors:
            return
        conn = self._get_conn()
        with conn:
            conn.executemany(
                """
                INSERT INTO context_vectors (context_key, vector)
                VALUES (?, ?)
                ON CONFLICT(context_key) DO UPDATE SET vector = excluded.vector
                """,
                [(k, json.dumps(list(v))) for k, v in vectors.items()],
            )

    def log_selection(
        self, trace_id: str, context_key: str, candidate_name: str
    ) -> None:
        conn = self._get_conn()
        timestamp = datetime.now(timezone.utc).isoformat()
        with conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO selection_log (trace_id, timestamp, context_key, candidate_name, reward)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (trace_id, timestamp, context_key, candidate_name),
            )

    def log_feedback(self, trace_id: str, reward: float) -> None:
        conn = self._get_conn()
        with conn:
            conn.execute(
                "UPDATE selection_log SET reward = ? WHERE trace_id = ?",
                (reward, trace_id),
            )

    def get_selection_logs(self) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT trace_id, timestamp, context_key, candidate_name, reward FROM selection_log ORDER BY timestamp ASC"
        )
        logs = []
        for row in cursor.fetchall():
            logs.append(
                {
                    "trace_id": row[0],
                    "timestamp": row[1],
                    "context_key": row[2],
                    "candidate_name": row[3],
                    "reward": float(row[4]) if row[4] is not None else None,
                }
            )
        return logs


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

    local new_alpha = math.max(1.0, alpha * decay + reward)
    local new_beta = math.max(1.0, beta * decay + reward_fail)

    redis.call('HSET', key, field_alpha, tostring(new_alpha), field_beta, tostring(new_beta))
    return {tostring(new_alpha), tostring(new_beta)}
    """

    def __init__(self, redis_client: Any, prefix: str = "bayesian_cortex:") -> None:
        """
        Initialize with a pre-configured redis-py Client.
        """
        self.client = redis_client
        self.prefix = prefix
        self._script = self.client.register_script(self.LUA_DECAY_UPDATE)

    def _get_key(self, context_key: str) -> str:
        return f"{self.prefix}{context_key}"

    def get_candidate_params(
        self, context_key: str, candidate_name: str
    ) -> Tuple[float, float]:
        key = self._get_key(context_key)
        alpha_val = self.client.hget(key, f"{candidate_name}:alpha")
        beta_val = self.client.hget(key, f"{candidate_name}:beta")

        alpha = float(alpha_val) if alpha_val is not None else 1.0
        beta = float(beta_val) if beta_val is not None else 1.0
        return alpha, beta

    def has_candidate_params(self, context_key: str, candidate_name: str) -> bool:
        key = self._get_key(context_key)
        return bool(self.client.hexists(key, f"{candidate_name}:alpha"))

    def update_candidate_params(
        self, context_key: str, candidate_name: str, alpha: float, beta: float
    ) -> None:
        key = self._get_key(context_key)
        self.client.hset(
            key,
            mapping={
                f"{candidate_name}:alpha": str(alpha),
                f"{candidate_name}:beta": str(beta),
            },
        )

    def decay_and_update(
        self, context_key: str, candidate_name: str, decay_factor: float, reward: float
    ) -> Tuple[float, float]:
        key = self._get_key(context_key)
        res = self._script(
            keys=[key],
            args=[
                f"{candidate_name}:alpha",
                f"{candidate_name}:beta",
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
                        self.client.hset(
                            f"{self.prefix}context_vectors", mapping=mapping
                        )
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
            value=json.dumps(list(vector)),
        )

    def get_linear_params(
        self, candidate_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        key_prec = f"{self.prefix}linear:{candidate_name}:precision"
        key_rew = f"{self.prefix}linear:{candidate_name}:reward"
        prec_val = self.client.get(key_prec)
        rew_val = self.client.get(key_rew)
        if prec_val is not None and rew_val is not None:
            precision = np.array(json.loads(prec_val), dtype=np.float32)
            reward_vector = np.array(json.loads(rew_val), dtype=np.float32)
            return precision, reward_vector
        return None, None

    def decay_and_update_linear(
        self,
        candidate_name: str,
        decay_factor: float,
        reward: float,
        x_augmented: np.ndarray,
        lambda_val: float,
        prior_p: float,
        diagonal: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        import redis

        d = len(x_augmented)
        key_prec = f"{self.prefix}linear:{candidate_name}:precision"
        key_rew = f"{self.prefix}linear:{candidate_name}:reward"
        pipe = self.client.pipeline()
        while True:
            try:
                pipe.watch(key_prec, key_rew)
                prec_val = pipe.get(key_prec)
                rew_val = pipe.get(key_rew)
                if prec_val is not None and rew_val is not None:
                    precision = np.array(json.loads(prec_val), dtype=np.float32)
                    reward_vector = np.array(json.loads(rew_val), dtype=np.float32)
                else:
                    precision = (
                        lambda_val * np.ones(d, dtype=np.float32)
                        if diagonal
                        else lambda_val * np.eye(d, dtype=np.float32)
                    )
                    reward_vector = np.zeros(d, dtype=np.float32)
                    reward_vector[-1] = lambda_val * prior_p

                prior_reward_vector = np.zeros(d, dtype=np.float32)
                prior_reward_vector[-1] = lambda_val * prior_p

                if diagonal:
                    new_precision = (
                        decay_factor * precision
                        + (1.0 - decay_factor)
                        * lambda_val
                        * np.ones(d, dtype=np.float32)
                        + (x_augmented**2)
                    )
                else:
                    new_precision = (
                        decay_factor * precision
                        + (1.0 - decay_factor)
                        * lambda_val
                        * np.eye(d, dtype=np.float32)
                        + np.outer(x_augmented, x_augmented)
                    )

                new_reward_vector = (
                    decay_factor * reward_vector
                    + (1.0 - decay_factor) * prior_reward_vector
                    + reward * x_augmented
                )

                pipe.multi()
                pipe.set(key_prec, json.dumps(new_precision.tolist()))
                pipe.set(key_rew, json.dumps(new_reward_vector.tolist()))
                pipe.execute()
                return new_precision, new_reward_vector
            except redis.WatchError:
                continue

    def get_candidate_params_batch(
        self, keys: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        if not keys:
            return {}
        pipe = self.client.pipeline()
        for context_key, candidate_name in keys:
            key = self._get_key(context_key)
            pipe.hget(key, f"{candidate_name}:alpha")
            pipe.hget(key, f"{candidate_name}:beta")

        results_raw = pipe.execute()
        results = {}
        for idx, (context_key, candidate_name) in enumerate(keys):
            alpha_val = results_raw[2 * idx]
            beta_val = results_raw[2 * idx + 1]
            alpha = float(alpha_val) if alpha_val is not None else 1.0
            beta = float(beta_val) if beta_val is not None else 1.0
            results[(context_key, candidate_name)] = (alpha, beta)
        return results

    def update_candidate_params_batch(
        self, params: Dict[Tuple[str, str], Tuple[float, float]]
    ) -> None:
        if not params:
            return
        pipe = self.client.pipeline()
        for (context_key, candidate_name), (alpha, beta) in params.items():
            key = self._get_key(context_key)
            pipe.hset(
                key,
                mapping={
                    f"{candidate_name}:alpha": str(alpha),
                    f"{candidate_name}:beta": str(beta),
                },
            )
        pipe.execute()

    def decay_and_update_batch(
        self, updates: List[Tuple[str, str, float, float]]
    ) -> List[Tuple[float, float]]:
        if not updates:
            return []
        pipe = self.client.pipeline()
        for context_key, candidate_name, decay_factor, reward in updates:
            key = self._get_key(context_key)
            self._script(
                keys=[key],
                args=[
                    f"{candidate_name}:alpha",
                    f"{candidate_name}:beta",
                    str(decay_factor),
                    str(reward),
                ],
                client=pipe,
            )
        raw_results = pipe.execute()
        return [(float(res[0]), float(res[1])) for res in raw_results]

    def get_linear_params_batch(
        self, candidate_names: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if not candidate_names:
            return {}
        pipe = self.client.pipeline()
        for c in candidate_names:
            pipe.get(f"{self.prefix}linear:{c}:precision")
            pipe.get(f"{self.prefix}linear:{c}:reward")
        raw_vals = pipe.execute()
        results = {}
        for idx, c in enumerate(candidate_names):
            prec_val = raw_vals[2 * idx]
            rew_val = raw_vals[2 * idx + 1]
            if prec_val is not None and rew_val is not None:
                precision = np.array(json.loads(prec_val), dtype=np.float32)
                reward_vector = np.array(json.loads(rew_val), dtype=np.float32)
                results[c] = (precision, reward_vector)
        return results

    def decay_and_update_linear_batch(
        self, updates: List[Tuple[str, float, float, np.ndarray, float, float, bool]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not updates:
            return []
        import redis

        candidate_names = list(set([item[0] for item in updates]))
        keys_prec = [f"{self.prefix}linear:{c}:precision" for c in candidate_names]
        keys_rew = [f"{self.prefix}linear:{c}:reward" for c in candidate_names]
        all_keys = keys_prec + keys_rew

        pipe = self.client.pipeline()
        while True:
            try:
                pipe.watch(*all_keys)
                for k in keys_prec:
                    pipe.get(k)
                for k in keys_rew:
                    pipe.get(k)
                raw_vals = pipe.execute()

                current_vals = {}
                num_cands = len(candidate_names)
                for idx, c in enumerate(candidate_names):
                    prec_val = raw_vals[idx]
                    rew_val = raw_vals[num_cands + idx]
                    if prec_val is not None and rew_val is not None:
                        precision = np.array(json.loads(prec_val), dtype=np.float32)
                        reward_vector = np.array(json.loads(rew_val), dtype=np.float32)
                        current_vals[c] = (precision, reward_vector)

                results = []
                for (
                    candidate_name,
                    decay_factor,
                    reward,
                    x_augmented,
                    lambda_val,
                    prior_p,
                    diagonal,
                ) in updates:
                    d = len(x_augmented)
                    val = current_vals.get(candidate_name)
                    if val is not None:
                        precision, reward_vector = val
                    else:
                        precision = (
                            lambda_val * np.ones(d, dtype=np.float32)
                            if diagonal
                            else lambda_val * np.eye(d, dtype=np.float32)
                        )
                        reward_vector = np.zeros(d, dtype=np.float32)
                        reward_vector[-1] = lambda_val * prior_p

                    prior_reward_vector = np.zeros(d, dtype=np.float32)
                    prior_reward_vector[-1] = lambda_val * prior_p

                    if diagonal:
                        new_precision = (
                            decay_factor * precision
                            + (1.0 - decay_factor)
                            * lambda_val
                            * np.ones(d, dtype=np.float32)
                            + (x_augmented**2)
                        )
                    else:
                        new_precision = (
                            decay_factor * precision
                            + (1.0 - decay_factor)
                            * lambda_val
                            * np.eye(d, dtype=np.float32)
                            + np.outer(x_augmented, x_augmented)
                        )

                    new_reward_vector = (
                        decay_factor * reward_vector
                        + (1.0 - decay_factor) * prior_reward_vector
                        + reward * x_augmented
                    )

                    current_vals[candidate_name] = (new_precision, new_reward_vector)
                    results.append((np.copy(new_precision), np.copy(new_reward_vector)))

                pipe.multi()
                for c, (prec, rew) in current_vals.items():
                    pipe.set(
                        f"{self.prefix}linear:{c}:precision", json.dumps(prec.tolist())
                    )
                    pipe.set(
                        f"{self.prefix}linear:{c}:reward", json.dumps(rew.tolist())
                    )
                pipe.execute()
                return results
            except redis.WatchError:
                continue

    def save_vectors(self, vectors: Dict[str, Sequence[float]]) -> None:
        if not vectors:
            return
        pipe = self.client.pipeline()
        for k, v in vectors.items():
            pipe.hset(f"{self.prefix}context_vectors", key=k, value=json.dumps(list(v)))
        pipe.execute()

    def log_selection(
        self, trace_id: str, context_key: str, candidate_name: str
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        key = f"{self.prefix}log:{trace_id}"
        self.client.hset(
            key,
            mapping={
                "trace_id": trace_id,
                "timestamp": timestamp,
                "context_key": context_key,
                "candidate_name": candidate_name,
            },
        )
        self.client.sadd(f"{self.prefix}log_ids", trace_id)

    def log_feedback(self, trace_id: str, reward: float) -> None:
        key = f"{self.prefix}log:{trace_id}"
        if self.client.exists(key):
            self.client.hset(key, "reward", str(reward))

    def get_selection_logs(self) -> List[Dict[str, Any]]:
        trace_ids = self.client.smembers(f"{self.prefix}log_ids")
        logs = []
        if trace_ids:
            pipe = self.client.pipeline()
            tid_list = [
                tid.decode("utf-8") if isinstance(tid, bytes) else str(tid)
                for tid in trace_ids
            ]
            for tid_str in tid_list:
                key = f"{self.prefix}log:{tid_str}"
                pipe.hgetall(key)
            results = pipe.execute()
            for tid_str, data in zip(tid_list, results):
                if data:
                    decoded = {}
                    for k, v in data.items():
                        k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                        v_str = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                        decoded[k_str] = v_str

                    logs.append(
                        {
                            "trace_id": decoded.get("trace_id", tid_str),
                            "timestamp": decoded.get("timestamp", ""),
                            "context_key": decoded.get("context_key", ""),
                            "candidate_name": decoded.get("candidate_name", ""),
                            "reward": (
                                float(decoded["reward"])
                                if decoded.get("reward") is not None
                                else None
                            ),
                        }
                    )
        return sorted(logs, key=lambda x: x["timestamp"])


class AsyncBaseStorage(abc.ABC):
    """Abstract base class defining the async storage backend interface for BayesianCortex."""

    @abc.abstractmethod
    async def get_candidate_params(
        self, context_key: str, candidate_name: str
    ) -> Tuple[float, float]:
        """
        Retrieve the (alpha, beta) posterior parameters for a candidate under a given context.
        Defaults to (1.0, 1.0) if not found.
        """
        pass

    @abc.abstractmethod
    async def ahas_candidate_params(
        self, context_key: str, candidate_name: str
    ) -> bool:
        """
        Return True if the candidate has ever been observed under the given context
        (i.e. a record exists in storage), False otherwise.

        This must NOT be inferred from the parameter values alone. After a failure
        with decay_factor=1.0 the parameters legitimately remain at the floor (1.0, 1.0),
        which is identical to the cold-start default. Only an explicit existence check
        can distinguish the two states.
        """
        pass

    @abc.abstractmethod
    async def update_candidate_params(
        self, context_key: str, candidate_name: str, alpha: float, beta: float
    ) -> None:
        """
        Directly set the (alpha, beta) parameters for a candidate under a given context.
        """
        pass

    @abc.abstractmethod
    async def decay_and_update(
        self, context_key: str, candidate_name: str, decay_factor: float, reward: float
    ) -> Tuple[float, float]:
        """
        Atomically decay the current parameters and add the reward, ensuring
        they do not drop below the flat prior baseline of 1.0:
        alpha_new = max(1.0, alpha_old * decay_factor + reward)
        beta_new = max(1.0, beta_old * decay_factor + (1 - reward))

        Note: Clamping alpha and beta to a lower bound of 1.0 is critical for standard Thompson Sampling.
        If they were allowed to drop below 1.0 (e.g. under aggressive decay), the Beta probability density
        function (PDF) would turn into a bimodal U-shape. This would force random draws to spike violently
        to either absolute 0 or absolute 1, causing completely erratic routing behavior. Clamping protects
        the stability of the distribution's mode.
        """
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """Close any resources associated with the storage backend."""
        pass

    @abc.abstractmethod
    async def load_metadata(self, key: str) -> Optional[str]:
        """Retrieve stored metadata for a given key, or None if not found."""
        pass

    @abc.abstractmethod
    async def save_metadata(self, key: str, value: str) -> None:
        """Store metadata key-value pair."""
        pass

    async def load_all_vectors(self) -> Dict[str, List[float]]:
        """
        Retrieve all stored context vectors from the backend.
        Fallback implementation uses metadata for backwards compatibility.
        """
        try:
            serialized = await self.load_metadata("vector_context_store")
            if serialized:
                data = json.loads(serialized)
                return {k: list(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    async def save_vector(self, context_key: str, vector: Sequence[float]) -> None:
        """
        Store a single context vector incrementally.
        Fallback implementation updates the entire metadata JSON string.
        """
        try:
            serialized = await self.load_metadata("vector_context_store")
            if serialized:
                data = json.loads(serialized)
            else:
                data = {}
            data[context_key] = list(vector)
            await self.save_metadata("vector_context_store", json.dumps(data))
        except Exception:
            pass

    @abc.abstractmethod
    async def aget_linear_params(
        self, candidate_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieve (precision, reward_vector) for a candidate.
        Returns (None, None) if not found.
        """
        pass

    @abc.abstractmethod
    async def adecay_and_update_linear(
        self,
        candidate_name: str,
        decay_factor: float,
        reward: float,
        x_augmented: np.ndarray,
        lambda_val: float,
        prior_p: float,
        diagonal: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Atomically decay the current parameters for a candidate and add the new observation,
        returning the updated (precision, reward_vector).
        """
        pass

    async def get_candidate_params_batch(
        self, keys: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """
        Batch retrieve the (alpha, beta) posterior parameters for a list of context and candidate keys.
        """
        res = {}
        for key in keys:
            res[key] = await self.get_candidate_params(key[0], key[1])
        return res

    async def update_candidate_params_batch(
        self, params: Dict[Tuple[str, str], Tuple[float, float]]
    ) -> None:
        """
        Batch set the (alpha, beta) parameters.
        """
        for (ctx, cand), (alpha, beta) in params.items():
            await self.update_candidate_params(ctx, cand, alpha, beta)

    async def decay_and_update_batch(
        self, updates: List[Tuple[str, str, float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Batch decay and update parameters in order.
        Each update is (context_key, candidate_name, decay_factor, reward).
        """
        res = []
        for ctx, cand, decay, reward in updates:
            res.append(await self.decay_and_update(ctx, cand, decay, reward))
        return res

    async def aget_linear_params_batch(
        self, candidate_names: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Batch retrieve (precision, reward_vector) for multiple candidates.
        """
        results = {}
        for c in candidate_names:
            val = await self.aget_linear_params(c)
            if val[0] is not None and val[1] is not None:
                results[c] = val
        return results

    async def adecay_and_update_linear_batch(
        self, updates: List[Tuple[str, float, float, np.ndarray, float, float, bool]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Batch decay and update linear parameters in order.
        Each update is (candidate_name, decay_factor, reward, x_augmented, lambda_val, prior_p, diagonal).
        """
        res = []
        for cand, decay, reward, x_aug, lamb, prior, diag in updates:
            res.append(
                await self.adecay_and_update_linear(
                    cand, decay, reward, x_aug, lamb, prior, diag
                )
            )
        return res

    async def asave_vectors(self, vectors: Dict[str, Sequence[float]]) -> None:
        """
        Batch store context vectors incrementally.
        """
        for key, vector in vectors.items():
            await self.save_vector(key, vector)

    @abc.abstractmethod
    async def log_selection(
        self, trace_id: str, context_key: str, candidate_name: str
    ) -> None:
        """Log a candidate selection event."""
        pass

    @abc.abstractmethod
    async def log_feedback(self, trace_id: str, reward: float) -> None:
        """Log reward feedback for a selection event."""
        pass

    @abc.abstractmethod
    async def get_selection_logs(self) -> List[Dict[str, Any]]:
        """Retrieve all selection logs, ordered by timestamp ascending."""
        return []


class AsyncInMemoryStorage(AsyncBaseStorage):
    """
    Async-native in-memory thread-safe implementation of AsyncBaseStorage.
    """

    def __init__(self) -> None:
        self._data: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._metadata: Dict[str, str] = {}
        self._vectors: Dict[str, List[float]] = {}
        self._linear_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._selection_logs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def get_candidate_params(
        self, context_key: str, candidate_name: str
    ) -> Tuple[float, float]:
        async with self._lock:
            return self._data.get((context_key, candidate_name), (1.0, 1.0))

    async def ahas_candidate_params(
        self, context_key: str, candidate_name: str
    ) -> bool:
        async with self._lock:
            return (context_key, candidate_name) in self._data

    async def update_candidate_params(
        self, context_key: str, candidate_name: str, alpha: float, beta: float
    ) -> None:
        async with self._lock:
            self._data[(context_key, candidate_name)] = (alpha, beta)

    async def decay_and_update(
        self, context_key: str, candidate_name: str, decay_factor: float, reward: float
    ) -> Tuple[float, float]:
        async with self._lock:
            alpha, beta = self._data.get((context_key, candidate_name), (1.0, 1.0))
            new_alpha = max(1.0, alpha * decay_factor + reward)
            new_beta = max(1.0, beta * decay_factor + (1.0 - reward))
            self._data[(context_key, candidate_name)] = (new_alpha, new_beta)
            return new_alpha, new_beta

    async def close(self) -> None:
        pass

    async def load_metadata(self, key: str) -> Optional[str]:
        async with self._lock:
            return self._metadata.get(key)

    async def save_metadata(self, key: str, value: str) -> None:
        async with self._lock:
            self._metadata[key] = value

    async def load_all_vectors(self) -> Dict[str, List[float]]:
        async with self._lock:
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

    async def save_vector(self, context_key: str, vector: Sequence[float]) -> None:
        async with self._lock:
            self._vectors[context_key] = list(vector)

    async def aget_linear_params(
        self, candidate_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        async with self._lock:
            val = self._linear_data.get(candidate_name)
            if val is not None:
                return np.copy(val[0]), np.copy(val[1])
            return None, None

    async def adecay_and_update_linear(
        self,
        candidate_name: str,
        decay_factor: float,
        reward: float,
        x_augmented: np.ndarray,
        lambda_val: float,
        prior_p: float,
        diagonal: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        d = len(x_augmented)
        async with self._lock:
            val = self._linear_data.get(candidate_name)
            if val is not None:
                precision, reward_vector = val
            else:
                precision = (
                    lambda_val * np.ones(d, dtype=np.float32)
                    if diagonal
                    else lambda_val * np.eye(d, dtype=np.float32)
                )
                reward_vector = np.zeros(d, dtype=np.float32)
                reward_vector[-1] = lambda_val * prior_p

            prior_reward_vector = np.zeros(d, dtype=np.float32)
            prior_reward_vector[-1] = lambda_val * prior_p

            if diagonal:
                new_precision = (
                    decay_factor * precision
                    + (1.0 - decay_factor) * lambda_val * np.ones(d, dtype=np.float32)
                    + (x_augmented**2)
                )
            else:
                new_precision = (
                    decay_factor * precision
                    + (1.0 - decay_factor) * lambda_val * np.eye(d, dtype=np.float32)
                    + np.outer(x_augmented, x_augmented)
                )

            new_reward_vector = (
                decay_factor * reward_vector
                + (1.0 - decay_factor) * prior_reward_vector
                + reward * x_augmented
            )

            self._linear_data[candidate_name] = (new_precision, new_reward_vector)
            return np.copy(new_precision), np.copy(new_reward_vector)

    async def get_candidate_params_batch(
        self, keys: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        async with self._lock:
            return {key: self._data.get(key, (1.0, 1.0)) for key in keys}

    async def update_candidate_params_batch(
        self, params: Dict[Tuple[str, str], Tuple[float, float]]
    ) -> None:
        async with self._lock:
            for key, val in params.items():
                self._data[key] = val

    async def decay_and_update_batch(
        self, updates: List[Tuple[str, str, float, float]]
    ) -> List[Tuple[float, float]]:
        async with self._lock:
            results = []
            for context_key, candidate_name, decay_factor, reward in updates:
                alpha, beta = self._data.get((context_key, candidate_name), (1.0, 1.0))
                new_alpha = max(1.0, alpha * decay_factor + reward)
                new_beta = max(1.0, beta * decay_factor + (1.0 - reward))
                self._data[(context_key, candidate_name)] = (new_alpha, new_beta)
                results.append((new_alpha, new_beta))
            return results

    async def aget_linear_params_batch(
        self, candidate_names: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        async with self._lock:
            results = {}
            for c in candidate_names:
                val = self._linear_data.get(c)
                if val is not None:
                    results[c] = (np.copy(val[0]), np.copy(val[1]))
            return results

    async def adecay_and_update_linear_batch(
        self, updates: List[Tuple[str, float, float, np.ndarray, float, float, bool]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        async with self._lock:
            results = []
            for (
                candidate_name,
                decay_factor,
                reward,
                x_augmented,
                lambda_val,
                prior_p,
                diagonal,
            ) in updates:
                d = len(x_augmented)
                val = self._linear_data.get(candidate_name)
                if val is not None:
                    precision, reward_vector = val
                else:
                    precision = (
                        lambda_val * np.ones(d, dtype=np.float32)
                        if diagonal
                        else lambda_val * np.eye(d, dtype=np.float32)
                    )
                    reward_vector = np.zeros(d, dtype=np.float32)
                    reward_vector[-1] = lambda_val * prior_p

                prior_reward_vector = np.zeros(d, dtype=np.float32)
                prior_reward_vector[-1] = lambda_val * prior_p

                if diagonal:
                    new_precision = (
                        decay_factor * precision
                        + (1.0 - decay_factor)
                        * lambda_val
                        * np.ones(d, dtype=np.float32)
                        + (x_augmented**2)
                    )
                else:
                    new_precision = (
                        decay_factor * precision
                        + (1.0 - decay_factor)
                        * lambda_val
                        * np.eye(d, dtype=np.float32)
                        + np.outer(x_augmented, x_augmented)
                    )

                new_reward_vector = (
                    decay_factor * reward_vector
                    + (1.0 - decay_factor) * prior_reward_vector
                    + reward * x_augmented
                )

                self._linear_data[candidate_name] = (new_precision, new_reward_vector)
                results.append((np.copy(new_precision), np.copy(new_reward_vector)))
            return results

    async def asave_vectors(self, vectors: Dict[str, Sequence[float]]) -> None:
        async with self._lock:
            for key, vector in vectors.items():
                self._vectors[key] = list(vector)

    async def log_selection(
        self, trace_id: str, context_key: str, candidate_name: str
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            if trace_id not in self._selection_logs:
                self._selection_logs[trace_id] = {
                    "trace_id": trace_id,
                    "timestamp": timestamp,
                    "context_key": context_key,
                    "candidate_name": candidate_name,
                    "reward": None,
                }

    async def log_feedback(self, trace_id: str, reward: float) -> None:
        async with self._lock:
            if trace_id in self._selection_logs:
                self._selection_logs[trace_id]["reward"] = reward

    async def get_selection_logs(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return sorted(self._selection_logs.values(), key=lambda x: x["timestamp"])


import contextlib


class AsyncSQLiteConnectionPool:
    """
    Lightweight connection pool for aiosqlite connections.
    """

    def __init__(self, db_path: str, max_size: int = 10, timeout: float = 5.0) -> None:
        self.db_path = db_path
        self.max_size = max_size
        self.timeout = timeout
        self._pool: asyncio.Queue = asyncio.Queue()
        self._created = 0
        self._lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def connection(self):
        conn = None
        async with self._lock:
            if not self._pool.empty():
                conn = self._pool.get_nowait()
            elif self._created < self.max_size:
                try:
                    import aiosqlite
                except ImportError:
                    raise ImportError(
                        "aiosqlite is required for AsyncSQLiteStorage. "
                        "Please install it with: pip install aiosqlite"
                    )
                conn = await aiosqlite.connect(self.db_path)
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)};")
                await conn.commit()
                self._created += 1

        if conn is None:
            conn = await self._pool.get()

        try:
            yield conn
        finally:
            if conn is not None:
                try:
                    await conn.rollback()
                except Exception:
                    pass
                await self._pool.put(conn)

    async def close_all(self) -> None:
        async with self._lock:
            while not self._pool.empty():
                conn = self._pool.get_nowait()
                await conn.close()
            self._created = 0


class AsyncSQLiteStorage(AsyncBaseStorage):
    """
    SQLite-backed storage for persistent local storage with async support.
    Uses a connection pool and lock-free retries with exponential backoff and jitter
    to handle high concurrency workloads across different threads, tasks, and processes.
    """

    def __init__(
        self,
        db_path: str = "bayesian_cortex.db",
        max_connections: int = 10,
        timeout: float = 5.0,
    ) -> None:
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool = AsyncSQLiteConnectionPool(
            db_path, max_size=max_connections, timeout=timeout
        )
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._conn: Optional[Any] = None
        self._lock = asyncio.Lock()

    async def _get_conn(self) -> Any:
        """
        Deprecated. Returns a connection from the pool.
        """
        if self._conn is None:
            import aiosqlite

            self._conn = await aiosqlite.connect(self.db_path)
            await self._conn.execute("PRAGMA journal_mode=WAL;")
            await self._conn.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)};")
            await self._conn.commit()
        return self._conn

    async def _execute_with_retry(
        self,
        func,
        *args,
        max_retries: int = 5,
        initial_delay: float = 0.05,
        max_delay: float = 1.0,
        **kwargs,
    ):
        import random
        import sqlite3

        try:
            import aiosqlite

            sqlite_errors = (sqlite3.OperationalError, aiosqlite.OperationalError)
        except ImportError:
            sqlite_errors = (sqlite3.OperationalError,)

        delay = initial_delay
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except sqlite_errors as e:
                if attempt == max_retries or "locked" not in str(e).lower():
                    raise
                sleep_time = delay * (0.5 + random.random())
                await asyncio.sleep(sleep_time)
                delay = min(delay * 2, max_delay)

    async def _migrate_database(self, conn) -> None:
        # Check if legacy tool_params exists
        async with conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='tool_params';"
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0] > 0:
                await conn.execute(
                    "ALTER TABLE tool_params RENAME TO candidate_params;"
                )
                await conn.execute(
                    "ALTER TABLE candidate_params RENAME COLUMN tool_name TO candidate_name;"
                )

        # Check if linear_bandit_params exists and has tool_name column
        async with conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='linear_bandit_params';"
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0] > 0:
                async with conn.execute(
                    "PRAGMA table_info(linear_bandit_params);"
                ) as info_cursor:
                    columns = [r[1] for r in await info_cursor.fetchall()]
                    if "tool_name" in columns:
                        await conn.execute(
                            "ALTER TABLE linear_bandit_params RENAME COLUMN tool_name TO candidate_name;"
                        )

        # Check if selection_log exists and has tool_name column
        async with conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='selection_log';"
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0] > 0:
                async with conn.execute(
                    "PRAGMA table_info(selection_log);"
                ) as info_cursor:
                    columns = [r[1] for r in await info_cursor.fetchall()]
                    if "tool_name" in columns:
                        await conn.execute(
                            "ALTER TABLE selection_log RENAME COLUMN tool_name TO candidate_name;"
                        )

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with self._pool.connection() as conn:
                await self._migrate_database(conn)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS candidate_params (
                        context_key TEXT,
                        candidate_name TEXT,
                        alpha REAL,
                        beta REAL,
                        PRIMARY KEY (context_key, candidate_name)
                    )
                    """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        val TEXT
                    )
                    """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS context_vectors (
                        context_key TEXT PRIMARY KEY,
                        vector TEXT
                    )
                    """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS linear_bandit_params (
                        candidate_name TEXT PRIMARY KEY,
                        precision_matrix TEXT,
                        reward_vector TEXT
                    )
                    """)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS selection_log (
                        trace_id TEXT PRIMARY KEY,
                        timestamp TEXT,
                        context_key TEXT,
                        candidate_name TEXT,
                        reward REAL
                    )
                    """)
                await conn.commit()
            self._initialized = True

    async def get_candidate_params(
        self, context_key: str, candidate_name: str
    ) -> Tuple[float, float]:
        await self._ensure_initialized()

        async def _run():
            async with self._pool.connection() as conn:
                async with conn.execute(
                    "SELECT alpha, beta FROM candidate_params WHERE context_key = ? AND candidate_name = ?",
                    (context_key, candidate_name),
                ) as cursor:
                    row = await cursor.fetchone()
                    if row is not None:
                        return float(row[0]), float(row[1])
                    return 1.0, 1.0

        return await self._execute_with_retry(_run)

    async def ahas_candidate_params(
        self, context_key: str, candidate_name: str
    ) -> bool:
        await self._ensure_initialized()

        async def _run():
            async with self._pool.connection() as conn:
                async with conn.execute(
                    "SELECT 1 FROM candidate_params WHERE context_key = ? AND candidate_name = ? LIMIT 1",
                    (context_key, candidate_name),
                ) as cursor:
                    row = await cursor.fetchone()
                    return row is not None

        return await self._execute_with_retry(_run)

    async def update_candidate_params(
        self, context_key: str, candidate_name: str, alpha: float, beta: float
    ) -> None:
        await self._ensure_initialized()

        async def _run():
            async with self._pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO candidate_params (context_key, candidate_name, alpha, beta)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(context_key, candidate_name) DO UPDATE SET
                        alpha = excluded.alpha,
                        beta = excluded.beta
                    """,
                    (context_key, candidate_name, alpha, beta),
                )
                await conn.commit()

        await self._execute_with_retry(_run)

    async def decay_and_update(
        self, context_key: str, candidate_name: str, decay_factor: float, reward: float
    ) -> Tuple[float, float]:
        await self._ensure_initialized()

        async def _run():
            async with self._pool.connection() as conn:
                try:
                    await conn.execute("BEGIN IMMEDIATE")
                    async with conn.execute(
                        "SELECT alpha, beta FROM candidate_params WHERE context_key = ? AND candidate_name = ?",
                        (context_key, candidate_name),
                    ) as cursor:
                        row = await cursor.fetchone()
                        if row is not None:
                            alpha, beta = float(row[0]), float(row[1])
                        else:
                            alpha, beta = 1.0, 1.0

                    new_alpha = max(1.0, alpha * decay_factor + reward)
                    new_beta = max(1.0, beta * decay_factor + (1.0 - reward))

                    await conn.execute(
                        """
                        INSERT INTO candidate_params (context_key, candidate_name, alpha, beta)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(context_key, candidate_name) DO UPDATE SET
                            alpha = excluded.alpha,
                            beta = excluded.beta
                        """,
                        (context_key, candidate_name, new_alpha, new_beta),
                    )
                    await conn.commit()
                    return new_alpha, new_beta
                except Exception:
                    await conn.rollback()
                    raise

        return await self._execute_with_retry(_run)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        await self._pool.close_all()

    async def load_metadata(self, key: str) -> Optional[str]:
        await self._ensure_initialized()

        async def _run():
            async with self._pool.connection() as conn:
                async with conn.execute(
                    "SELECT val FROM metadata WHERE key = ?", (key,)
                ) as cursor:
                    row = await cursor.fetchone()
                    return row[0] if row is not None else None

        return await self._execute_with_retry(_run)

    async def save_metadata(self, key: str, value: str) -> None:
        await self._ensure_initialized()

        async def _run():
            async with self._pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO metadata (key, val) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET val = excluded.val
                    """,
                    (key, value),
                )
                await conn.commit()

        await self._execute_with_retry(_run)

    async def load_all_vectors(self) -> Dict[str, List[float]]:
        await self._ensure_initialized()

        async def _run():
            async with self._pool.connection() as conn:
                async with conn.execute(
                    "SELECT context_key, vector FROM context_vectors"
                ) as cursor:
                    rows = await cursor.fetchall()

                if not rows:
                    # Migration check
                    serialized = await self.load_metadata("vector_context_store")
                    if serialized:
                        try:
                            data = json.loads(serialized)
                            for k, v in data.items():
                                await conn.execute(
                                    """
                                    INSERT OR IGNORE INTO context_vectors (context_key, vector)
                                    VALUES (?, ?)
                                    """,
                                    (k, json.dumps(v)),
                                )
                            await conn.commit()
                            # Query again
                            async with conn.execute(
                                "SELECT context_key, vector FROM context_vectors"
                            ) as cursor2:
                                rows = await cursor2.fetchall()
                        except Exception:
                            pass

                res = {}
                for row in rows:
                    res[row[0]] = json.loads(row[1])
                return res

        return await self._execute_with_retry(_run)

    async def save_vector(self, context_key: str, vector: Sequence[float]) -> None:
        await self._ensure_initialized()

        async def _run():
            async with self._pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO context_vectors (context_key, vector)
                    VALUES (?, ?)
                    ON CONFLICT(context_key) DO UPDATE SET vector = excluded.vector
                    """,
                    (context_key, json.dumps(list(vector))),
                )
                await conn.commit()

        await self._execute_with_retry(_run)

    async def aget_linear_params(
        self, candidate_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        await self._ensure_initialized()

        async def _run():
            async with self._pool.connection() as conn:
                async with conn.execute(
                    "SELECT precision_matrix, reward_vector FROM linear_bandit_params WHERE candidate_name = ?",
                    (candidate_name,),
                ) as cursor:
                    row = await cursor.fetchone()
                    if row is not None:
                        precision = np.array(json.loads(row[0]), dtype=np.float32)
                        reward_vector = np.array(json.loads(row[1]), dtype=np.float32)
                        return precision, reward_vector
                    return None, None

        return await self._execute_with_retry(_run)

    async def adecay_and_update_linear(
        self,
        candidate_name: str,
        decay_factor: float,
        reward: float,
        x_augmented: np.ndarray,
        lambda_val: float,
        prior_p: float,
        diagonal: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        await self._ensure_initialized()
        d = len(x_augmented)

        async def _run():
            async with self._pool.connection() as conn:
                try:
                    await conn.execute("BEGIN IMMEDIATE")
                    async with conn.execute(
                        "SELECT precision_matrix, reward_vector FROM linear_bandit_params WHERE candidate_name = ?",
                        (candidate_name,),
                    ) as cursor:
                        row = await cursor.fetchone()
                    if row is not None:
                        precision = np.array(json.loads(row[0]), dtype=np.float32)
                        reward_vector = np.array(json.loads(row[1]), dtype=np.float32)
                    else:
                        precision = (
                            lambda_val * np.ones(d, dtype=np.float32)
                            if diagonal
                            else lambda_val * np.eye(d, dtype=np.float32)
                        )
                        reward_vector = np.zeros(d, dtype=np.float32)
                        reward_vector[-1] = lambda_val * prior_p

                    prior_reward_vector = np.zeros(d, dtype=np.float32)
                    prior_reward_vector[-1] = lambda_val * prior_p

                    if diagonal:
                        new_precision = (
                            decay_factor * precision
                            + (1.0 - decay_factor)
                            * lambda_val
                            * np.ones(d, dtype=np.float32)
                            + (x_augmented**2)
                        )
                    else:
                        new_precision = (
                            decay_factor * precision
                            + (1.0 - decay_factor)
                            * lambda_val
                            * np.eye(d, dtype=np.float32)
                            + np.outer(x_augmented, x_augmented)
                        )

                    new_reward_vector = (
                        decay_factor * reward_vector
                        + (1.0 - decay_factor) * prior_reward_vector
                        + reward * x_augmented
                    )

                    await conn.execute(
                        """
                        INSERT INTO linear_bandit_params (candidate_name, precision_matrix, reward_vector)
                        VALUES (?, ?, ?)
                        ON CONFLICT(candidate_name) DO UPDATE SET
                            precision_matrix = excluded.precision_matrix,
                            reward_vector = excluded.reward_vector
                        """,
                        (
                            candidate_name,
                            json.dumps(new_precision.tolist()),
                            json.dumps(new_reward_vector.tolist()),
                        ),
                    )
                    await conn.commit()
                    return new_precision, new_reward_vector
                except Exception:
                    await conn.rollback()
                    raise

        return await self._execute_with_retry(_run)

    async def get_candidate_params_batch(
        self, keys: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        await self._ensure_initialized()
        if not keys:
            return {}

        async def _run():
            results = dict.fromkeys(keys, (1.0, 1.0))
            async with self._pool.connection() as conn:
                chunk_size = 200
                for i in range(0, len(keys), chunk_size):
                    chunk = keys[i : i + chunk_size]
                    clauses = []
                    params = []
                    for c_key, cand_name in chunk:
                        clauses.append("(context_key = ? AND candidate_name = ?)")
                        params.extend([c_key, cand_name])
                    query = (
                        "SELECT context_key, candidate_name, alpha, beta FROM candidate_params WHERE "
                        + " OR ".join(clauses)
                    )
                    async with conn.execute(query, params) as cursor:
                        rows = await cursor.fetchall()
                        for row in rows:
                            results[(row[0], row[1])] = (float(row[2]), float(row[3]))
            return results

        return await self._execute_with_retry(_run)

    async def update_candidate_params_batch(
        self, params: Dict[Tuple[str, str], Tuple[float, float]]
    ) -> None:
        await self._ensure_initialized()
        if not params:
            return

        async def _run():
            async with self._pool.connection() as conn:
                await conn.executemany(
                    """
                    INSERT INTO candidate_params (context_key, candidate_name, alpha, beta)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(context_key, candidate_name) DO UPDATE SET
                        alpha = excluded.alpha,
                        beta = excluded.beta
                    """,
                    [
                        (ctx, cand, alpha, beta)
                        for (ctx, cand), (alpha, beta) in params.items()
                    ],
                )
                await conn.commit()

        await self._execute_with_retry(_run)

    async def decay_and_update_batch(
        self, updates: List[Tuple[str, str, float, float]]
    ) -> List[Tuple[float, float]]:
        await self._ensure_initialized()
        if not updates:
            return []

        async def _run():
            async with self._pool.connection() as conn:
                try:
                    await conn.execute("BEGIN IMMEDIATE")

                    keys = [(ctx, cand) for ctx, cand, _, _ in updates]
                    current_vals = {}
                    chunk_size = 200
                    for i in range(0, len(keys), chunk_size):
                        chunk = keys[i : i + chunk_size]
                        clauses = []
                        params = []
                        for c_key, cand_name in chunk:
                            clauses.append("(context_key = ? AND candidate_name = ?)")
                            params.extend([c_key, cand_name])
                        query = (
                            "SELECT context_key, candidate_name, alpha, beta FROM candidate_params WHERE "
                            + " OR ".join(clauses)
                        )
                        async with conn.execute(query, params) as cursor:
                            rows = await cursor.fetchall()
                            for row in rows:
                                current_vals[(row[0], row[1])] = (
                                    float(row[2]),
                                    float(row[3]),
                                )

                    updated_params = []
                    for ctx, cand, decay_factor, reward in updates:
                        alpha, beta = current_vals.get((ctx, cand), (1.0, 1.0))
                        new_alpha = max(1.0, alpha * decay_factor + reward)
                        new_beta = max(1.0, beta * decay_factor + (1.0 - reward))
                        current_vals[(ctx, cand)] = (new_alpha, new_beta)
                        updated_params.append((ctx, cand, new_alpha, new_beta))

                    db_updates = [
                        (ctx, cand, val[0], val[1])
                        for (ctx, cand), val in current_vals.items()
                    ]
                    await conn.executemany(
                        """
                        INSERT INTO candidate_params (context_key, candidate_name, alpha, beta)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(context_key, candidate_name) DO UPDATE SET
                            alpha = excluded.alpha,
                            beta = excluded.beta
                        """,
                        db_updates,
                    )
                    await conn.commit()
                    return [(item[2], item[3]) for item in updated_params]
                except Exception:
                    await conn.rollback()
                    raise

        return await self._execute_with_retry(_run)

    async def aget_linear_params_batch(
        self, candidate_names: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        await self._ensure_initialized()
        if not candidate_names:
            return {}

        async def _run():
            async with self._pool.connection() as conn:
                placeholders = ",".join(["?"] * len(candidate_names))
                async with conn.execute(
                    f"SELECT candidate_name, precision_matrix, reward_vector FROM linear_bandit_params WHERE candidate_name IN ({placeholders})",
                    candidate_names,
                ) as cursor:
                    rows = await cursor.fetchall()
                results = {}
                for row in rows:
                    precision = np.array(json.loads(row[1]), dtype=np.float32)
                    reward_vector = np.array(json.loads(row[2]), dtype=np.float32)
                    results[row[0]] = (precision, reward_vector)
                return results

        return await self._execute_with_retry(_run)

    async def adecay_and_update_linear_batch(
        self, updates: List[Tuple[str, float, float, np.ndarray, float, float, bool]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        await self._ensure_initialized()
        if not updates:
            return []

        async def _run():
            async with self._pool.connection() as conn:
                try:
                    await conn.execute("BEGIN IMMEDIATE")

                    candidate_names = list(set([item[0] for item in updates]))
                    current_vals = {}
                    if candidate_names:
                        placeholders = ",".join(["?"] * len(candidate_names))
                        query = f"SELECT candidate_name, precision_matrix, reward_vector FROM linear_bandit_params WHERE candidate_name IN ({placeholders})"
                        async with conn.execute(query, candidate_names) as cursor:
                            rows = await cursor.fetchall()
                            for row in rows:
                                precision = np.array(
                                    json.loads(row[1]), dtype=np.float32
                                )
                                reward_vector = np.array(
                                    json.loads(row[2]), dtype=np.float32
                                )
                                current_vals[row[0]] = (precision, reward_vector)

                    results = []
                    for (
                        candidate_name,
                        decay_factor,
                        reward,
                        x_augmented,
                        lambda_val,
                        prior_p,
                        diagonal,
                    ) in updates:
                        d = len(x_augmented)
                        val = current_vals.get(candidate_name)
                        if val is not None:
                            precision, reward_vector = val
                        else:
                            precision = (
                                lambda_val * np.ones(d, dtype=np.float32)
                                if diagonal
                                else lambda_val * np.eye(d, dtype=np.float32)
                            )
                            reward_vector = np.zeros(d, dtype=np.float32)
                            reward_vector[-1] = lambda_val * prior_p

                        prior_reward_vector = np.zeros(d, dtype=np.float32)
                        prior_reward_vector[-1] = lambda_val * prior_p

                        if diagonal:
                            new_precision = (
                                decay_factor * precision
                                + (1.0 - decay_factor)
                                * lambda_val
                                * np.ones(d, dtype=np.float32)
                                + (x_augmented**2)
                            )
                        else:
                            new_precision = (
                                decay_factor * precision
                                + (1.0 - decay_factor)
                                * lambda_val
                                * np.eye(d, dtype=np.float32)
                                + np.outer(x_augmented, x_augmented)
                            )

                        new_reward_vector = (
                            decay_factor * reward_vector
                            + (1.0 - decay_factor) * prior_reward_vector
                            + reward * x_augmented
                        )

                        current_vals[candidate_name] = (
                            new_precision,
                            new_reward_vector,
                        )
                        results.append(
                            (np.copy(new_precision), np.copy(new_reward_vector))
                        )

                    db_updates = []
                    for c_name, (prec, rew) in current_vals.items():
                        db_updates.append(
                            (
                                c_name,
                                json.dumps(prec.tolist()),
                                json.dumps(rew.tolist()),
                            )
                        )

                    await conn.executemany(
                        """
                        INSERT INTO linear_bandit_params (candidate_name, precision_matrix, reward_vector)
                        VALUES (?, ?, ?)
                        ON CONFLICT(candidate_name) DO UPDATE SET
                            precision_matrix = excluded.precision_matrix,
                            reward_vector = excluded.reward_vector
                        """,
                        db_updates,
                    )
                    await conn.commit()
                    return results
                except Exception:
                    await conn.rollback()
                    raise

        return await self._execute_with_retry(_run)

    async def asave_vectors(self, vectors: Dict[str, Sequence[float]]) -> None:
        await self._ensure_initialized()
        if not vectors:
            return

        async def _run():
            async with self._pool.connection() as conn:
                await conn.executemany(
                    """
                    INSERT INTO context_vectors (context_key, vector)
                    VALUES (?, ?)
                    ON CONFLICT(context_key) DO UPDATE SET vector = excluded.vector
                    """,
                    [(k, json.dumps(list(v))) for k, v in vectors.items()],
                )
                await conn.commit()

        await self._execute_with_retry(_run)

    async def log_selection(
        self, trace_id: str, context_key: str, candidate_name: str
    ) -> None:
        await self._ensure_initialized()
        timestamp = datetime.now(timezone.utc).isoformat()

        async def _run():
            async with self._pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT OR IGNORE INTO selection_log (trace_id, timestamp, context_key, candidate_name, reward)
                    VALUES (?, ?, ?, ?, NULL)
                    """,
                    (trace_id, timestamp, context_key, candidate_name),
                )
                await conn.commit()

        await self._execute_with_retry(_run)

    async def log_feedback(self, trace_id: str, reward: float) -> None:
        await self._ensure_initialized()

        async def _run():
            async with self._pool.connection() as conn:
                await conn.execute(
                    "UPDATE selection_log SET reward = ? WHERE trace_id = ?",
                    (reward, trace_id),
                )
                await conn.commit()

        await self._execute_with_retry(_run)

    async def get_selection_logs(self) -> List[Dict[str, Any]]:
        await self._ensure_initialized()

        async def _run():
            async with self._pool.connection() as conn:
                async with conn.execute(
                    "SELECT trace_id, timestamp, context_key, candidate_name, reward FROM selection_log ORDER BY timestamp ASC"
                ) as cursor:
                    logs = []
                    async for row in cursor:
                        logs.append(
                            {
                                "trace_id": row[0],
                                "timestamp": row[1],
                                "context_key": row[2],
                                "candidate_name": row[3],
                                "reward": float(row[4]) if row[4] is not None else None,
                            }
                        )
                    return logs

        return await self._execute_with_retry(_run)


class AsyncRedisStorage(AsyncBaseStorage):
    """
    Redis-backed storage backend with async support.
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

    local new_alpha = math.max(1.0, alpha * decay + reward)
    local new_beta = math.max(1.0, beta * decay + reward_fail)

    redis.call('HSET', key, field_alpha, tostring(new_alpha), field_beta, tostring(new_beta))
    return {tostring(new_alpha), tostring(new_beta)}
    """

    def __init__(self, redis_client: Any, prefix: str = "bayesian_cortex:") -> None:
        self.client = redis_client
        self.prefix = prefix
        self._script = self.client.register_script(self.LUA_DECAY_UPDATE)

    def _get_key(self, context_key: str) -> str:
        return f"{self.prefix}{context_key}"

    async def get_candidate_params(
        self, context_key: str, candidate_name: str
    ) -> Tuple[float, float]:
        key = self._get_key(context_key)
        alpha_val = await self.client.hget(key, f"{candidate_name}:alpha")
        beta_val = await self.client.hget(key, f"{candidate_name}:beta")

        alpha = float(alpha_val) if alpha_val is not None else 1.0
        beta = float(beta_val) if beta_val is not None else 1.0
        return alpha, beta

    async def ahas_candidate_params(
        self, context_key: str, candidate_name: str
    ) -> bool:
        key = self._get_key(context_key)
        return bool(await self.client.hexists(key, f"{candidate_name}:alpha"))

    async def update_candidate_params(
        self, context_key: str, candidate_name: str, alpha: float, beta: float
    ) -> None:
        key = self._get_key(context_key)
        await self.client.hset(
            key,
            mapping={
                f"{candidate_name}:alpha": str(alpha),
                f"{candidate_name}:beta": str(beta),
            },
        )

    async def decay_and_update(
        self, context_key: str, candidate_name: str, decay_factor: float, reward: float
    ) -> Tuple[float, float]:
        key = self._get_key(context_key)
        res = await self._script(
            keys=[key],
            args=[
                f"{candidate_name}:alpha",
                f"{candidate_name}:beta",
                str(decay_factor),
                str(reward),
            ],
        )
        return float(res[0]), float(res[1])

    async def close(self) -> None:
        pass

    async def load_metadata(self, key: str) -> Optional[str]:
        val = await self.client.get(f"{self.prefix}metadata:{key}")
        if val is None:
            return None
        return val.decode("utf-8") if isinstance(val, bytes) else str(val)

    async def save_metadata(self, key: str, value: str) -> None:
        await self.client.set(f"{self.prefix}metadata:{key}", value)

    async def load_all_vectors(self) -> Dict[str, List[float]]:
        vectors_hash = await self.client.hgetall(f"{self.prefix}context_vectors")
        if not vectors_hash:
            # Check for legacy metadata to migrate
            serialized = await self.load_metadata("vector_context_store")
            if serialized:
                try:
                    data = json.loads(serialized)
                    mapping = {k: json.dumps(v) for k, v in data.items()}
                    if mapping:
                        await self.client.hset(
                            f"{self.prefix}context_vectors", mapping=mapping
                        )
                    return data
                except Exception:
                    pass
        res = {}
        for k, v in vectors_hash.items():
            key_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            val_str = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            res[key_str] = json.loads(val_str)
        return res

    async def save_vector(self, context_key: str, vector: Sequence[float]) -> None:
        await self.client.hset(
            f"{self.prefix}context_vectors",
            key=context_key,
            value=json.dumps(list(vector)),
        )

    async def aget_linear_params(
        self, candidate_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        key_prec = f"{self.prefix}linear:{candidate_name}:precision"
        key_rew = f"{self.prefix}linear:{candidate_name}:reward"
        prec_val = await self.client.get(key_prec)
        rew_val = await self.client.get(key_rew)
        if prec_val is not None and rew_val is not None:
            precision = np.array(json.loads(prec_val), dtype=np.float32)
            reward_vector = np.array(json.loads(rew_val), dtype=np.float32)
            return precision, reward_vector
        return None, None

    async def adecay_and_update_linear(
        self,
        candidate_name: str,
        decay_factor: float,
        reward: float,
        x_augmented: np.ndarray,
        lambda_val: float,
        prior_p: float,
        diagonal: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        import redis

        d = len(x_augmented)
        key_prec = f"{self.prefix}linear:{candidate_name}:precision"
        key_rew = f"{self.prefix}linear:{candidate_name}:reward"
        pipe = self.client.pipeline()
        while True:
            try:
                await pipe.watch(key_prec, key_rew)
                prec_val = await self.client.get(key_prec)
                rew_val = await self.client.get(key_rew)
                if prec_val is not None and rew_val is not None:
                    precision = np.array(json.loads(prec_val), dtype=np.float32)
                    reward_vector = np.array(json.loads(rew_val), dtype=np.float32)
                else:
                    precision = (
                        lambda_val * np.ones(d, dtype=np.float32)
                        if diagonal
                        else lambda_val * np.eye(d, dtype=np.float32)
                    )
                    reward_vector = np.zeros(d, dtype=np.float32)
                    reward_vector[-1] = lambda_val * prior_p

                prior_reward_vector = np.zeros(d, dtype=np.float32)
                prior_reward_vector[-1] = lambda_val * prior_p

                if diagonal:
                    new_precision = (
                        decay_factor * precision
                        + (1.0 - decay_factor)
                        * lambda_val
                        * np.ones(d, dtype=np.float32)
                        + (x_augmented**2)
                    )
                else:
                    new_precision = (
                        decay_factor * precision
                        + (1.0 - decay_factor)
                        * lambda_val
                        * np.eye(d, dtype=np.float32)
                        + np.outer(x_augmented, x_augmented)
                    )

                new_reward_vector = (
                    decay_factor * reward_vector
                    + (1.0 - decay_factor) * prior_reward_vector
                    + reward * x_augmented
                )

                pipe.multi()
                pipe.set(key_prec, json.dumps(new_precision.tolist()))
                pipe.set(key_rew, json.dumps(new_reward_vector.tolist()))
                await pipe.execute()
                return new_precision, new_reward_vector
            except redis.WatchError:
                continue

    async def get_candidate_params_batch(
        self, keys: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Tuple[float, float]]:
        if not keys:
            return {}
        pipe = self.client.pipeline()
        for context_key, candidate_name in keys:
            key = self._get_key(context_key)
            pipe.hget(key, f"{candidate_name}:alpha")
            pipe.hget(key, f"{candidate_name}:beta")

        results_raw = await pipe.execute()
        results = {}
        for idx, (context_key, candidate_name) in enumerate(keys):
            alpha_val = results_raw[2 * idx]
            beta_val = results_raw[2 * idx + 1]
            alpha = float(alpha_val) if alpha_val is not None else 1.0
            beta = float(beta_val) if beta_val is not None else 1.0
            results[(context_key, candidate_name)] = (alpha, beta)
        return results

    async def update_candidate_params_batch(
        self, params: Dict[Tuple[str, str], Tuple[float, float]]
    ) -> None:
        if not params:
            return
        pipe = self.client.pipeline()
        for (context_key, candidate_name), (alpha, beta) in params.items():
            key = self._get_key(context_key)
            pipe.hset(
                key,
                mapping={
                    f"{candidate_name}:alpha": str(alpha),
                    f"{candidate_name}:beta": str(beta),
                },
            )
        await pipe.execute()

    async def decay_and_update_batch(
        self, updates: List[Tuple[str, str, float, float]]
    ) -> List[Tuple[float, float]]:
        if not updates:
            return []
        pipe = self.client.pipeline()
        for context_key, candidate_name, decay_factor, reward in updates:
            key = self._get_key(context_key)
            await self._script(
                keys=[key],
                args=[
                    f"{candidate_name}:alpha",
                    f"{candidate_name}:beta",
                    str(decay_factor),
                    str(reward),
                ],
                client=pipe,
            )
        raw_results = await pipe.execute()
        return [(float(res[0]), float(res[1])) for res in raw_results]

    async def aget_linear_params_batch(
        self, candidate_names: List[str]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if not candidate_names:
            return {}
        pipe = self.client.pipeline()
        for c in candidate_names:
            pipe.get(f"{self.prefix}linear:{c}:precision")
            pipe.get(f"{self.prefix}linear:{c}:reward")
        raw_vals = await pipe.execute()
        results = {}
        for idx, c in enumerate(candidate_names):
            prec_val = raw_vals[2 * idx]
            rew_val = raw_vals[2 * idx + 1]
            if prec_val is not None and rew_val is not None:
                precision = np.array(json.loads(prec_val), dtype=np.float32)
                reward_vector = np.array(json.loads(rew_val), dtype=np.float32)
                results[c] = (precision, reward_vector)
        return results

    async def adecay_and_update_linear_batch(
        self, updates: List[Tuple[str, float, float, np.ndarray, float, float, bool]]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if not updates:
            return []
        import redis

        candidate_names = list(set([item[0] for item in updates]))
        keys_prec = [f"{self.prefix}linear:{c}:precision" for c in candidate_names]
        keys_rew = [f"{self.prefix}linear:{c}:reward" for c in candidate_names]
        all_keys = keys_prec + keys_rew

        pipe = self.client.pipeline()
        while True:
            try:
                await pipe.watch(*all_keys)
                for k in keys_prec:
                    pipe.get(k)
                for k in keys_rew:
                    pipe.get(k)
                raw_vals = await pipe.execute()

                current_vals = {}
                num_cands = len(candidate_names)
                for idx, c in enumerate(candidate_names):
                    prec_val = raw_vals[idx]
                    rew_val = raw_vals[num_cands + idx]
                    if prec_val is not None and rew_val is not None:
                        precision = np.array(json.loads(prec_val), dtype=np.float32)
                        reward_vector = np.array(json.loads(rew_val), dtype=np.float32)
                        current_vals[c] = (precision, reward_vector)

                results = []
                for (
                    candidate_name,
                    decay_factor,
                    reward,
                    x_augmented,
                    lambda_val,
                    prior_p,
                    diagonal,
                ) in updates:
                    d = len(x_augmented)
                    val = current_vals.get(candidate_name)
                    if val is not None:
                        precision, reward_vector = val
                    else:
                        precision = (
                            lambda_val * np.ones(d, dtype=np.float32)
                            if diagonal
                            else lambda_val * np.eye(d, dtype=np.float32)
                        )
                        reward_vector = np.zeros(d, dtype=np.float32)
                        reward_vector[-1] = lambda_val * prior_p

                    prior_reward_vector = np.zeros(d, dtype=np.float32)
                    prior_reward_vector[-1] = lambda_val * prior_p

                    if diagonal:
                        new_precision = (
                            decay_factor * precision
                            + (1.0 - decay_factor)
                            * lambda_val
                            * np.ones(d, dtype=np.float32)
                            + (x_augmented**2)
                        )
                    else:
                        new_precision = (
                            decay_factor * precision
                            + (1.0 - decay_factor)
                            * lambda_val
                            * np.eye(d, dtype=np.float32)
                            + np.outer(x_augmented, x_augmented)
                        )

                    new_reward_vector = (
                        decay_factor * reward_vector
                        + (1.0 - decay_factor) * prior_reward_vector
                        + reward * x_augmented
                    )

                    current_vals[candidate_name] = (new_precision, new_reward_vector)
                    results.append((np.copy(new_precision), np.copy(new_reward_vector)))

                pipe.multi()
                for c, (prec, rew) in current_vals.items():
                    pipe.set(
                        f"{self.prefix}linear:{c}:precision", json.dumps(prec.tolist())
                    )
                    pipe.set(
                        f"{self.prefix}linear:{c}:reward", json.dumps(rew.tolist())
                    )
                await pipe.execute()
                return results
            except redis.WatchError:
                continue

    async def asave_vectors(self, vectors: Dict[str, Sequence[float]]) -> None:
        if not vectors:
            return
        pipe = self.client.pipeline()
        for k, v in vectors.items():
            pipe.hset(f"{self.prefix}context_vectors", key=k, value=json.dumps(list(v)))
        await pipe.execute()

    async def log_selection(
        self, trace_id: str, context_key: str, candidate_name: str
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        key = f"{self.prefix}log:{trace_id}"
        await self.client.hset(
            key,
            mapping={
                "trace_id": trace_id,
                "timestamp": timestamp,
                "context_key": context_key,
                "candidate_name": candidate_name,
            },
        )
        await self.client.sadd(f"{self.prefix}log_ids", trace_id)

    async def log_feedback(self, trace_id: str, reward: float) -> None:
        key = f"{self.prefix}log:{trace_id}"
        if await self.client.exists(key):
            await self.client.hset(key, "reward", str(reward))

    async def get_selection_logs(self) -> List[Dict[str, Any]]:
        trace_ids = await self.client.smembers(f"{self.prefix}log_ids")
        logs = []
        for tid in trace_ids:
            tid_str = tid.decode("utf-8") if isinstance(tid, bytes) else str(tid)
            key = f"{self.prefix}log:{tid_str}"
            data = await self.client.hgetall(key)
            if data:
                decoded = {}
                for k, v in data.items():
                    k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                    v_str = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    decoded[k_str] = v_str

                logs.append(
                    {
                        "trace_id": decoded.get("trace_id", tid_str),
                        "timestamp": decoded.get("timestamp", ""),
                        "context_key": decoded.get("context_key", ""),
                        "candidate_name": decoded.get("candidate_name", ""),
                        "reward": (
                            float(decoded["reward"])
                            if decoded.get("reward") is not None
                            else None
                        ),
                    }
                )
        return sorted(logs, key=lambda x: x["timestamp"])
