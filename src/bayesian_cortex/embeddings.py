import asyncio
import json
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

class ContextEmbedder(Protocol):
    """Protocol defining how to convert text into a vector context key."""

    def embed_query(self, text: str) -> Sequence[float]:
        """Convert a text query (prompt) into a vector of floats."""
        ...

    def embed_queries(self, texts: List[str]) -> List[Sequence[float]]:
        """Convert multiple text queries into vectors of floats."""
        ...


class AsyncContextEmbedder(Protocol):
    """Protocol defining how to convert text into a vector context key asynchronously."""

    async def aembed_query(self, text: str) -> Sequence[float]:
        """Convert a text query (prompt) into a vector of floats."""
        ...

    async def aembed_queries(self, texts: List[str]) -> List[Sequence[float]]:
        """Convert multiple text queries into vectors of floats asynchronously."""
        ...



class VectorStoreProtocol(Protocol):
    """Protocol defining the interface for context vector storage and search."""

    def add_context(self, context_key: str, vector: Sequence[float]) -> None:
        """Add or update a context vector in the index."""
        ...

    def get_nearest_context(
        self, query_vector: Sequence[float], similarity_threshold: float = 0.8
    ) -> Optional[str]:
        """
        Find the context_key whose stored vector is closest to query_vector,
        provided the cosine similarity is above the threshold.
        """
        ...

    def get_context_vector(self, context_key: str) -> Optional[Sequence[float]]:
        """Retrieve the original vector associated with context_key."""
        ...


class LocalSentenceTransformerEmbedder:
    """
    Batteries-included embedder using sentence-transformers.
    Loaded lazily, requiring `pip install 'bayesian-cortex[local-ml]'`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "The sentence-transformers package is required for LocalSentenceTransformerEmbedder. "
                    "Please install it with: pip install 'bayesian-cortex[local-ml]'"
                )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, text: str) -> Sequence[float]:
        embedding = self.model.encode(text)
        return [float(x) for x in embedding]

    async def aembed_query(self, text: str) -> Sequence[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    def embed_queries(self, texts: List[str]) -> List[Sequence[float]]:
        embeddings = self.model.encode(texts)
        return [[float(x) for x in emb] for emb in embeddings]

    async def aembed_queries(self, texts: List[str]) -> List[Sequence[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_queries, texts)



class GeminiEmbedder:
    """
    Lightweight, API-driven embedder using Google's Gemini (Generative Language) API.
    Can be used via raw HTTP requests (using standard urllib) or via an optionally provided client SDK.
    """

    def __init__(
        self,
        model_name: str = "models/text-embedding-004",
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com",
        api_version: str = "v1beta",
        client: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.client = client

    def embed_query(self, text: str) -> Sequence[float]:
        if self.client is not None:
            # Try new SDK style: client.models.embed_content
            if hasattr(self.client, "models") and hasattr(self.client.models, "embed_content"):
                resp = self.client.models.embed_content(model=self.model_name, contents=text)
                if hasattr(resp, "embedding") and hasattr(resp.embedding, "values"):
                    return [float(x) for x in resp.embedding.values]
            # Try legacy SDK style or module: client.embed_content
            if hasattr(self.client, "embed_content"):
                resp = self.client.embed_content(model=self.model_name, contents=text)
                if isinstance(resp, dict) and "embedding" in resp:
                    embedding_val = resp["embedding"]
                    if isinstance(embedding_val, dict) and "values" in embedding_val:
                        return [float(x) for x in embedding_val["values"]]
                    return [float(x) for x in embedding_val]
                if hasattr(resp, "embedding"):
                    emb = resp.embedding
                    if hasattr(emb, "values"):
                        return [float(x) for x in emb.values]
                    return [float(x) for x in emb]
            raise ValueError("Provided client does not have embed_content method or expected structure.")

        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Pass it via api_key or set the GEMINI_API_KEY environment variable."
            )

        model = self.model_name
        if not model.startswith("models/") and not model.startswith("tunedModels/"):
            model = f"models/{model}"

        url = f"{self.base_url}/{self.api_version}/{model}:embedContent?key={self.api_key}"
        payload = {
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                if "embedding" in resp_data and "values" in resp_data["embedding"]:
                    return [float(x) for x in resp_data["embedding"]["values"]]
                raise ValueError(f"Unexpected response structure from Gemini API: {resp_data}")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            raise RuntimeError(f"Gemini API request failed with status {e.code}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with Gemini API: {e}") from e

    async def aembed_query(self, text: str) -> Sequence[float]:
        if self.client is not None:
            if hasattr(self.client, "aio") and hasattr(self.client.aio, "models") and hasattr(self.client.aio.models, "embed_content"):
                resp = await self.client.aio.models.embed_content(model=self.model_name, contents=text)
                if hasattr(resp, "embedding") and hasattr(resp.embedding, "values"):
                    return [float(x) for x in resp.embedding.values]
            if hasattr(self.client, "models") and hasattr(self.client.models, "embed_content"):
                func = self.client.models.embed_content
                if asyncio.iscoroutinefunction(func):
                    resp = await func(model=self.model_name, contents=text)
                else:
                    resp = func(model=self.model_name, contents=text)
                if hasattr(resp, "embedding") and hasattr(resp.embedding, "values"):
                    return [float(x) for x in resp.embedding.values]
            if hasattr(self.client, "embed_content_async"):
                resp = await self.client.embed_content_async(model=self.model_name, contents=text)
                if isinstance(resp, dict) and "embedding" in resp:
                    embedding_val = resp["embedding"]
                    if isinstance(embedding_val, dict) and "values" in embedding_val:
                        return [float(x) for x in embedding_val["values"]]
                    return [float(x) for x in embedding_val]
                if hasattr(resp, "embedding"):
                    emb = resp.embedding
                    if hasattr(emb, "values"):
                        return [float(x) for x in emb.values]
                    return [float(x) for x in emb]
            if hasattr(self.client, "embed_content"):
                func = self.client.embed_content
                if asyncio.iscoroutinefunction(func):
                    resp = await func(model=self.model_name, contents=text)
                else:
                    resp = func(model=self.model_name, contents=text)
                if isinstance(resp, dict) and "embedding" in resp:
                    embedding_val = resp["embedding"]
                    if isinstance(embedding_val, dict) and "values" in embedding_val:
                        return [float(x) for x in embedding_val["values"]]
                    return [float(x) for x in embedding_val]
                if hasattr(resp, "embedding"):
                    emb = resp.embedding
                    if hasattr(emb, "values"):
                        return [float(x) for x in emb.values]
                    return [float(x) for x in emb]
            raise ValueError("Provided client does not have embed_content method or expected structure.")

        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Pass it via api_key or set the GEMINI_API_KEY environment variable."
            )

        model = self.model_name
        if not model.startswith("models/") and not model.startswith("tunedModels/"):
            model = f"models/{model}"

        url = f"{self.base_url}/{self.api_version}/{model}:embedContent?key={self.api_key}"
        payload = {
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for async embedding calls. "
                "Please install it with: pip install httpx"
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                response = await httpx_client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                resp_data = response.json()
                if "embedding" in resp_data and "values" in resp_data["embedding"]:
                    return [float(x) for x in resp_data["embedding"]["values"]]
                raise ValueError(f"Unexpected response structure from Gemini API: {resp_data}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Gemini API request failed with status {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with Gemini API: {e}") from e

    def embed_queries(self, texts: List[str]) -> List[Sequence[float]]:
        if not texts:
            return []
        if self.client is not None:
            try:
                if hasattr(self.client, "models") and hasattr(self.client.models, "embed_content"):
                    resp = self.client.models.embed_content(model=self.model_name, contents=texts)
                    if hasattr(resp, "embeddings"):
                        return [[float(x) for x in emb.values] for emb in resp.embeddings]
            except Exception:
                pass
            return [self.embed_query(t) for t in texts]

        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Pass it via api_key or set the GEMINI_API_KEY environment variable."
            )

        model = self.model_name
        if not model.startswith("models/") and not model.startswith("tunedModels/"):
            model = f"models/{model}"

        url = f"{self.base_url}/{self.api_version}/{model}:batchEmbedContents?key={self.api_key}"
        payload = {
            "requests": [
                {
                    "model": model,
                    "content": {"parts": [{"text": text}]}
                }
                for text in texts
            ]
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                if "embeddings" in resp_data:
                    return [[float(x) for x in emb["values"]] for emb in resp_data["embeddings"]]
                raise ValueError(f"Unexpected response structure from Gemini API: {resp_data}")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            raise RuntimeError(f"Gemini API request failed with status {e.code}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with Gemini API: {e}") from e

    async def aembed_queries(self, texts: List[str]) -> List[Sequence[float]]:
        if not texts:
            return []
        if self.client is not None:
            try:
                if hasattr(self.client, "aio") and hasattr(self.client.aio, "models") and hasattr(self.client.aio.models, "embed_content"):
                    resp = await self.client.aio.models.embed_content(model=self.model_name, contents=texts)
                    if hasattr(resp, "embeddings"):
                        return [[float(x) for x in emb.values] for emb in resp.embeddings]
            except Exception:
                pass
            return await asyncio.gather(*(self.aembed_query(t) for t in texts))

        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Pass it via api_key or set the GEMINI_API_KEY environment variable."
            )

        model = self.model_name
        if not model.startswith("models/") and not model.startswith("tunedModels/"):
            model = f"models/{model}"

        url = f"{self.base_url}/{self.api_version}/{model}:batchEmbedContents?key={self.api_key}"
        payload = {
            "requests": [
                {
                    "model": model,
                    "content": {"parts": [{"text": text}]}
                }
                for text in texts
            ]
        }
        
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for async embedding calls. "
                "Please install it with: pip install httpx"
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                response = await httpx_client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                resp_data = response.json()
                if "embeddings" in resp_data:
                    return [[float(x) for x in emb["values"]] for emb in resp_data["embeddings"]]
                raise ValueError(f"Unexpected response structure from Gemini API: {resp_data}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Gemini API request failed with status {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with Gemini API: {e}") from e



class OpenAIEmbedder:
    """
    Lightweight, API-driven embedder using OpenAI's Embeddings API.
    Can be used via raw HTTP requests (using standard urllib) or via an optionally provided client SDK.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        env_base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif env_base_url:
            self.base_url = env_base_url.rstrip("/")
        else:
            self.base_url = "https://api.openai.com/v1"
            
        self.client = client

    def embed_query(self, text: str) -> Sequence[float]:
        if self.client is not None:
            if hasattr(self.client, "embeddings") and hasattr(self.client.embeddings, "create"):
                resp = self.client.embeddings.create(input=text, model=self.model_name)
                if hasattr(resp, "data") and len(resp.data) > 0:
                    embedding_val = resp.data[0].embedding
                    return [float(x) for x in embedding_val]
            raise ValueError("Provided client does not have embeddings.create method or expected structure.")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Pass it via api_key or set the OPENAI_API_KEY environment variable."
            )

        url = f"{self.base_url}/embeddings"
        payload = {
            "input": text,
            "model": self.model_name,
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                if "data" in resp_data and len(resp_data["data"]) > 0 and "embedding" in resp_data["data"][0]:
                    return [float(x) for x in resp_data["data"][0]["embedding"]]
                raise ValueError(f"Unexpected response structure from OpenAI API: {resp_data}")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            raise RuntimeError(f"OpenAI API request failed with status {e.code}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with OpenAI API: {e}") from e

    async def aembed_query(self, text: str) -> Sequence[float]:
        if self.client is not None:
            if hasattr(self.client, "embeddings") and hasattr(self.client.embeddings, "create"):
                func = self.client.embeddings.create
                if asyncio.iscoroutinefunction(func):
                    resp = await func(input=text, model=self.model_name)
                else:
                    resp = func(input=text, model=self.model_name)
                if hasattr(resp, "data") and len(resp.data) > 0:
                    embedding_val = resp.data[0].embedding
                    return [float(x) for x in embedding_val]
            raise ValueError("Provided client does not have embeddings.create method or expected structure.")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Pass it via api_key or set the OPENAI_API_KEY environment variable."
            )

        url = f"{self.base_url}/embeddings"
        payload = {
            "input": text,
            "model": self.model_name,
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for async embedding calls. "
                "Please install it with: pip install httpx"
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                response = await httpx_client.post(
                    url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                resp_data = response.json()
                if "data" in resp_data and len(resp_data["data"]) > 0 and "embedding" in resp_data["data"][0]:
                    return [float(x) for x in resp_data["data"][0]["embedding"]]
                raise ValueError(f"Unexpected response structure from OpenAI API: {resp_data}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"OpenAI API request failed with status {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with OpenAI API: {e}") from e

    def embed_queries(self, texts: List[str]) -> List[Sequence[float]]:
        if not texts:
            return []
        if self.client is not None:
            if hasattr(self.client, "embeddings") and hasattr(self.client.embeddings, "create"):
                resp = self.client.embeddings.create(input=texts, model=self.model_name)
                if hasattr(resp, "data"):
                    sorted_data = sorted(resp.data, key=lambda x: getattr(x, "index", 0))
                    return [[float(x) for x in item.embedding] for item in sorted_data]
            raise ValueError("Provided client does not have embeddings.create method or expected structure.")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Pass it via api_key or set the OPENAI_API_KEY environment variable."
            )

        url = f"{self.base_url}/embeddings"
        payload = {
            "input": texts,
            "model": self.model_name,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                if "data" in resp_data:
                    sorted_data = sorted(resp_data["data"], key=lambda x: x.get("index", 0))
                    return [[float(x) for x in item["embedding"]] for item in sorted_data]
                raise ValueError(f"Unexpected response structure from OpenAI API: {resp_data}")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            raise RuntimeError(f"OpenAI API request failed with status {e.code}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with OpenAI API: {e}") from e

    async def aembed_queries(self, texts: List[str]) -> List[Sequence[float]]:
        if not texts:
            return []
        if self.client is not None:
            if hasattr(self.client, "embeddings") and hasattr(self.client.embeddings, "create"):
                func = self.client.embeddings.create
                if asyncio.iscoroutinefunction(func):
                    resp = await func(input=texts, model=self.model_name)
                else:
                    resp = func(input=texts, model=self.model_name)
                if hasattr(resp, "data"):
                    sorted_data = sorted(resp.data, key=lambda x: getattr(x, "index", 0))
                    return [[float(x) for x in item.embedding] for item in sorted_data]
            raise ValueError("Provided client does not have embeddings.create method or expected structure.")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Pass it via api_key or set the OPENAI_API_KEY environment variable."
            )

        url = f"{self.base_url}/embeddings"
        payload = {
            "input": texts,
            "model": self.model_name,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for async embedding calls. "
                "Please install it with: pip install httpx"
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as httpx_client:
                response = await httpx_client.post(
                    url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                resp_data = response.json()
                if "data" in resp_data:
                    sorted_data = sorted(resp_data["data"], key=lambda x: x.get("index", 0))
                    return [[float(x) for x in item["embedding"]] for item in sorted_data]
                raise ValueError(f"Unexpected response structure from OpenAI API: {resp_data}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"OpenAI API request failed with status {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with OpenAI API: {e}") from e



class VectorContextStore:
    """
    A lightweight, in-memory vector index for storing and querying reference contexts.
    Uses cosine similarity to map query vectors to nearest contextual keys.
    """

    def __init__(self) -> None:
        # Maps context_key to its vector representation
        self._contexts: Dict[str, np.ndarray] = {}

    def add_context(self, context_key: str, vector: Sequence[float]) -> None:
        """Add or update a context vector in the index."""
        self._contexts[context_key] = np.array(vector, dtype=np.float32)

    def get_context_vector(self, context_key: str) -> Optional[Sequence[float]]:
        """Retrieve the original vector associated with context_key."""
        vec = self._contexts.get(context_key)
        if vec is not None:
            return vec.tolist()
        return None

    def get_nearest_context(
        self, query_vector: Sequence[float], similarity_threshold: float = 0.8
    ) -> Optional[str]:
        """
        Find the context_key whose stored vector is closest to query_vector,
        provided the cosine similarity is above the threshold.
        """
        if not self._contexts:
            return None

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0.0:
            return None

        best_key = None
        best_similarity = -1.0

        for key, ref_vec in self._contexts.items():
            ref_norm = np.linalg.norm(ref_vec)
            if ref_norm == 0.0:
                continue
            
            # Cosine similarity calculation
            similarity = float(np.dot(q_vec, ref_vec) / (q_norm * ref_norm))
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key

        if best_similarity >= similarity_threshold:
            return best_key
            
        return None

    def to_json(self) -> str:
        """Serialize the context store to a JSON string."""
        serializable = {
            key: vec.tolist() for key, vec in self._contexts.items()
        }
        return json.dumps(serializable)

    @classmethod
    def from_json(cls, data_str: str) -> "VectorContextStore":
        """Deserialize context store from a JSON string."""
        store = cls()
        data = json.loads(data_str)
        for key, vec_list in data.items():
            store.add_context(key, vec_list)
        return store


class SQLiteVectorStore:
    """
    A persistent, index-backed vector store implementation using sqlite-vec
    that executes vector search natively in SQLite.
    """

    def __init__(
        self,
        db_path: str = "bayesian_cortex_vectors.db",
        dimension: int = 384,
        table_name: str = "vec_context_store",
    ) -> None:
        self.db_path = db_path
        self.dimension = dimension
        self.table_name = table_name
        self._local = threading.local()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        try:
            import sqlite_vec
        except ImportError:
            raise ImportError(
                "sqlite-vec is required for SQLiteVectorStore. "
                "Please install it with: pip install sqlite-vec"
            )
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = self._connect()
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name} USING vec0("
                    f"context_key TEXT,"
                    f"embedding float[{self.dimension}] distance_metric=cosine"
                    f")"
                )
        finally:
            conn.close()

    def add_context(self, context_key: str, vector: Sequence[float]) -> None:
        import sqlite_vec
        conn = self._get_conn()
        serialized_vec = sqlite_vec.serialize_float32(list(vector))
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT rowid FROM {self.table_name} WHERE context_key = ?",
                (context_key,),
            )
            row = cursor.fetchone()
            if row:
                rowid = row[0]
                conn.execute(
                    f"UPDATE {self.table_name} SET embedding = ? WHERE rowid = ?",
                    (serialized_vec, rowid),
                )
            else:
                conn.execute(
                    f"INSERT INTO {self.table_name}(context_key, embedding) VALUES (?, ?)",
                    (context_key, serialized_vec),
                )

    def get_nearest_context(
        self, query_vector: Sequence[float], similarity_threshold: float = 0.8
    ) -> Optional[str]:
        import sqlite_vec
        conn = self._get_conn()
        serialized_query = sqlite_vec.serialize_float32(list(query_vector))
        cursor = conn.cursor()
        
        # Cosine distance = 1.0 - similarity.
        # similarity >= similarity_threshold => 1.0 - distance >= similarity_threshold => distance <= 1.0 - similarity_threshold.
        max_distance = 1.0 - similarity_threshold
        
        cursor.execute(
            f"SELECT context_key, distance FROM {self.table_name} "
            f"WHERE embedding MATCH ? "
            f"ORDER BY distance "
            f"LIMIT 1",
            (serialized_query,),
        )
        row = cursor.fetchone()
        if row:
            matched_key, distance = row[0], float(row[1])
            if distance <= max_distance:
                return matched_key
        return None

    def get_context_vector(self, context_key: str) -> Optional[Sequence[float]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT embedding FROM {self.table_name} WHERE context_key = ?",
            (context_key,),
        )
        row = cursor.fetchone()
        if row:
            import sqlite_vec
            return sqlite_vec.deserialize_float32(row[0])
        return None

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            delattr(self._local, "conn")


class AsyncVectorStoreProtocol(Protocol):
    """Protocol defining the async interface for context vector storage and search."""

    async def aadd_context(self, context_key: str, vector: Sequence[float]) -> None:
        """Add or update a context vector in the index."""
        ...

    async def aget_nearest_context(
        self, query_vector: Sequence[float], similarity_threshold: float = 0.8
    ) -> Optional[str]:
        """
        Find the context_key whose stored vector is closest to query_vector,
        provided the cosine similarity is above the threshold.
        """
        ...

    async def aget_context_vector(self, context_key: str) -> Optional[Sequence[float]]:
        """Retrieve the original vector associated with context_key."""
        ...


class AsyncVectorContextStore:
    """
    A lightweight, async-native in-memory vector index for storing and querying reference contexts.
    Uses cosine similarity to map query vectors to nearest contextual keys.
    """

    def __init__(self) -> None:
        self._contexts: Dict[str, np.ndarray] = {}
        self._lock = asyncio.Lock()

    async def aadd_context(self, context_key: str, vector: Sequence[float]) -> None:
        async with self._lock:
            self._contexts[context_key] = np.array(vector, dtype=np.float32)

    async def aget_context_vector(self, context_key: str) -> Optional[Sequence[float]]:
        async with self._lock:
            vec = self._contexts.get(context_key)
            if vec is not None:
                return vec.tolist()
            return None

    async def aget_nearest_context(
        self, query_vector: Sequence[float], similarity_threshold: float = 0.8
    ) -> Optional[str]:
        async with self._lock:
            if not self._contexts:
                return None

            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0.0:
                return None

            best_key = None
            best_similarity = -1.0

            for key, ref_vec in self._contexts.items():
                ref_norm = np.linalg.norm(ref_vec)
                if ref_norm == 0.0:
                    continue
                
                similarity = float(np.dot(q_vec, ref_vec) / (q_norm * ref_norm))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_key = key

            if best_similarity >= similarity_threshold:
                return best_key
                
            return None

    def to_json(self) -> str:
        serializable = {
            key: vec.tolist() for key, vec in self._contexts.items()
        }
        return json.dumps(serializable)

    @classmethod
    def from_json(cls, data_str: str) -> "AsyncVectorContextStore":
        store = cls()
        data = json.loads(data_str)
        for key, vec_list in data.items():
            store._contexts[key] = np.array(vec_list, dtype=np.float32)
        return store


class AsyncSQLiteVectorStore:
    """
    A persistent, index-backed vector store implementation using sqlite-vec
    and aiosqlite that executes vector search natively in SQLite asynchronously.
    """

    def __init__(
        self,
        db_path: str = "bayesian_cortex_vectors.db",
        dimension: int = 384,
        table_name: str = "vec_context_store",
    ) -> None:
        self.db_path = db_path
        self.dimension = dimension
        self.table_name = table_name
        self._conn: Optional[Any] = None
        self._lock = asyncio.Lock()

    async def _get_conn(self) -> Any:
        async with self._lock:
            if self._conn is None:
                try:
                    import aiosqlite
                    import sqlite_vec
                except ImportError:
                    raise ImportError(
                        "aiosqlite and sqlite-vec are required for AsyncSQLiteVectorStore. "
                        "Please install them."
                    )
                self._conn = await aiosqlite.connect(self.db_path)
                await self._conn.execute("PRAGMA journal_mode=WAL;")
                await self._conn.execute("PRAGMA busy_timeout=5000;")
                await self._conn.enable_load_extension(True)
                await self._conn.load_extension(sqlite_vec.loadable_path())
                await self._conn.enable_load_extension(False)
                
                await self._conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name} USING vec0("
                    f"context_key TEXT,"
                    f"embedding float[{self.dimension}] distance_metric=cosine"
                    f")"
                )
                await self._conn.commit()
            return self._conn

    async def aadd_context(self, context_key: str, vector: Sequence[float]) -> None:
        import sqlite_vec
        conn = await self._get_conn()
        serialized_vec = sqlite_vec.serialize_float32(list(vector))
        async with self._lock:
            async with conn.execute(
                f"SELECT rowid FROM {self.table_name} WHERE context_key = ?",
                (context_key,),
            ) as cursor:
                row = await cursor.fetchone()
            if row:
                rowid = row[0]
                await conn.execute(
                    f"UPDATE {self.table_name} SET embedding = ? WHERE rowid = ?",
                    (serialized_vec, rowid),
                )
            else:
                await conn.execute(
                    f"INSERT INTO {self.table_name}(context_key, embedding) VALUES (?, ?)",
                    (context_key, serialized_vec),
                )
            await conn.commit()

    async def aget_nearest_context(
        self, query_vector: Sequence[float], similarity_threshold: float = 0.8
    ) -> Optional[str]:
        import sqlite_vec
        conn = await self._get_conn()
        serialized_query = sqlite_vec.serialize_float32(list(query_vector))
        max_distance = 1.0 - similarity_threshold
        
        async with self._lock:
            async with conn.execute(
                f"SELECT context_key, distance FROM {self.table_name} "
                f"WHERE embedding MATCH ? "
                f"ORDER BY distance "
                f"LIMIT 1",
                (serialized_query,),
            ) as cursor:
                row = await cursor.fetchone()
        
        if row:
            matched_key, distance = row[0], float(row[1])
            if distance <= max_distance:
                return matched_key
        return None

    async def aget_context_vector(self, context_key: str) -> Optional[Sequence[float]]:
        conn = await self._get_conn()
        async with self._lock:
            async with conn.execute(
                f"SELECT embedding FROM {self.table_name} WHERE context_key = ?",
                (context_key,),
            ) as cursor:
                row = await cursor.fetchone()
        if row:
            import sqlite_vec
            return sqlite_vec.deserialize_float32(row[0])
        return None

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None

