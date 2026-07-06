import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

class ContextEmbedder(Protocol):
    """Protocol defining how to convert text into a vector context key."""

    def embed_query(self, text: str) -> Sequence[float]:
        """Convert a text query (prompt) into a vector of floats."""
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


class LocalSentenceTransformerEmbedder:
    """
    Batteries-included embedder using sentence-transformers.
    Loaded lazily, requiring `pip install 'bayes-brain[local-ml]'`.
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
                    "Please install it with: pip install 'bayes-brain[local-ml]'"
                )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, text: str) -> Sequence[float]:
        embedding = self.model.encode(text)
        return [float(x) for x in embedding]


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
