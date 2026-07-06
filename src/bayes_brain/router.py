import base64
import hashlib
import json
import logging
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from bayes_brain.embeddings import ContextEmbedder, VectorContextStore, VectorStoreProtocol
from bayes_brain.storage import BaseStorage, InMemoryStorage

logger = logging.getLogger(__name__)


class BayesianToolRouter:
    """
    Decoupled tool routing middleware implementing a Contextual Multi-Armed Bandit
    via Thompson Sampling.
    """

    def __init__(
        self,
        storage: Optional[BaseStorage] = None,
        embedder: Optional[ContextEmbedder] = None,
        decay_factor: float = 1.0,
        similarity_threshold: float = 0.8,
        priors: Optional[Dict[str, Tuple[float, float]]] = None,
        vector_store: Optional[VectorStoreProtocol] = None,
    ) -> None:
        """
        Initialize the BayesianToolRouter.

        Args:
            storage: Storage backend for persisting alphas and betas. Defaults to InMemoryStorage.
            embedder: Optional ContextEmbedder protocol to generate query embeddings.
            decay_factor: Exponential decay / discount factor (gamma) in (0, 1]. Defaults to 1.0.
            similarity_threshold: Cosine similarity threshold for mapping embeddings to contexts.
            priors: Preseeded alpha/beta priors for tools to mitigate cold start (e.g. {"tool": (10, 2)}).
            vector_store: Optional custom VectorStoreProtocol implementation.
        """
        self.storage = storage or InMemoryStorage()
        self.embedder = embedder
        
        if embedder is None:
            logger.warning(
                "No ContextEmbedder provided. Operating in exact-match fallback mode. "
                "For contextual tasks with semantic variation, providing an embedder is highly recommended."
            )
        
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError("decay_factor must be in the range (0, 1]")
        self.decay_factor = decay_factor
        self.similarity_threshold = similarity_threshold
        self.priors = priors or {}

        self._custom_vector_store_active = vector_store is not None
        if vector_store is not None:
            self._context_store = vector_store
        else:
            self._context_store = VectorContextStore()
            self._load_context_store()

    def _load_context_store(self) -> None:
        """Attempt to restore the VectorContextStore from the storage backend."""
        if self._custom_vector_store_active:
            return
        try:
            vectors = self.storage.load_all_vectors()
            for key, vector in vectors.items():
                self._context_store.add_context(key, vector)
        except Exception:
            # Fall back to empty context store if retrieval or decoding fails
            pass

    def _save_context_store(self) -> None:
        """Persist the VectorContextStore to the storage backend."""
        if self._custom_vector_store_active:
            return
        try:
            if hasattr(self._context_store, "_contexts"):
                for key, vector in self._context_store._contexts.items():
                    self.storage.save_vector(key, vector)
        except Exception:
            pass

    def _hash_context_text(self, context_text: str) -> str:
        """
        Normalize the context string (strip and collapse multiple whitespaces)
        and hash it using SHA-256 to ensure short, fixed-length keys.
        """
        normalized = " ".join(context_text.strip().split())
        sha256_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"hash_{sha256_hash}"

    def _resolve_context_key(self, context_text: str) -> str:
        """
        Resolve the given raw context string into a normalized context key.
        If an embedder is active, maps to the closest vector cluster context;
        otherwise, does a direct, exact string lookup.
        """
        if not self.embedder:
            return self._hash_context_text(context_text)

        try:
            vector = self.embedder.embed_query(context_text)
        except Exception:
            # Fall back to exact string context key if embedding extraction fails
            logger.warning(
                "Failed to generate embedding for context. Falling back to exact-match hashing."
            )
            return self._hash_context_text(context_text)

        # Find nearest vector context in index
        matched_key = self._context_store.get_nearest_context(
            query_vector=vector,
            similarity_threshold=self.similarity_threshold,
        )

        if matched_key is not None:
            return matched_key

        # No match found: spawn a new context cluster and save it
        new_key = f"ctx_{uuid.uuid4().hex}"
        self._context_store.add_context(new_key, vector)
        if not self._custom_vector_store_active:
            self.storage.save_vector(new_key, vector)
        return new_key

    def _generate_trace_id(self, context_key: str, tool_name: str) -> str:
        """Encodes context key and tool name into a stateless token."""
        payload = {
            "ctx": context_key,
            "tool": tool_name,
            "nonce": uuid.uuid4().hex,
        }
        json_bytes = json.dumps(payload).encode("utf-8")
        return base64.urlsafe_b64encode(json_bytes).decode("utf-8")

    def _decode_trace_id(self, trace_id: str) -> Tuple[str, str]:
        """Decodes context key and tool name from a trace ID token."""
        try:
            json_bytes = base64.urlsafe_b64decode(trace_id.encode("utf-8"))
            payload = json.loads(json_bytes.decode("utf-8"))
            return payload["ctx"], payload["tool"]
        except Exception as e:
            raise ValueError(f"Invalid or corrupted trace ID: {trace_id}") from e

    def route(self, context_text: str, candidate_tools: List[str]) -> str:
        """
        Implements Thompson Sampling across a filtered list of valid tools.
        Returns the name of the tool selected.
        """
        chosen_tool, _ = self.route_with_trace(context_text, candidate_tools)
        return chosen_tool

    def route_with_trace(
        self, context_text: str, candidate_tools: List[str]
    ) -> Tuple[str, str]:
        """
        Implements Thompson Sampling and returns a tuple of (chosen_tool_name, trace_id).
        The trace_id allows reward signals to be logged completely asynchronously.
        """
        if not candidate_tools:
            raise ValueError("Candidate tools list cannot be empty")

        context_key = self._resolve_context_key(context_text)
        best_tool = None
        highest_sample = -1.0

        for tool_name in candidate_tools:
            alpha, beta = self.storage.get_tool_params(context_key, tool_name)

            # Check for seeded priors on cold start
            if alpha == 1.0 and beta == 1.0 and tool_name in self.priors:
                alpha, beta = self.priors[tool_name]
                self.storage.update_tool_params(context_key, tool_name, alpha, beta)

            # Sample belief matching beta-binomial posterior
            sampled_score = np.random.beta(alpha, beta)

            if sampled_score > highest_sample:
                highest_sample = sampled_score
                best_tool = tool_name

        if best_tool is None:
            best_tool = candidate_tools[0]

        trace_id = self._generate_trace_id(context_key, best_tool)
        return best_tool, trace_id

    def feedback(self, context_text: str, tool_name: str, success: bool) -> Tuple[float, float]:
        """
        Directly submit tool execution feedback using the raw context string.
        """
        context_key = self._resolve_context_key(context_text)
        reward = 1.0 if success else 0.0
        return self.storage.decay_and_update(
            context_key, tool_name, self.decay_factor, reward
        )

    def feedback_by_trace(self, trace_id: str, success: bool) -> Tuple[float, float]:
        """
        Directly submit tool execution feedback using a generated trace ID.
        Ideal for asynchronous and decoupled systems.
        """
        context_key, tool_name = self._decode_trace_id(trace_id)
        reward = 1.0 if success else 0.0
        return self.storage.decay_and_update(
            context_key, tool_name, self.decay_factor, reward
        )

    def get_tool_beliefs(self, context_text: str, tool_name: str) -> Tuple[float, float]:
        """
        Retrieve current posterior alpha and beta beliefs for a given context and tool.
        """
        context_key = self._resolve_context_key(context_text)
        return self.storage.get_tool_params(context_key, tool_name)
