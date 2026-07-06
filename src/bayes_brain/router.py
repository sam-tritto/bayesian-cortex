import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from bayes_brain.embeddings import (
    AsyncContextEmbedder,
    AsyncVectorContextStore,
    AsyncVectorStoreProtocol,
    ContextEmbedder,
    VectorContextStore,
    VectorStoreProtocol,
)
from bayes_brain.storage import (
    AsyncBaseStorage,
    AsyncInMemoryStorage,
    BaseStorage,
    InMemoryStorage,
)

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
        contextual_priors: Optional[List[Dict[str, Any]]] = None,
        vector_store: Optional[VectorStoreProtocol] = None,
        fallback_tool: Optional[str] = None,
        telemetry_hook: Optional[Callable[[str, Exception, Dict[str, Any]], None]] = None,
        mode: str = "clustering",
        exploration_weight: float = 1.0,
        lambda_val: float = 1.0,
        diagonal_covariance: bool = False,
        secret_key: Optional[Union[str, bytes]] = None,
    ) -> None:
        """
        Initialize the BayesianToolRouter.

        Args:
            storage: Storage backend for persisting alphas and betas. Defaults to InMemoryStorage.
            embedder: Optional ContextEmbedder protocol to generate query embeddings.
            decay_factor: Exponential decay / discount factor (gamma) in (0, 1]. Defaults to 1.0.
            similarity_threshold: Cosine similarity threshold for mapping embeddings to contexts.
            priors: Preseeded alpha/beta priors for tools to mitigate cold start (e.g. {"tool": (10, 2)}).
            contextual_priors: List of context-specific prior rules matching regex or embedding clusters.
            vector_store: Optional custom VectorStoreProtocol implementation.
            mode: routing mode ("clustering", "lints", "linucb").
            exploration_weight: exploration factor (v for lints, alpha for linucb).
            lambda_val: L2 regularization coefficient.
            diagonal_covariance: whether to use diagonal covariance approximation.
            secret_key: Secret key used to sign and verify trace IDs via HMAC.
        """
        self.storage = storage or InMemoryStorage()
        self.embedder = embedder
        self.fallback_tool = fallback_tool
        self.telemetry_hook = telemetry_hook
        
        self.mode = mode
        if mode not in ("clustering", "lints", "linucb"):
            raise ValueError("mode must be 'clustering', 'lints', or 'linucb'")
        self.exploration_weight = exploration_weight
        self.lambda_val = lambda_val
        self.diagonal_covariance = diagonal_covariance

        # Determine secret key for signing trace IDs
        if secret_key is not None:
            if isinstance(secret_key, str):
                self.secret_key = secret_key.encode("utf-8")
            else:
                self.secret_key = secret_key
        else:
            env_key = os.environ.get("BAYES_BRAIN_SECRET_KEY")
            if env_key:
                self.secret_key = env_key.encode("utf-8")
            else:
                self.secret_key = os.urandom(32)

        if self.mode in ("lints", "linucb") and self.embedder is None:
            raise ValueError("Linear bandit modes ('lints', 'linucb') require a ContextEmbedder.")

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

        # Validate and parse contextual priors
        self.contextual_priors = []
        if contextual_priors:
            for item in contextual_priors:
                parsed_item = {}
                if "priors" not in item or not isinstance(item["priors"], dict):
                    raise ValueError("Each contextual prior must contain a 'priors' dictionary.")
                
                priors_map = {}
                for t_name, params in item["priors"].items():
                    if not isinstance(params, (list, tuple)) or len(params) != 2:
                        raise ValueError(f"Prior parameters for tool '{t_name}' must be a tuple/list of (alpha, beta).")
                    priors_map[t_name] = (float(params[0]), float(params[1]))
                parsed_item["priors"] = priors_map

                if "pattern" in item:
                    if not isinstance(item["pattern"], str):
                        raise ValueError("Contextual prior pattern must be a regex string.")
                    try:
                        parsed_item["pattern"] = re.compile(item["pattern"])
                    except re.error as e:
                        raise ValueError(f"Invalid regex pattern '{item['pattern']}': {e}")
                
                if "reference_context" in item:
                    if not isinstance(item["reference_context"], str):
                        raise ValueError("Contextual prior reference_context must be a string.")
                    parsed_item["reference_context"] = item["reference_context"]
                
                if "embedding" in item:
                    if not isinstance(item["embedding"], (list, tuple, np.ndarray)):
                        raise ValueError("Contextual prior embedding must be a list/tuple/numpy array of floats.")
                    parsed_item["embedding"] = np.array(item["embedding"], dtype=np.float32)
                
                if "similarity_threshold" in item:
                    parsed_item["similarity_threshold"] = float(item["similarity_threshold"])
                
                if "pattern" not in parsed_item and "reference_context" not in parsed_item and "embedding" not in parsed_item:
                    raise ValueError("Each contextual prior must specify at least one of 'pattern', 'reference_context', or 'embedding'.")
                
                self.contextual_priors.append(parsed_item)

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

    def get_prior(self, context_text: str, tool_name: str) -> Tuple[float, float]:
        """
        Retrieve context-specific prior parameters if a matching contextual prior rule exists.
        Falls back to global priors or default (1.0, 1.0).
        """
        if self.contextual_priors:
            query_vector = None
            for prior_item in self.contextual_priors:
                # 1. Regex pattern matching
                pattern = prior_item.get("pattern")
                if pattern is not None:
                    if pattern.search(context_text):
                        if tool_name in prior_item["priors"]:
                            return prior_item["priors"][tool_name]
                        continue

                # 2. Embedding similarity matching
                if self.embedder is not None:
                    ref_vector = None
                    if "embedding" in prior_item:
                        ref_vector = prior_item["embedding"]
                    elif "reference_context" in prior_item:
                        if "_embedding" not in prior_item:
                            ref_ctx = prior_item["reference_context"]
                            try:
                                prior_item["_embedding"] = np.array(
                                    self.embedder.embed_query(ref_ctx), dtype=np.float32
                                )
                            except Exception as e:
                                logger.warning(f"Failed to generate embedding for reference context '{ref_ctx}': {e}")
                                prior_item["_embedding"] = None
                        ref_vector = prior_item["_embedding"]

                    if ref_vector is not None:
                        if query_vector is None:
                            try:
                                query_vector = np.array(
                                    self.embedder.embed_query(context_text), dtype=np.float32
                                )
                            except Exception as e:
                                logger.warning(f"Failed to generate embedding for query context '{context_text}': {e}")
                                query_vector = np.array([], dtype=np.float32)

                        if len(query_vector) > 0 and len(ref_vector) > 0:
                            q_norm = np.linalg.norm(query_vector)
                            r_norm = np.linalg.norm(ref_vector)
                            if q_norm > 0.0 and r_norm > 0.0:
                                similarity = float(np.dot(query_vector, ref_vector) / (q_norm * r_norm))
                                threshold = prior_item.get("similarity_threshold", self.similarity_threshold)
                                if similarity >= threshold:
                                    if tool_name in prior_item["priors"]:
                                        return prior_item["priors"][tool_name]
                                    continue
        # Fall back to global priors
        return self.priors.get(tool_name, (1.0, 1.0))

    def _generate_trace_id(self, context_key: str, tool_name: str) -> str:
        """Encodes context key and tool name into a stateless token and signs it using HMAC."""
        payload = {
            "ctx": context_key,
            "tool": tool_name,
            "nonce": uuid.uuid4().hex,
        }
        json_bytes = json.dumps(payload).encode("utf-8")
        payload_b64 = base64.urlsafe_b64encode(json_bytes).decode("utf-8")
        
        # Compute HMAC signature over the payload
        signature = hmac.new(self.secret_key, payload_b64.encode("utf-8"), hashlib.sha256).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode("utf-8")
        
        return f"{payload_b64}.{signature_b64}"

    def _decode_trace_id(self, trace_id: str) -> Tuple[str, str]:
        """Decodes and verifies context key and tool name from a signed trace ID token."""
        try:
            if "." not in trace_id:
                raise ValueError("Missing signature in trace ID")
            
            payload_b64, signature_b64 = trace_id.rsplit(".", 1)
            
            # Verify signature
            expected_sig = hmac.new(self.secret_key, payload_b64.encode("utf-8"), hashlib.sha256).digest()
            expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode("utf-8")
            
            if not hmac.compare_digest(signature_b64.encode("utf-8"), expected_sig_b64.encode("utf-8")):
                raise ValueError("Invalid trace ID signature")
                
            json_bytes = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
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

        try:
            if self.mode == "clustering":
                context_key = self._resolve_context_key(context_text)
                best_tool = None
                highest_sample = -1.0

                for tool_name in candidate_tools:
                    alpha, beta = self.storage.get_tool_params(context_key, tool_name)

                    # Check for seeded priors on cold start
                    if alpha == 1.0 and beta == 1.0:
                        prior_alpha, prior_beta = self.get_prior(context_text, tool_name)
                        if prior_alpha != 1.0 or prior_beta != 1.0:
                            alpha, beta = prior_alpha, prior_beta
                            self.storage.update_tool_params(context_key, tool_name, alpha, beta)

                    # Sample belief matching beta-binomial posterior
                    sampled_score = np.random.beta(alpha, beta)

                    if sampled_score > highest_sample:
                        highest_sample = sampled_score
                        best_tool = tool_name

                if best_tool is None:
                    best_tool = candidate_tools[0]

                trace_id = self._generate_trace_id(context_key, best_tool)
                self.storage.log_selection(trace_id, context_key, best_tool)
                return best_tool, trace_id

            else:
                if self.embedder is None:
                    raise ValueError("embedder is required for linear bandit mode")
                x = np.array(self.embedder.embed_query(context_text), dtype=np.float32)
                d = len(x)
                context_key = self._resolve_context_key(context_text)
                x_augmented = np.append(x, 1.0)
                d_aug = d + 1
                
                best_tool = None
                highest_score = -float("inf")
                
                for tool_name in candidate_tools:
                    prior_alpha, prior_beta = self.get_prior(context_text, tool_name)
                    prior_p = prior_alpha / (prior_alpha + prior_beta)
                    
                    precision, reward_vector = self.storage.get_linear_params(tool_name)
                    if precision is None or reward_vector is None:
                        precision = self.lambda_val * np.ones(d_aug, dtype=np.float32) if self.diagonal_covariance else self.lambda_val * np.eye(d_aug, dtype=np.float32)
                        reward_vector = np.zeros(d_aug, dtype=np.float32)
                        reward_vector[-1] = self.lambda_val * prior_p

                    if self.diagonal_covariance:
                        theta_hat = reward_vector / precision
                    else:
                        theta_hat = np.linalg.solve(precision, reward_vector)

                    if self.mode == "lints":
                        if self.diagonal_covariance:
                            std_devs = self.exploration_weight / np.sqrt(precision)
                            theta_sample = np.random.normal(theta_hat, std_devs)
                        else:
                            cov = np.linalg.inv(precision)
                            cov = 0.5 * (cov + cov.T)
                            try:
                                L = np.linalg.cholesky(cov)
                                z = np.random.normal(size=d_aug)
                                theta_sample = theta_hat + self.exploration_weight * np.dot(L, z)
                            except np.linalg.LinAlgError:
                                theta_sample = np.random.multivariate_normal(
                                    theta_hat, (self.exploration_weight ** 2) * cov
                                )
                        score = float(np.dot(x_augmented, theta_sample))
                    else:
                        expected_reward = float(np.dot(x_augmented, theta_hat))
                        if self.diagonal_covariance:
                            uncertainty = np.sqrt(np.sum((x_augmented ** 2) / precision))
                        else:
                            cov = np.linalg.inv(precision)
                            uncertainty = np.sqrt(np.dot(x_augmented, np.dot(cov, x_augmented)))
                        score = expected_reward + self.exploration_weight * uncertainty

                    if score > highest_score:
                        highest_score = score
                        best_tool = tool_name

                if best_tool is None:
                    best_tool = candidate_tools[0]

                trace_id = self._generate_trace_id(context_key, best_tool)
                self.storage.log_selection(trace_id, context_key, best_tool)
                return best_tool, trace_id

        except Exception as e:
            logger.exception(
                "BayesianToolRouter routing failed. Triggering fail-safe fallback."
            )
            if self.telemetry_hook:
                try:
                    self.telemetry_hook(
                        "route_failure",
                        e,
                        {
                            "context_text": context_text,
                            "candidate_tools": candidate_tools,
                        },
                    )
                except Exception as hook_err:
                    logger.error(f"Telemetry hook failed: {hook_err}")

            if self.fallback_tool and self.fallback_tool in candidate_tools:
                fallback_choice = self.fallback_tool
            else:
                fallback_choice = candidate_tools[0]

            fallback_trace_id = self._generate_trace_id("fallback_ctx", fallback_choice)
            self.storage.log_selection(fallback_trace_id, "fallback_ctx", fallback_choice)
            return fallback_choice, fallback_trace_id

    def feedback(
        self,
        context_text: str,
        tool_name: str,
        success: Optional[bool] = None,
        reward: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Directly submit tool execution feedback using the raw context string.
        Either success (boolean) or reward (float between 0.0 and 1.0) must be provided.
        """
        if success is None and reward is None:
            raise ValueError("Either 'success' or 'reward' must be provided.")

        if success is not None and reward is not None:
            expected_reward = 1.0 if success else 0.0
            if reward != expected_reward:
                raise ValueError(
                    f"Conflicting feedback: success={success} and reward={reward}. "
                    "Please provide only one, or ensure they are consistent."
                )

        if reward is not None:
            if not (0.0 <= reward <= 1.0):
                raise ValueError("reward must be between 0.0 and 1.0 inclusive")
            reward_val = float(reward)
        else:
            reward_val = 1.0 if success else 0.0

        try:
            if self.mode == "clustering":
                context_key = self._resolve_context_key(context_text)
                return self.storage.decay_and_update(
                    context_key, tool_name, self.decay_factor, reward_val
                )
            else:
                if self.embedder is None:
                    raise ValueError("embedder is required for linear bandit mode")
                x = np.array(self.embedder.embed_query(context_text), dtype=np.float32)
                context_key = self._resolve_context_key(context_text)
                x_augmented = np.append(x, 1.0)
                
                if tool_name in self.priors:
                    alpha, beta = self.priors[tool_name]
                    prior_p = alpha / (alpha + beta)
                else:
                    prior_p = 0.5
                
                precision, reward_vector = self.storage.decay_and_update_linear(
                    tool_name=tool_name,
                    decay_factor=self.decay_factor,
                    reward=reward_val,
                    x_augmented=x_augmented,
                    lambda_val=self.lambda_val,
                    prior_p=prior_p,
                    diagonal=self.diagonal_covariance,
                )
                
                if self.diagonal_covariance:
                    theta_hat = reward_vector / precision
                    expected_reward = float(np.dot(x_augmented, theta_hat))
                    uncertainty = float(np.sqrt(np.sum((x_augmented ** 2) / precision)))
                else:
                    theta_hat = np.linalg.solve(precision, reward_vector)
                    expected_reward = float(np.dot(x_augmented, theta_hat))
                    cov = np.linalg.inv(precision)
                    uncertainty = float(np.sqrt(np.dot(x_augmented, np.dot(cov, x_augmented))))
                
                return expected_reward, uncertainty

        except Exception as e:
            logger.exception("BayesianToolRouter feedback submission failed.")
            if self.telemetry_hook:
                try:
                    self.telemetry_hook(
                        "feedback_failure",
                        e,
                        {
                            "context_text": context_text,
                            "tool_name": tool_name,
                            "success": success,
                            "reward": reward,
                        },
                    )
                except Exception as hook_err:
                    logger.error(f"Telemetry hook failed: {hook_err}")
            return 1.0, 1.0

    def feedback_by_trace(
        self,
        trace_id: str,
        success: Optional[bool] = None,
        reward: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Directly submit tool execution feedback using a generated trace ID.
        Ideal for asynchronous and decoupled systems.
        Either success (boolean) or reward (float between 0.0 and 1.0) must be provided.
        """
        if success is None and reward is None:
            raise ValueError("Either 'success' or 'reward' must be provided.")

        if success is not None and reward is not None:
            expected_reward = 1.0 if success else 0.0
            if reward != expected_reward:
                raise ValueError(
                    f"Conflicting feedback: success={success} and reward={reward}. "
                    "Please provide only one, or ensure they are consistent."
                )

        if reward is not None:
            if not (0.0 <= reward <= 1.0):
                raise ValueError("reward must be between 0.0 and 1.0 inclusive")
            reward_val = float(reward)
        else:
            reward_val = 1.0 if success else 0.0

        try:
            context_key, tool_name = self._decode_trace_id(trace_id)
            self.storage.log_feedback(trace_id, reward_val)
            if self.mode == "clustering":
                return self.storage.decay_and_update(
                    context_key, tool_name, self.decay_factor, reward_val
                )
            else:
                x_seq = self._context_store.get_context_vector(context_key)
                if x_seq is None:
                    logger.warning(
                        f"Context vector not found for key {context_key}. Using zero vector as fallback."
                    )
                    d = 384
                    precision, _ = self.storage.get_linear_params(tool_name)
                    if precision is not None:
                        d = len(precision) - 1
                    x = np.zeros(d, dtype=np.float32)
                else:
                    x = np.array(x_seq, dtype=np.float32)
                
                x_augmented = np.append(x, 1.0)
                
                if tool_name in self.priors:
                    alpha, beta = self.priors[tool_name]
                    prior_p = alpha / (alpha + beta)
                else:
                    prior_p = 0.5
                
                precision, reward_vector = self.storage.decay_and_update_linear(
                    tool_name=tool_name,
                    decay_factor=self.decay_factor,
                    reward=reward_val,
                    x_augmented=x_augmented,
                    lambda_val=self.lambda_val,
                    prior_p=prior_p,
                    diagonal=self.diagonal_covariance,
                )
                
                if self.diagonal_covariance:
                    theta_hat = reward_vector / precision
                    expected_reward = float(np.dot(x_augmented, theta_hat))
                    uncertainty = float(np.sqrt(np.sum((x_augmented ** 2) / precision)))
                else:
                    theta_hat = np.linalg.solve(precision, reward_vector)
                    expected_reward = float(np.dot(x_augmented, theta_hat))
                    cov = np.linalg.inv(precision)
                    uncertainty = float(np.sqrt(np.dot(x_augmented, np.dot(cov, x_augmented))))
                
                return expected_reward, uncertainty

        except Exception as e:
            logger.exception("BayesianToolRouter feedback by trace submission failed.")
            if self.telemetry_hook:
                try:
                    self.telemetry_hook(
                        "feedback_by_trace_failure",
                        e,
                        {
                            "trace_id": trace_id,
                            "success": success,
                            "reward": reward,
                        },
                    )
                except Exception as hook_err:
                    logger.error(f"Telemetry hook failed: {hook_err}")
            return 1.0, 1.0

    def get_tool_beliefs(self, context_text: str, tool_name: str) -> Tuple[float, float]:
        """
        Retrieve current posterior alpha and beta beliefs (or expected reward and uncertainty) for a given context and tool.
        """
        try:
            context_key = self._resolve_context_key(context_text)
            if self.mode == "clustering":
                alpha, beta = self.storage.get_tool_params(context_key, tool_name)
                if alpha == 1.0 and beta == 1.0:
                    alpha, beta = self.get_prior(context_text, tool_name)
                return alpha, beta
            else:
                if self.embedder is None:
                    raise ValueError("embedder is required for linear bandit mode")
                x = np.array(self.embedder.embed_query(context_text), dtype=np.float32)
                x_augmented = np.append(x, 1.0)
                d_aug = len(x_augmented)
                
                prior_alpha, prior_beta = self.get_prior(context_text, tool_name)
                prior_p = prior_alpha / (prior_alpha + prior_beta)
                
                precision, reward_vector = self.storage.get_linear_params(tool_name)
                if precision is None or reward_vector is None:
                    precision = self.lambda_val * np.ones(d_aug, dtype=np.float32) if self.diagonal_covariance else self.lambda_val * np.eye(d_aug, dtype=np.float32)
                    reward_vector = np.zeros(d_aug, dtype=np.float32)
                    reward_vector[-1] = self.lambda_val * prior_p
                
                if self.diagonal_covariance:
                    theta_hat = reward_vector / precision
                    expected_reward = float(np.dot(x_augmented, theta_hat))
                    uncertainty = float(np.sqrt(np.sum((x_augmented ** 2) / precision)))
                else:
                    theta_hat = np.linalg.solve(precision, reward_vector)
                    expected_reward = float(np.dot(x_augmented, theta_hat))
                    cov = np.linalg.inv(precision)
                    uncertainty = float(np.sqrt(np.dot(x_augmented, np.dot(cov, x_augmented))))
                
                return expected_reward, uncertainty
        except Exception as e:
            logger.exception("BayesianToolRouter get_tool_beliefs failed.")
            if self.telemetry_hook:
                try:
                    self.telemetry_hook(
                        "get_tool_beliefs_failure",
                        e,
                        {
                            "context_text": context_text,
                            "tool_name": tool_name,
                        },
                    )
                except Exception as hook_err:
                    logger.error(f"Telemetry hook failed: {hook_err}")
            return 1.0, 1.0

    def _resolve_context_keys(self, contexts: List[str]) -> List[str]:
        if not self.embedder:
            return [self._hash_context_text(ctx) for ctx in contexts]

        try:
            if hasattr(self.embedder, "embed_queries"):
                vectors = self.embedder.embed_queries(contexts)
            else:
                vectors = [self.embedder.embed_query(ctx) for ctx in contexts]
        except Exception:
            logger.warning(
                "Failed to generate embeddings in batch. Falling back to exact-match hashing."
            )
            return [self._hash_context_text(ctx) for ctx in contexts]

        resolved_keys = []
        new_contexts_to_save = []

        for vector in vectors:
            matched_key = self._context_store.get_nearest_context(
                query_vector=vector,
                similarity_threshold=self.similarity_threshold,
            )

            if matched_key is not None:
                resolved_keys.append(matched_key)
            else:
                new_key = f"ctx_{uuid.uuid4().hex}"
                self._context_store.add_context(new_key, vector)
                if not self._custom_vector_store_active:
                    new_contexts_to_save.append((new_key, vector))
                resolved_keys.append(new_key)

        if new_contexts_to_save:
            if hasattr(self.storage, "save_vectors"):
                self.storage.save_vectors(dict(new_contexts_to_save))
            else:
                for k, v in new_contexts_to_save:
                    self.storage.save_vector(k, v)

        return resolved_keys

    def route_batch(self, contexts: List[str], candidate_tools: List[str]) -> List[str]:
        results = self.route_batch_with_trace(contexts, candidate_tools)
        return [tool for tool, _ in results]

    def route_batch_with_trace(
        self, contexts: List[str], candidate_tools: List[str]
    ) -> List[Tuple[str, str]]:
        if not candidate_tools:
            raise ValueError("Candidate tools list cannot be empty")
        if not contexts:
            return []

        try:
            if self.mode == "clustering":
                context_keys = self._resolve_context_keys(contexts)
                
                param_keys = [(ctx_key, tool_name) for ctx_key in context_keys for tool_name in candidate_tools]
                param_dict = self.storage.get_tool_params_batch(param_keys)
                
                priors_to_update = {}
                results = []
                for idx, context_key in enumerate(context_keys):
                    context_text = contexts[idx]
                    best_tool = None
                    highest_sample = -1.0

                    for tool_name in candidate_tools:
                        alpha, beta = param_dict.get((context_key, tool_name), (1.0, 1.0))

                        if alpha == 1.0 and beta == 1.0:
                            prior_alpha, prior_beta = self.get_prior(context_text, tool_name)
                            if prior_alpha != 1.0 or prior_beta != 1.0:
                                alpha, beta = prior_alpha, prior_beta
                                priors_to_update[(context_key, tool_name)] = (alpha, beta)

                        sampled_score = np.random.beta(alpha, beta)

                        if sampled_score > highest_sample:
                            highest_sample = sampled_score
                            best_tool = tool_name

                    if best_tool is None:
                        best_tool = candidate_tools[0]

                    trace_id = self._generate_trace_id(context_key, best_tool)
                    self.storage.log_selection(trace_id, context_key, best_tool)
                    results.append((best_tool, trace_id))

                if priors_to_update:
                    if hasattr(self.storage, "update_tool_params_batch"):
                        self.storage.update_tool_params_batch(priors_to_update)
                    else:
                        for (ctx_key, tool_name), (alpha, beta) in priors_to_update.items():
                            self.storage.update_tool_params(ctx_key, tool_name, alpha, beta)

                return results

            else:
                if self.embedder is None:
                    raise ValueError("embedder is required for linear bandit mode")
                
                if hasattr(self.embedder, "embed_queries"):
                    vectors = self.embedder.embed_queries(contexts)
                else:
                    vectors = [self.embedder.embed_query(ctx) for ctx in contexts]
                
                tool_params = {}
                if hasattr(self.storage, "get_linear_params_batch"):
                    tool_params = self.storage.get_linear_params_batch(candidate_tools)
                else:
                    for tool_name in candidate_tools:
                        tool_params[tool_name] = self.storage.get_linear_params(tool_name)

                context_keys = self._resolve_context_keys(contexts)
                results = []
                for idx, x_seq in enumerate(vectors):
                    x = np.array(x_seq, dtype=np.float32)
                    d = len(x)
                    x_augmented = np.append(x, 1.0)
                    d_aug = d + 1
                    context_key = context_keys[idx]
                    context_text = contexts[idx]
                    
                    best_tool = None
                    highest_score = -float("inf")
                    
                    for tool_name in candidate_tools:
                        prior_alpha, prior_beta = self.get_prior(context_text, tool_name)
                        prior_p = prior_alpha / (prior_alpha + prior_beta)
                        
                        precision, reward_vector = tool_params.get(tool_name, (None, None))
                        if precision is None or reward_vector is None:
                            precision = self.lambda_val * np.ones(d_aug, dtype=np.float32) if self.diagonal_covariance else self.lambda_val * np.eye(d_aug, dtype=np.float32)
                            reward_vector = np.zeros(d_aug, dtype=np.float32)
                            reward_vector[-1] = self.lambda_val * prior_p

                        if self.diagonal_covariance:
                            theta_hat = reward_vector / precision
                        else:
                            theta_hat = np.linalg.solve(precision, reward_vector)

                        if self.mode == "lints":
                            if self.diagonal_covariance:
                                std_devs = self.exploration_weight / np.sqrt(precision)
                                theta_sample = np.random.normal(theta_hat, std_devs)
                            else:
                                cov = np.linalg.inv(precision)
                                cov = 0.5 * (cov + cov.T)
                                try:
                                    L = np.linalg.cholesky(cov)
                                    z = np.random.normal(size=d_aug)
                                    theta_sample = theta_hat + self.exploration_weight * np.dot(L, z)
                                except np.linalg.LinAlgError:
                                    theta_sample = np.random.multivariate_normal(
                                        theta_hat, (self.exploration_weight ** 2) * cov
                                    )
                            score = float(np.dot(x_augmented, theta_sample))
                        else:
                            expected_reward = float(np.dot(x_augmented, theta_hat))
                            if self.diagonal_covariance:
                                uncertainty = np.sqrt(np.sum((x_augmented ** 2) / precision))
                            else:
                                cov = np.linalg.inv(precision)
                                uncertainty = np.sqrt(np.dot(x_augmented, np.dot(cov, x_augmented)))
                            score = expected_reward + self.exploration_weight * uncertainty

                        if score > highest_score:
                            highest_score = score
                            best_tool = tool_name

                    if best_tool is None:
                        best_tool = candidate_tools[0]

                    trace_id = self._generate_trace_id(context_key, best_tool)
                    self.storage.log_selection(trace_id, context_key, best_tool)
                    results.append((best_tool, trace_id))

                return results

        except Exception as e:
            logger.exception("BayesianToolRouter batch routing failed. Triggering fail-safe fallback.")
            if self.telemetry_hook:
                try:
                    self.telemetry_hook(
                        "route_batch_failure",
                        e,
                        {
                            "contexts": contexts,
                            "candidate_tools": candidate_tools,
                        },
                    )
                except Exception as hook_err:
                    logger.error(f"Telemetry hook failed: {hook_err}")

            fallback_choice = self.fallback_tool if (self.fallback_tool and self.fallback_tool in candidate_tools) else candidate_tools[0]
            fallback_trace_id = self._generate_trace_id("fallback_ctx", fallback_choice)
            for _ in contexts:
                self.storage.log_selection(fallback_trace_id, "fallback_ctx", fallback_choice)
            return [(fallback_choice, fallback_trace_id)] * len(contexts)

    def feedback_batch(self, feedbacks: List[Dict[str, Any]]) -> None:
        if not feedbacks:
            return

        try:
            contexts_to_embed = []
            contexts_to_embed_indices = []
            prepared_feedbacks = []
            
            for fb in feedbacks:
                success = fb.get("success")
                reward = fb.get("reward")
                
                if success is None and reward is None:
                    raise ValueError("Either 'success' or 'reward' must be provided in feedback.")
                if success is not None and reward is not None:
                    expected_reward = 1.0 if success else 0.0
                    if reward != expected_reward:
                        raise ValueError(
                            f"Conflicting feedback: success={success} and reward={reward}."
                        )
                
                reward_val = float(reward) if reward is not None else (1.0 if success else 0.0)
                
                trace_id = fb.get("trace_id")
                if trace_id is not None:
                    context_key, tool_name = self._decode_trace_id(trace_id)
                    prepared_feedbacks.append({
                        "type": "trace",
                        "context_key": context_key,
                        "tool_name": tool_name,
                        "reward_val": reward_val,
                    })
                else:
                    context_text = fb.get("context_text")
                    tool_name = fb.get("tool_name")
                    if not context_text or not tool_name:
                        raise ValueError("Feedback must contain either 'trace_id' or both 'context_text' and 'tool_name'.")
                    
                    prepared_feedbacks.append({
                        "type": "text",
                        "context_text": context_text,
                        "tool_name": tool_name,
                        "reward_val": reward_val,
                    })
                    contexts_to_embed.append(context_text)
                    contexts_to_embed_indices.append(len(prepared_feedbacks) - 1)

            if contexts_to_embed:
                resolved_keys = self._resolve_context_keys(contexts_to_embed)
                for idx, key in zip(contexts_to_embed_indices, resolved_keys):
                    prepared_feedbacks[idx]["context_key"] = key
                    
                if self.mode != "clustering":
                    if hasattr(self.embedder, "embed_queries"):
                        vectors = self.embedder.embed_queries(contexts_to_embed)
                    else:
                        vectors = [self.embedder.embed_query(t) for t in contexts_to_embed]
                    for idx, vector in zip(contexts_to_embed_indices, vectors):
                        prepared_feedbacks[idx]["vector"] = vector

            if self.mode == "clustering":
                updates = []
                for fb in prepared_feedbacks:
                    updates.append((fb["context_key"], fb["tool_name"], self.decay_factor, fb["reward_val"]))
                self.storage.decay_and_update_batch(updates)
            else:
                updates = []
                for fb in prepared_feedbacks:
                    tool_name = fb["tool_name"]
                    reward_val = fb["reward_val"]
                    
                    if fb["type"] == "trace":
                        x_seq = self._context_store.get_context_vector(fb["context_key"])
                        if x_seq is None:
                            logger.warning(
                                f"Context vector not found for key {fb['context_key']}. Using zero vector as fallback."
                            )
                            d = 384
                            precision, _ = self.storage.get_linear_params(tool_name)
                            if precision is not None:
                                d = len(precision) - 1
                            x = np.zeros(d, dtype=np.float32)
                        else:
                            x = np.array(x_seq, dtype=np.float32)
                    else:
                        x = np.array(fb["vector"], dtype=np.float32)
                    
                    x_augmented = np.append(x, 1.0)
                    
                    if tool_name in self.priors:
                        alpha, beta = self.priors[tool_name]
                        prior_p = alpha / (alpha + beta)
                    else:
                        prior_p = 0.5
                        
                    updates.append((tool_name, self.decay_factor, reward_val, x_augmented, self.lambda_val, prior_p, self.diagonal_covariance))
                    
                self.storage.decay_and_update_linear_batch(updates)

            # Log trace feedback
            for fb in feedbacks:
                trace_id = fb.get("trace_id")
                if trace_id is not None:
                    reward_val = float(fb.get("reward")) if fb.get("reward") is not None else (1.0 if fb.get("success") else 0.0)
                    self.storage.log_feedback(trace_id, reward_val)

        except Exception as e:
            logger.exception("BayesianToolRouter batch feedback submission failed.")
            if self.telemetry_hook:
                try:
                    self.telemetry_hook("feedback_batch_failure", e, {"feedbacks": feedbacks})
                except Exception as hook_err:
                    logger.error(f"Telemetry hook failed: {hook_err}")



class AsyncBayesianToolRouter:
    """
    Decoupled tool routing middleware implementing a Contextual Multi-Armed Bandit
    via Thompson Sampling with fully asynchronous operation.
    """

    def __init__(
        self,
        storage: Optional[AsyncBaseStorage] = None,
        embedder: Optional[Any] = None,  # Can be ContextEmbedder or AsyncContextEmbedder
        decay_factor: float = 1.0,
        similarity_threshold: float = 0.8,
        priors: Optional[Dict[str, Tuple[float, float]]] = None,
        contextual_priors: Optional[List[Dict[str, Any]]] = None,
        vector_store: Optional[AsyncVectorStoreProtocol] = None,
        fallback_tool: Optional[str] = None,
        telemetry_hook: Optional[Callable[[str, Exception, Dict[str, Any]], Any]] = None,
        mode: str = "clustering",
        exploration_weight: float = 1.0,
        lambda_val: float = 1.0,
        diagonal_covariance: bool = False,
        secret_key: Optional[Union[str, bytes]] = None,
    ) -> None:
        """
        Initialize the AsyncBayesianToolRouter.
        """
        self.storage = storage or AsyncInMemoryStorage()
        self.embedder = embedder
        self.fallback_tool = fallback_tool
        self.telemetry_hook = telemetry_hook
        
        self.mode = mode
        if mode not in ("clustering", "lints", "linucb"):
            raise ValueError("mode must be 'clustering', 'lints', or 'linucb'")
        self.exploration_weight = exploration_weight
        self.lambda_val = lambda_val
        self.diagonal_covariance = diagonal_covariance

        # Determine secret key for signing trace IDs
        if secret_key is not None:
            if isinstance(secret_key, str):
                self.secret_key = secret_key.encode("utf-8")
            else:
                self.secret_key = secret_key
        else:
            env_key = os.environ.get("BAYES_BRAIN_SECRET_KEY")
            if env_key:
                self.secret_key = env_key.encode("utf-8")
            else:
                self.secret_key = os.urandom(32)

        if self.mode in ("lints", "linucb") and self.embedder is None:
            raise ValueError("Linear bandit modes ('lints', 'linucb') require a ContextEmbedder/AsyncContextEmbedder.")

        if embedder is None:
            logger.warning(
                "No ContextEmbedder/AsyncContextEmbedder provided to async router. "
                "Operating in exact-match fallback mode."
            )
        
        if not (0.0 < decay_factor <= 1.0):
            raise ValueError("decay_factor must be in the range (0, 1]")
        self.decay_factor = decay_factor
        self.similarity_threshold = similarity_threshold
        self.priors = priors or {}

        # Validate and parse contextual priors
        self.contextual_priors = []
        if contextual_priors:
            for item in contextual_priors:
                parsed_item = {}
                if "priors" not in item or not isinstance(item["priors"], dict):
                    raise ValueError("Each contextual prior must contain a 'priors' dictionary.")
                
                priors_map = {}
                for t_name, params in item["priors"].items():
                    if not isinstance(params, (list, tuple)) or len(params) != 2:
                        raise ValueError(f"Prior parameters for tool '{t_name}' must be a tuple/list of (alpha, beta).")
                    priors_map[t_name] = (float(params[0]), float(params[1]))
                parsed_item["priors"] = priors_map

                if "pattern" in item:
                    if not isinstance(item["pattern"], str):
                        raise ValueError("Contextual prior pattern must be a regex string.")
                    try:
                        parsed_item["pattern"] = re.compile(item["pattern"])
                    except re.error as e:
                        raise ValueError(f"Invalid regex pattern '{item['pattern']}': {e}")
                
                if "reference_context" in item:
                    if not isinstance(item["reference_context"], str):
                        raise ValueError("Contextual prior reference_context must be a string.")
                    parsed_item["reference_context"] = item["reference_context"]
                
                if "embedding" in item:
                    if not isinstance(item["embedding"], (list, tuple, np.ndarray)):
                        raise ValueError("Contextual prior embedding must be a list/tuple/numpy array of floats.")
                    parsed_item["embedding"] = np.array(item["embedding"], dtype=np.float32)
                
                if "similarity_threshold" in item:
                    parsed_item["similarity_threshold"] = float(item["similarity_threshold"])
                
                if "pattern" not in parsed_item and "reference_context" not in parsed_item and "embedding" not in parsed_item:
                    raise ValueError("Each contextual prior must specify at least one of 'pattern', 'reference_context', or 'embedding'.")
                
                self.contextual_priors.append(parsed_item)

        self._custom_vector_store_active = vector_store is not None
        if vector_store is not None:
            self._context_store = vector_store
        else:
            self._context_store = AsyncVectorContextStore()

        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._load_context_store()
            self._initialized = True

    async def _load_context_store(self) -> None:
        """Attempt to restore the VectorContextStore from the storage backend."""
        if self._custom_vector_store_active:
            return
        try:
            vectors = await self.storage.load_all_vectors()
            for key, vector in vectors.items():
                await self._context_store.aadd_context(key, vector)
        except Exception:
            pass

    async def _save_context_store(self) -> None:
        """Persist the VectorContextStore to the storage backend."""
        if self._custom_vector_store_active:
            return
        try:
            if hasattr(self._context_store, "_contexts"):
                for key, vector in self._context_store._contexts.items():
                    await self.storage.save_vector(key, vector)
        except Exception:
            pass

    def _hash_context_text(self, context_text: str) -> str:
        normalized = " ".join(context_text.strip().split())
        sha256_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"hash_{sha256_hash}"

    async def _resolve_context_key(self, context_text: str) -> str:
        if not self.embedder:
            return self._hash_context_text(context_text)

        try:
            if hasattr(self.embedder, "aembed_query"):
                vector = await self.embedder.aembed_query(context_text)
            else:
                vector = self.embedder.embed_query(context_text)
        except Exception:
            logger.warning(
                "Failed to generate embedding for context. Falling back to exact-match hashing."
            )
            return self._hash_context_text(context_text)

        matched_key = await self._context_store.aget_nearest_context(
            query_vector=vector,
            similarity_threshold=self.similarity_threshold,
        )

        if matched_key is not None:
            return matched_key

        new_key = f"ctx_{uuid.uuid4().hex}"
        await self._context_store.aadd_context(new_key, vector)
        if not self._custom_vector_store_active:
            await self.storage.save_vector(new_key, vector)
        return new_key

    async def get_prior(self, context_text: str, tool_name: str) -> Tuple[float, float]:
        """
        Retrieve context-specific prior parameters if a matching contextual prior rule exists.
        Falls back to global priors or default (1.0, 1.0).
        """
        if self.contextual_priors:
            query_vector = None
            for prior_item in self.contextual_priors:
                # 1. Regex pattern matching
                pattern = prior_item.get("pattern")
                if pattern is not None:
                    if pattern.search(context_text):
                        if tool_name in prior_item["priors"]:
                            return prior_item["priors"][tool_name]
                        continue

                # 2. Embedding similarity matching
                if self.embedder is not None:
                    ref_vector = None
                    if "embedding" in prior_item:
                        ref_vector = prior_item["embedding"]
                    elif "reference_context" in prior_item:
                        if "_embedding" not in prior_item:
                            ref_ctx = prior_item["reference_context"]
                            try:
                                if hasattr(self.embedder, "aembed_query"):
                                    vector = await self.embedder.aembed_query(ref_ctx)
                                else:
                                    vector = self.embedder.embed_query(ref_ctx)
                                prior_item["_embedding"] = np.array(vector, dtype=np.float32)
                            except Exception as e:
                                logger.warning(f"Failed to generate embedding for reference context '{ref_ctx}': {e}")
                                prior_item["_embedding"] = None
                        ref_vector = prior_item["_embedding"]

                    if ref_vector is not None:
                        if query_vector is None:
                            try:
                                if hasattr(self.embedder, "aembed_query"):
                                    vector = await self.embedder.aembed_query(context_text)
                                else:
                                    vector = self.embedder.embed_query(context_text)
                                query_vector = np.array(vector, dtype=np.float32)
                            except Exception as e:
                                logger.warning(f"Failed to generate embedding for query context '{context_text}': {e}")
                                query_vector = np.array([], dtype=np.float32)

                        if len(query_vector) > 0 and len(ref_vector) > 0:
                            q_norm = np.linalg.norm(query_vector)
                            r_norm = np.linalg.norm(ref_vector)
                            if q_norm > 0.0 and r_norm > 0.0:
                                similarity = float(np.dot(query_vector, ref_vector) / (q_norm * r_norm))
                                threshold = prior_item.get("similarity_threshold", self.similarity_threshold)
                                if similarity >= threshold:
                                    if tool_name in prior_item["priors"]:
                                        return prior_item["priors"][tool_name]
                                    continue
        # Fall back to global priors
        return self.priors.get(tool_name, (1.0, 1.0))

    def _generate_trace_id(self, context_key: str, tool_name: str) -> str:
        """Encodes context key and tool name into a stateless token and signs it using HMAC."""
        payload = {
            "ctx": context_key,
            "tool": tool_name,
            "nonce": uuid.uuid4().hex,
        }
        json_bytes = json.dumps(payload).encode("utf-8")
        payload_b64 = base64.urlsafe_b64encode(json_bytes).decode("utf-8")
        
        # Compute HMAC signature over the payload
        signature = hmac.new(self.secret_key, payload_b64.encode("utf-8"), hashlib.sha256).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode("utf-8")
        
        return f"{payload_b64}.{signature_b64}"

    def _decode_trace_id(self, trace_id: str) -> Tuple[str, str]:
        """Decodes and verifies context key and tool name from a signed trace ID token."""
        try:
            if "." not in trace_id:
                raise ValueError("Missing signature in trace ID")
            
            payload_b64, signature_b64 = trace_id.rsplit(".", 1)
            
            # Verify signature
            expected_sig = hmac.new(self.secret_key, payload_b64.encode("utf-8"), hashlib.sha256).digest()
            expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode("utf-8")
            
            if not hmac.compare_digest(signature_b64.encode("utf-8"), expected_sig_b64.encode("utf-8")):
                raise ValueError("Invalid trace ID signature")
                
            json_bytes = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
            payload = json.loads(json_bytes.decode("utf-8"))
            return payload["ctx"], payload["tool"]
        except Exception as e:
            raise ValueError(f"Invalid or corrupted trace ID: {trace_id}") from e

    async def _call_telemetry(self, event: str, exc: Exception, ctx: Dict[str, Any]) -> None:
        if not self.telemetry_hook:
            return
        try:
            if asyncio.iscoroutinefunction(self.telemetry_hook):
                await self.telemetry_hook(event, exc, ctx)
            else:
                self.telemetry_hook(event, exc, ctx)
        except Exception as hook_err:
            logger.error(f"Telemetry hook failed: {hook_err}")

    async def aroute(self, context_text: str, candidate_tools: List[str]) -> str:
        chosen_tool, _ = await self.aroute_with_trace(context_text, candidate_tools)
        return chosen_tool

    async def aroute_with_trace(
        self, context_text: str, candidate_tools: List[str]
    ) -> Tuple[str, str]:
        if not candidate_tools:
            raise ValueError("Candidate tools list cannot be empty")

        await self._ensure_initialized()

        try:
            if self.mode == "clustering":
                context_key = await self._resolve_context_key(context_text)
                best_tool = None
                highest_sample = -1.0

                for tool_name in candidate_tools:
                    alpha, beta = await self.storage.get_tool_params(context_key, tool_name)

                    if alpha == 1.0 and beta == 1.0:
                        prior_alpha, prior_beta = await self.get_prior(context_text, tool_name)
                        if prior_alpha != 1.0 or prior_beta != 1.0:
                            alpha, beta = prior_alpha, prior_beta
                            await self.storage.update_tool_params(context_key, tool_name, alpha, beta)

                    sampled_score = np.random.beta(alpha, beta)

                    if sampled_score > highest_sample:
                        highest_sample = sampled_score
                        best_tool = tool_name

                if best_tool is None:
                    best_tool = candidate_tools[0]

                trace_id = self._generate_trace_id(context_key, best_tool)
                await self.storage.log_selection(trace_id, context_key, best_tool)
                return best_tool, trace_id

            else:
                if self.embedder is None:
                    raise ValueError("embedder is required for linear bandit mode")
                
                if hasattr(self.embedder, "aembed_query"):
                    x_seq = await self.embedder.aembed_query(context_text)
                else:
                    x_seq = self.embedder.embed_query(context_text)
                
                x = np.array(x_seq, dtype=np.float32)
                d = len(x)
                context_key = await self._resolve_context_key(context_text)
                x_augmented = np.append(x, 1.0)
                d_aug = d + 1
                
                best_tool = None
                highest_score = -float("inf")
                
                for tool_name in candidate_tools:
                    prior_alpha, prior_beta = await self.get_prior(context_text, tool_name)
                    prior_p = prior_alpha / (prior_alpha + prior_beta)
                    
                    precision, reward_vector = await self.storage.aget_linear_params(tool_name)
                    if precision is None or reward_vector is None:
                        precision = self.lambda_val * np.ones(d_aug, dtype=np.float32) if self.diagonal_covariance else self.lambda_val * np.eye(d_aug, dtype=np.float32)
                        reward_vector = np.zeros(d_aug, dtype=np.float32)
                        reward_vector[-1] = self.lambda_val * prior_p

                    if self.diagonal_covariance:
                        theta_hat = reward_vector / precision
                    else:
                        theta_hat = np.linalg.solve(precision, reward_vector)

                    if self.mode == "lints":
                        if self.diagonal_covariance:
                            std_devs = self.exploration_weight / np.sqrt(precision)
                            theta_sample = np.random.normal(theta_hat, std_devs)
                        else:
                            cov = np.linalg.inv(precision)
                            cov = 0.5 * (cov + cov.T)
                            try:
                                L = np.linalg.cholesky(cov)
                                z = np.random.normal(size=d_aug)
                                theta_sample = theta_hat + self.exploration_weight * np.dot(L, z)
                            except np.linalg.LinAlgError:
                                theta_sample = np.random.multivariate_normal(
                                    theta_hat, (self.exploration_weight ** 2) * cov
                                )
                        score = float(np.dot(x_augmented, theta_sample))
                    else:
                        expected_reward = float(np.dot(x_augmented, theta_hat))
                        if self.diagonal_covariance:
                            uncertainty = np.sqrt(np.sum((x_augmented ** 2) / precision))
                        else:
                            cov = np.linalg.inv(precision)
                            uncertainty = np.sqrt(np.dot(x_augmented, np.dot(cov, x_augmented)))
                        score = expected_reward + self.exploration_weight * uncertainty

                    if score > highest_score:
                        highest_score = score
                        best_tool = tool_name

                if best_tool is None:
                    best_tool = candidate_tools[0]

                trace_id = self._generate_trace_id(context_key, best_tool)
                await self.storage.log_selection(trace_id, context_key, best_tool)
                return best_tool, trace_id

        except Exception as e:
            logger.exception(
                "AsyncBayesianToolRouter routing failed. Triggering fail-safe fallback."
            )
            await self._call_telemetry(
                "route_failure",
                e,
                {
                    "context_text": context_text,
                    "candidate_tools": candidate_tools,
                },
            )

            if self.fallback_tool and self.fallback_tool in candidate_tools:
                fallback_choice = self.fallback_tool
            else:
                fallback_choice = candidate_tools[0]

            fallback_trace_id = self._generate_trace_id("fallback_ctx", fallback_choice)
            await self.storage.log_selection(fallback_trace_id, "fallback_ctx", fallback_choice)
            return fallback_choice, fallback_trace_id

    async def afeedback(
        self,
        context_text: str,
        tool_name: str,
        success: Optional[bool] = None,
        reward: Optional[float] = None,
    ) -> Tuple[float, float]:
        if success is None and reward is None:
            raise ValueError("Either 'success' or 'reward' must be provided.")

        if success is not None and reward is not None:
            expected_reward = 1.0 if success else 0.0
            if reward != expected_reward:
                raise ValueError(
                    f"Conflicting feedback: success={success} and reward={reward}. "
                    "Please provide only one, or ensure they are consistent."
                )

        if reward is not None:
            if not (0.0 <= reward <= 1.0):
                raise ValueError("reward must be between 0.0 and 1.0 inclusive")
            reward_val = float(reward)
        else:
            reward_val = 1.0 if success else 0.0

        await self._ensure_initialized()

        try:
            if self.mode == "clustering":
                context_key = await self._resolve_context_key(context_text)
                return await self.storage.decay_and_update(
                    context_key, tool_name, self.decay_factor, reward_val
                )
            else:
                if self.embedder is None:
                    raise ValueError("embedder is required for linear bandit mode")
                
                if hasattr(self.embedder, "aembed_query"):
                    x_seq = await self.embedder.aembed_query(context_text)
                else:
                    x_seq = self.embedder.embed_query(context_text)
                    
                x = np.array(x_seq, dtype=np.float32)
                context_key = await self._resolve_context_key(context_text)
                x_augmented = np.append(x, 1.0)
                
                if tool_name in self.priors:
                    alpha, beta = self.priors[tool_name]
                    prior_p = alpha / (alpha + beta)
                else:
                    prior_p = 0.5
                
                precision, reward_vector = await self.storage.adecay_and_update_linear(
                    tool_name=tool_name,
                    decay_factor=self.decay_factor,
                    reward=reward_val,
                    x_augmented=x_augmented,
                    lambda_val=self.lambda_val,
                    prior_p=prior_p,
                    diagonal=self.diagonal_covariance,
                )
                
                if self.diagonal_covariance:
                    theta_hat = reward_vector / precision
                    expected_reward = float(np.dot(x_augmented, theta_hat))
                    uncertainty = float(np.sqrt(np.sum((x_augmented ** 2) / precision)))
                else:
                    theta_hat = np.linalg.solve(precision, reward_vector)
                    expected_reward = float(np.dot(x_augmented, theta_hat))
                    cov = np.linalg.inv(precision)
                    uncertainty = float(np.sqrt(np.dot(x_augmented, np.dot(cov, x_augmented))))
                
                return expected_reward, uncertainty

        except Exception as e:
            logger.exception("AsyncBayesianToolRouter feedback submission failed.")
            await self._call_telemetry(
                "feedback_failure",
                e,
                {
                    "context_text": context_text,
                    "tool_name": tool_name,
                    "success": success,
                    "reward": reward,
                },
            )
            return 1.0, 1.0

    async def afeedback_by_trace(
        self,
        trace_id: str,
        success: Optional[bool] = None,
        reward: Optional[float] = None,
    ) -> Tuple[float, float]:
        if success is None and reward is None:
            raise ValueError("Either 'success' or 'reward' must be provided.")

        if success is not None and reward is not None:
            expected_reward = 1.0 if success else 0.0
            if reward != expected_reward:
                raise ValueError(
                    f"Conflicting feedback: success={success} and reward={reward}. "
                    "Please provide only one, or ensure they are consistent."
                )

        if reward is not None:
            if not (0.0 <= reward <= 1.0):
                raise ValueError("reward must be between 0.0 and 1.0 inclusive")
            reward_val = float(reward)
        else:
            reward_val = 1.0 if success else 0.0

        await self._ensure_initialized()

        try:
            context_key, tool_name = self._decode_trace_id(trace_id)
            await self.storage.log_feedback(trace_id, reward_val)
            if self.mode == "clustering":
                return await self.storage.decay_and_update(
                    context_key, tool_name, self.decay_factor, reward_val
                )
            else:
                x_seq = await self._context_store.aget_context_vector(context_key)
                if x_seq is None:
                    logger.warning(
                        f"Context vector not found for key {context_key}. Using zero vector as fallback."
                    )
                    d = 384
                    precision, _ = await self.storage.aget_linear_params(tool_name)
                    if precision is not None:
                        d = len(precision) - 1
                    x = np.zeros(d, dtype=np.float32)
                else:
                    x = np.array(x_seq, dtype=np.float32)
                
                x_augmented = np.append(x, 1.0)
                
                if tool_name in self.priors:
                    alpha, beta = self.priors[tool_name]
                    prior_p = alpha / (alpha + beta)
                else:
                    prior_p = 0.5
                
                precision, reward_vector = await self.storage.adecay_and_update_linear(
                    tool_name=tool_name,
                    decay_factor=self.decay_factor,
                    reward=reward_val,
                    x_augmented=x_augmented,
                    lambda_val=self.lambda_val,
                    prior_p=prior_p,
                    diagonal=self.diagonal_covariance,
                )
                
                if self.diagonal_covariance:
                    theta_hat = reward_vector / precision
                    expected_reward = float(np.dot(x_augmented, theta_hat))
                    uncertainty = float(np.sqrt(np.sum((x_augmented ** 2) / precision)))
                else:
                    theta_hat = np.linalg.solve(precision, reward_vector)
                    expected_reward = float(np.dot(x_augmented, theta_hat))
                    cov = np.linalg.inv(precision)
                    uncertainty = float(np.sqrt(np.dot(x_augmented, np.dot(cov, x_augmented))))
                
                return expected_reward, uncertainty

        except Exception as e:
            logger.exception("AsyncBayesianToolRouter feedback by trace submission failed.")
            await self._call_telemetry(
                "feedback_by_trace_failure",
                e,
                {
                    "trace_id": trace_id,
                    "success": success,
                    "reward": reward,
                },
            )
            return 1.0, 1.0

    async def aget_tool_beliefs(self, context_text: str, tool_name: str) -> Tuple[float, float]:
        await self._ensure_initialized()
        try:
            context_key = await self._resolve_context_key(context_text)
            if self.mode == "clustering":
                alpha, beta = await self.storage.get_tool_params(context_key, tool_name)
                if alpha == 1.0 and beta == 1.0:
                    alpha, beta = await self.get_prior(context_text, tool_name)
                return alpha, beta
            else:
                if self.embedder is None:
                    raise ValueError("embedder is required for linear bandit mode")
                
                if hasattr(self.embedder, "aembed_query"):
                    x_seq = await self.embedder.aembed_query(context_text)
                else:
                    x_seq = self.embedder.embed_query(context_text)
                    
                x = np.array(x_seq, dtype=np.float32)
                x_augmented = np.append(x, 1.0)
                d_aug = len(x_augmented)
                
                prior_alpha, prior_beta = await self.get_prior(context_text, tool_name)
                prior_p = prior_alpha / (prior_alpha + prior_beta)
                
                precision, reward_vector = await self.storage.aget_linear_params(tool_name)
                if precision is None or reward_vector is None:
                    precision = self.lambda_val * np.ones(d_aug, dtype=np.float32) if self.diagonal_covariance else self.lambda_val * np.eye(d_aug, dtype=np.float32)
                    reward_vector = np.zeros(d_aug, dtype=np.float32)
                    reward_vector[-1] = self.lambda_val * prior_p
                
                if self.diagonal_covariance:
                    theta_hat = reward_vector / precision
                    expected_reward = float(np.dot(x_augmented, theta_hat))
                    uncertainty = float(np.sqrt(np.sum((x_augmented ** 2) / precision)))
                else:
                    theta_hat = np.linalg.solve(precision, reward_vector)
                    expected_reward = float(np.dot(x_augmented, theta_hat))
                    cov = np.linalg.inv(precision)
                    uncertainty = float(np.sqrt(np.dot(x_augmented, np.dot(cov, x_augmented))))
                
                return expected_reward, uncertainty
        except Exception as e:
            logger.exception("AsyncBayesianToolRouter aget_tool_beliefs failed.")
            await self._call_telemetry(
                "get_tool_beliefs_failure",
                e,
                {
                    "context_text": context_text,
                    "tool_name": tool_name,
                },
            )
            return 1.0, 1.0

    async def _resolve_context_keys(self, contexts: List[str]) -> List[str]:
        if not self.embedder:
            return [self._hash_context_text(ctx) for ctx in contexts]

        try:
            if hasattr(self.embedder, "aembed_queries"):
                vectors = await self.embedder.aembed_queries(contexts)
            elif hasattr(self.embedder, "embed_queries"):
                vectors = self.embedder.embed_queries(contexts)
            elif hasattr(self.embedder, "aembed_query"):
                vectors = await asyncio.gather(*(self.embedder.aembed_query(ctx) for ctx in contexts))
            else:
                vectors = [self.embedder.embed_query(ctx) for ctx in contexts]
        except Exception:
            logger.warning(
                "Failed to generate embeddings in batch. Falling back to exact-match hashing."
            )
            return [self._hash_context_text(ctx) for ctx in contexts]

        resolved_keys = []
        new_contexts_to_save = []

        for vector in vectors:
            matched_key = await self._context_store.aget_nearest_context(
                query_vector=vector,
                similarity_threshold=self.similarity_threshold,
            )

            if matched_key is not None:
                resolved_keys.append(matched_key)
            else:
                new_key = f"ctx_{uuid.uuid4().hex}"
                await self._context_store.aadd_context(new_key, vector)
                if not self._custom_vector_store_active:
                    new_contexts_to_save.append((new_key, vector))
                resolved_keys.append(new_key)

        if new_contexts_to_save:
            if hasattr(self.storage, "asave_vectors"):
                await self.storage.asave_vectors(dict(new_contexts_to_save))
            elif hasattr(self.storage, "save_vectors"):
                await self.storage.save_vectors(dict(new_contexts_to_save))
            else:
                for k, v in new_contexts_to_save:
                    await self.storage.save_vector(k, v)

        return resolved_keys

    async def aroute_batch(self, contexts: List[str], candidate_tools: List[str]) -> List[str]:
        results = await self.aroute_batch_with_trace(contexts, candidate_tools)
        return [tool for tool, _ in results]

    async def aroute_batch_with_trace(
        self, contexts: List[str], candidate_tools: List[str]
    ) -> List[Tuple[str, str]]:
        if not candidate_tools:
            raise ValueError("Candidate tools list cannot be empty")
        if not contexts:
            return []

        await self._ensure_initialized()

        try:
            if self.mode == "clustering":
                context_keys = await self._resolve_context_keys(contexts)
                
                param_keys = [(ctx_key, tool_name) for ctx_key in context_keys for tool_name in candidate_tools]
                param_dict = await self.storage.get_tool_params_batch(param_keys)
                
                priors_to_update = {}
                results = []
                for idx, context_key in enumerate(context_keys):
                    context_text = contexts[idx]
                    best_tool = None
                    highest_sample = -1.0

                    for tool_name in candidate_tools:
                        alpha, beta = param_dict.get((context_key, tool_name), (1.0, 1.0))

                        if alpha == 1.0 and beta == 1.0:
                            prior_alpha, prior_beta = await self.get_prior(context_text, tool_name)
                            if prior_alpha != 1.0 or prior_beta != 1.0:
                                alpha, beta = prior_alpha, prior_beta
                                priors_to_update[(context_key, tool_name)] = (alpha, beta)

                        sampled_score = np.random.beta(alpha, beta)

                        if sampled_score > highest_sample:
                            highest_sample = sampled_score
                            best_tool = tool_name

                    if best_tool is None:
                        best_tool = candidate_tools[0]

                    trace_id = self._generate_trace_id(context_key, best_tool)
                    await self.storage.log_selection(trace_id, context_key, best_tool)
                    results.append((best_tool, trace_id))

                if priors_to_update:
                    if hasattr(self.storage, "update_tool_params_batch"):
                        await self.storage.update_tool_params_batch(priors_to_update)
                    else:
                        for (ctx_key, tool_name), (alpha, beta) in priors_to_update.items():
                            await self.storage.update_tool_params(ctx_key, tool_name, alpha, beta)

                return results

            else:
                if self.embedder is None:
                    raise ValueError("embedder is required for linear bandit mode")
                
                if hasattr(self.embedder, "aembed_queries"):
                    vectors = await self.embedder.aembed_queries(contexts)
                elif hasattr(self.embedder, "embed_queries"):
                    vectors = self.embedder.embed_queries(contexts)
                elif hasattr(self.embedder, "aembed_query"):
                    vectors = await asyncio.gather(*(self.embedder.aembed_query(ctx) for ctx in contexts))
                else:
                    vectors = [self.embedder.embed_query(ctx) for ctx in contexts]
                
                tool_params = {}
                if hasattr(self.storage, "aget_linear_params_batch"):
                    tool_params = await self.storage.aget_linear_params_batch(candidate_tools)
                else:
                    for tool_name in candidate_tools:
                        tool_params[tool_name] = await self.storage.aget_linear_params(tool_name)

                context_keys = await self._resolve_context_keys(contexts)
                results = []
                for idx, x_seq in enumerate(vectors):
                    x = np.array(x_seq, dtype=np.float32)
                    d = len(x)
                    x_augmented = np.append(x, 1.0)
                    d_aug = d + 1
                    context_key = context_keys[idx]
                    context_text = contexts[idx]
                    
                    best_tool = None
                    highest_score = -float("inf")
                    
                    for tool_name in candidate_tools:
                        prior_alpha, prior_beta = await self.get_prior(context_text, tool_name)
                        prior_p = prior_alpha / (prior_alpha + prior_beta)
                        
                        precision, reward_vector = tool_params.get(tool_name, (None, None))
                        if precision is None or reward_vector is None:
                            precision = self.lambda_val * np.ones(d_aug, dtype=np.float32) if self.diagonal_covariance else self.lambda_val * np.eye(d_aug, dtype=np.float32)
                            reward_vector = np.zeros(d_aug, dtype=np.float32)
                            reward_vector[-1] = self.lambda_val * prior_p

                        if self.diagonal_covariance:
                            theta_hat = reward_vector / precision
                        else:
                            theta_hat = np.linalg.solve(precision, reward_vector)

                        if self.mode == "lints":
                            if self.diagonal_covariance:
                                std_devs = self.exploration_weight / np.sqrt(precision)
                                theta_sample = np.random.normal(theta_hat, std_devs)
                            else:
                                cov = np.linalg.inv(precision)
                                cov = 0.5 * (cov + cov.T)
                                try:
                                    L = np.linalg.cholesky(cov)
                                    z = np.random.normal(size=d_aug)
                                    theta_sample = theta_hat + self.exploration_weight * np.dot(L, z)
                                except np.linalg.LinAlgError:
                                    theta_sample = np.random.multivariate_normal(
                                        theta_hat, (self.exploration_weight ** 2) * cov
                                    )
                            score = float(np.dot(x_augmented, theta_sample))
                        else:
                            expected_reward = float(np.dot(x_augmented, theta_hat))
                            if self.diagonal_covariance:
                                uncertainty = np.sqrt(np.sum((x_augmented ** 2) / precision))
                            else:
                                cov = np.linalg.inv(precision)
                                uncertainty = np.sqrt(np.dot(x_augmented, np.dot(cov, x_augmented)))
                            score = expected_reward + self.exploration_weight * uncertainty

                        if score > highest_score:
                            highest_score = score
                            best_tool = tool_name

                    if best_tool is None:
                        best_tool = candidate_tools[0]

                    trace_id = self._generate_trace_id(context_key, best_tool)
                    await self.storage.log_selection(trace_id, context_key, best_tool)
                    results.append((best_tool, trace_id))

                return results

        except Exception as e:
            logger.exception("AsyncBayesianToolRouter batch routing failed. Triggering fail-safe fallback.")
            await self._call_telemetry(
                "route_batch_failure",
                e,
                {
                    "contexts": contexts,
                    "candidate_tools": candidate_tools,
                },
            )

            fallback_choice = self.fallback_tool if (self.fallback_tool and self.fallback_tool in candidate_tools) else candidate_tools[0]
            fallback_trace_id = self._generate_trace_id("fallback_ctx", fallback_choice)
            for _ in contexts:
                await self.storage.log_selection(fallback_trace_id, "fallback_ctx", fallback_choice)
            return [(fallback_choice, fallback_trace_id)] * len(contexts)

    async def afeedback_batch(self, feedbacks: List[Dict[str, Any]]) -> None:
        if not feedbacks:
            return

        await self._ensure_initialized()

        try:
            contexts_to_embed = []
            contexts_to_embed_indices = []
            prepared_feedbacks = []
            
            for fb in feedbacks:
                success = fb.get("success")
                reward = fb.get("reward")
                
                if success is None and reward is None:
                    raise ValueError("Either 'success' or 'reward' must be provided in feedback.")
                if success is not None and reward is not None:
                    expected_reward = 1.0 if success else 0.0
                    if reward != expected_reward:
                        raise ValueError(
                            f"Conflicting feedback: success={success} and reward={reward}."
                        )
                
                reward_val = float(reward) if reward is not None else (1.0 if success else 0.0)
                
                trace_id = fb.get("trace_id")
                if trace_id is not None:
                    context_key, tool_name = self._decode_trace_id(trace_id)
                    prepared_feedbacks.append({
                        "type": "trace",
                        "context_key": context_key,
                        "tool_name": tool_name,
                        "reward_val": reward_val,
                    })
                else:
                    context_text = fb.get("context_text")
                    tool_name = fb.get("tool_name")
                    if not context_text or not tool_name:
                        raise ValueError("Feedback must contain either 'trace_id' or both 'context_text' and 'tool_name'.")
                    
                    prepared_feedbacks.append({
                        "type": "text",
                        "context_text": context_text,
                        "tool_name": tool_name,
                        "reward_val": reward_val,
                    })
                    contexts_to_embed.append(context_text)
                    contexts_to_embed_indices.append(len(prepared_feedbacks) - 1)

            if contexts_to_embed:
                resolved_keys = await self._resolve_context_keys(contexts_to_embed)
                for idx, key in zip(contexts_to_embed_indices, resolved_keys):
                    prepared_feedbacks[idx]["context_key"] = key
                    
                if self.mode != "clustering":
                    if hasattr(self.embedder, "aembed_queries"):
                        vectors = await self.embedder.aembed_queries(contexts_to_embed)
                    elif hasattr(self.embedder, "embed_queries"):
                        vectors = self.embedder.embed_queries(contexts_to_embed)
                    elif hasattr(self.embedder, "aembed_query"):
                        vectors = await asyncio.gather(*(self.embedder.aembed_query(t) for t in contexts_to_embed))
                    else:
                        vectors = [self.embedder.embed_query(t) for t in contexts_to_embed]
                    for idx, vector in zip(contexts_to_embed_indices, vectors):
                        prepared_feedbacks[idx]["vector"] = vector

            if self.mode == "clustering":
                updates = []
                for fb in prepared_feedbacks:
                    updates.append((fb["context_key"], fb["tool_name"], self.decay_factor, fb["reward_val"]))
                await self.storage.decay_and_update_batch(updates)
            else:
                updates = []
                for fb in prepared_feedbacks:
                    tool_name = fb["tool_name"]
                    reward_val = fb["reward_val"]
                    
                    if fb["type"] == "trace":
                        x_seq = await self._context_store.aget_context_vector(fb["context_key"])
                        if x_seq is None:
                            logger.warning(
                                f"Context vector not found for key {fb['context_key']}. Using zero vector as fallback."
                            )
                            d = 384
                            precision, _ = await self.storage.aget_linear_params(tool_name)
                            if precision is not None:
                                d = len(precision) - 1
                            x = np.zeros(d, dtype=np.float32)
                        else:
                            x = np.array(x_seq, dtype=np.float32)
                    else:
                        x = np.array(fb["vector"], dtype=np.float32)
                    
                    x_augmented = np.append(x, 1.0)
                    
                    if tool_name in self.priors:
                        alpha, beta = self.priors[tool_name]
                        prior_p = alpha / (alpha + beta)
                    else:
                        prior_p = 0.5
                        
                    updates.append((tool_name, self.decay_factor, reward_val, x_augmented, self.lambda_val, prior_p, self.diagonal_covariance))
                    
                await self.storage.adecay_and_update_linear_batch(updates)

            # Log trace feedback
            for fb in feedbacks:
                trace_id = fb.get("trace_id")
                if trace_id is not None:
                    reward_val = float(fb.get("reward")) if fb.get("reward") is not None else (1.0 if fb.get("success") else 0.0)
                    await self.storage.log_feedback(trace_id, reward_val)

        except Exception as e:
            logger.exception("AsyncBayesianToolRouter batch feedback submission failed.")
            await self._call_telemetry("feedback_batch_failure", e, {"feedbacks": feedbacks})


