from bayesian_cortex.embeddings import (
    AnthropicEmbedder,
    AsyncContextEmbedder,
    AsyncSQLiteVectorStore,
    AsyncVectorContextStore,
    AsyncVectorStoreProtocol,
    CohereEmbedder,
    ContextEmbedder,
    GeminiEmbedder,
    LlamaCppEmbedder,
    LocalSentenceTransformerEmbedder,
    OpenAIEmbedder,
    SQLiteVectorStore,
    VectorContextStore,
    VectorStoreProtocol,
)
from bayesian_cortex.exceptions import (
    BayesianCortexError,
    EmbeddingError,
    TamperDetectedError,
)
from bayesian_cortex.mcp_server import create_mcp_server
from bayesian_cortex.rag import (
    aprocess_ui_feedback,
    calculate_faithfulness,
    check_citation,
    evaluate_rag_success,
    process_ui_feedback,
)
from bayesian_cortex.router import (
    AsyncBayesianRouter,
    BayesianRouter,
)
from bayesian_cortex.storage import (
    AsyncBaseStorage,
    AsyncInMemoryStorage,
    AsyncRedisStorage,
    AsyncSQLiteStorage,
    BaseStorage,
    InMemoryStorage,
    RedisStorage,
    SQLiteStorage,
)

try:
    import importlib.metadata as _metadata
    __version__ = _metadata.version("bayesian_cortex")
except Exception:
    __version__ = "0.1.3"

__all__ = [
    "__version__",
    "BayesianRouter",
    "AsyncBayesianRouter",
    "BaseStorage",
    "AsyncBaseStorage",
    "InMemoryStorage",
    "AsyncInMemoryStorage",
    "SQLiteStorage",
    "AsyncSQLiteStorage",
    "RedisStorage",
    "AsyncRedisStorage",
    "ContextEmbedder",
    "AsyncContextEmbedder",
    "GeminiEmbedder",
    "AnthropicEmbedder",
    "CohereEmbedder",
    "LlamaCppEmbedder",
    "LocalSentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "SQLiteVectorStore",
    "AsyncSQLiteVectorStore",
    "VectorContextStore",
    "AsyncVectorContextStore",
    "VectorStoreProtocol",
    "AsyncVectorStoreProtocol",
    "create_mcp_server",
    "check_citation",
    "calculate_faithfulness",
    "evaluate_rag_success",
    "process_ui_feedback",
    "aprocess_ui_feedback",
    "BayesianCortexError",
    "TamperDetectedError",
    "EmbeddingError",
]
