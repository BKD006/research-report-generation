from typing import List, Optional

from langchain_core.documents import Document

from src.api.retriever.vectorstore import VectorStoreManager
from src.utils.config_loader import load_config
from src.logger import GLOBAL_LOGGER

logger = GLOBAL_LOGGER.bind(module="AutonomousReportGenerator")


class SimpleRAGRetriever:
    """
    Simple retrieval with just vector similarity search.
    No compression, no reranking - just straightforward retrieval.
    """

    def __init__(self):
        self.config = load_config()
        self.vector_store_manager = VectorStoreManager()
        self._retriever = None

    def get_retriever(self):
        """Build a basic similarity retriever"""
        if self._retriever is not None:
            return self._retriever

        retriever_cfg = self.config.get("retriever", {})
        vector_store = self.vector_store_manager.get_vector_store()

        self._retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": retriever_cfg.get("top_k", 5),
            },
        )

        logger.info("Simple similarity retriever initialized.")
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents for a query"""
        if not query:
            raise ValueError("Query cannot be empty.")

        retriever = self.get_retriever()
        documents = retriever.invoke(query)

        logger.info(f"Retrieved {len(documents)} documents.")
        return documents


# ===================================
# SLIGHTLY ENHANCED VERSION (Optional)
# ===================================

class SimpleRAGRetrieverWithMMR:
    """
    Simple retrieval with MMR for diversity.
    Still simple, but reduces duplicate results.
    """

    def __init__(self):
        self.config = load_config()
        self.vector_store_manager = VectorStoreManager()
        self._retriever = None

    def get_retriever(self):
        """Build MMR retriever for diverse results"""
        if self._retriever is not None:
            return self._retriever

        retriever_cfg = self.config.get("retriever", {})
        vector_store = self.vector_store_manager.get_vector_store()

        self._retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": retriever_cfg.get("top_k", 5),
                "fetch_k": retriever_cfg.get("fetch_k", 20),
                "lambda_mult": retriever_cfg.get("lambda_mult", 0.5),
            },
        )

        logger.info("MMR retriever initialized.")
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents for a query"""
        if not query:
            raise ValueError("Query cannot be empty.")

        retriever = self.get_retriever()
        documents = retriever.invoke(query)

        logger.info(f"Retrieved {len(documents)} documents.")
        return documents


# ===================================
# WITH SCORE THRESHOLD (Optional)
# ===================================

class SimpleRAGRetrieverWithThreshold:
    """
    Simple retrieval with quality threshold.
    Only returns highly relevant documents.
    """

    def __init__(self, score_threshold: float = 0.7):
        self.config = load_config()
        self.vector_store_manager = VectorStoreManager()
        self.score_threshold = score_threshold
        self._retriever = None

    def get_retriever(self):
        """Build threshold retriever"""
        if self._retriever is not None:
            return self._retriever

        retriever_cfg = self.config.get("retriever", {})
        vector_store = self.vector_store_manager.get_vector_store()

        self._retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": self.score_threshold,
                "k": retriever_cfg.get("top_k", 5),
            },
        )

        logger.info(
            f"Threshold retriever initialized (threshold={self.score_threshold})."
        )
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents for a query"""
        if not query:
            raise ValueError("Query cannot be empty.")

        retriever = self.get_retriever()
        documents = retriever.invoke(query)

        logger.info(f"Retrieved {len(documents)} documents above threshold.")
        return documents