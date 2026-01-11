from typing import List
from langchain_core.documents import Document

from src.api.retriever.vectorstore import VectorStoreManager
from src.utils.config_loader import load_config
from src.logger import GLOBAL_LOGGER

logger = GLOBAL_LOGGER.bind(module="AutonomousReportGenerator")


# ===================================
# BASIC SIMILARITY RETRIEVER
# ===================================

class SimpleRAGRetriever:
    """
    Simple retrieval with vector similarity search.
    """

    def __init__(self, top_k: int = 5):
        self.config = load_config()
        self.vector_store_manager = VectorStoreManager()
        self._retriever = None
        self.top_k= top_k

    def get_retriever(self):
        if self._retriever:
            return self._retriever

        retriever_cfg = self.config.get("retriever", {})
        vector_store = self.vector_store_manager.get_or_create()

        self._retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retriever_cfg.get("top_k", self.top_k)},
        )

        logger.info("Simple similarity retriever initialized.")
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        if not query:
            raise ValueError("Query cannot be empty.")
        return self.get_retriever().invoke(query)


# ===================================
# MMR RETRIEVER
# ===================================

class SimpleRAGRetrieverWithMMR:
    """
    Retrieval with MMR for diversity.
    """

    def __init__(self, top_k: int = 5):
        self.config = load_config()
        self.vector_store_manager = VectorStoreManager()
        self._retriever = None
        self.top_k= top_k

    def get_retriever(self):
        if self._retriever:
            return self._retriever

        retriever_cfg = self.config.get("retriever", {})
        vector_store = self.vector_store_manager.get_or_create()

        self._retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": retriever_cfg.get("top_k", self.top_k),
                "fetch_k": retriever_cfg.get("fetch_k", 20),
                "lambda_mult": retriever_cfg.get("lambda_mult", 0.5),
            },
        )

        logger.info("MMR retriever initialized.")
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        if not query:
            raise ValueError("Query cannot be empty.")
        return self.get_retriever().invoke(query)


# ===================================
# SCORE THRESHOLD RETRIEVER
# ===================================

class SimpleRAGRetrieverWithThreshold:
    """
    Retrieval with similarity score threshold.
    """

    def __init__(self, top_k: int=5, score_threshold: float = 0.7):
        self.config = load_config()
        self.vector_store_manager = VectorStoreManager()
        self.score_threshold = score_threshold
        self._retriever = None
        self.top_k= top_k

    def get_retriever(self):
        if self._retriever:
            return self._retriever

        retriever_cfg = self.config.get("retriever", {})
        vector_store = self.vector_store_manager.get_or_create()

        self._retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": retriever_cfg.get("top_k", self.top_k),
                "score_threshold": self.score_threshold,
            },
        )

        logger.info(
            f"Threshold retriever initialized (threshold={self.score_threshold})."
        )
        return self._retriever

    def retrieve(self, query: str) -> List[Document]:
        if not query:
            raise ValueError("Query cannot be empty.")
        return self.get_retriever().invoke(query)