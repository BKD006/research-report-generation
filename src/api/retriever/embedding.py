from typing import List
from langchain_core.documents import Document

from src.utils.model_loader import ModelLoader
from src.logger import GLOBAL_LOGGER

logger = GLOBAL_LOGGER.bind(module="EmbeddingService")


class EmbeddingService:
    """
    Provides embedding functions for documents and queries.
    """

    def __init__(self):
        self.embedding_model = ModelLoader().load_embeddings()
        logger.info("Embedding model loaded.")

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        if not query:
            raise ValueError("Query cannot be empty")
        return self.embedding_model.embed_query(query)