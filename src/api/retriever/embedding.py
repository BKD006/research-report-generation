from typing import List

from langchain_core.documents import Document

from src.utils.model_loader import ModelLoader
from src.utils.config_loader import load_config
from src.logger import GLOBAL_LOGGER
logger= GLOBAL_LOGGER.bind(module="AutonomousReportGenerator")


class EmbeddingService:
    """
    Handles embedding logic for RAG pipelines.

    Responsibilities:
      - Convert documents into vector embeddings
      - Provide embedding functions to vector stores
      - Abstract embedding model configuration
    """

    def __init__(self):
        self.config = load_config()
        self.model_loader = ModelLoader()

        self.embedding_model = self.model_loader.load_embeddings()
        logger.info("Embedding model initialized successfully.")

    def embed_documents(
        self, documents: List[Document]
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents (List[Document]): Documents to embed

        Returns:
            List[List[float]]: Embedding vectors
        """
        if not documents:
            logger.warning("No documents provided for embedding.")
            return []

        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)

        logger.info(
            f"Generated embeddings for {len(embeddings)} documents."
        )
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query string.

        Args:
            query (str): Query text

        Returns:
            List[float]: Query embedding
        """
        if not query:
            raise ValueError("Query cannot be empty.")

        embedding = self.embedding_model.embed_query(query)
        logger.info("Generated query embedding successfully.")
        return embedding
