from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.utils.config_loader import load_config
from src.utils.model_loader import ModelLoader
from src.logger import GLOBAL_LOGGER
logger= GLOBAL_LOGGER.bind(module="AutonomousReportGenerator")


class VectorStoreManager:
    """
    Manages vector store initialization and persistence.

    Responsibilities:
      - Initialize Chroma vector store
      - Add documents to collection
      - Persist vector store to disk
      - Provide retriever access point
    """

    def __init__(self):
        self.config = load_config()
        self.model_loader = ModelLoader()

        chroma_cfg = self.config.get("chroma", {})
        self.collection_name = chroma_cfg.get(
            "collection_name", "research_report"
        )
        self.persist_directory = chroma_cfg.get(
            "persist_directory", "./chroma_db"
        )

        self._vector_store: Optional[Chroma] = None

    def get_vector_store(self) -> Chroma:
        """
        Lazily initialize and return the Chroma vector store.

        Returns:
            Chroma
        """
        if self._vector_store is None:
            logger.info(
                f"Initializing Chroma collection '{self.collection_name}'"
            )

            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.model_loader.load_embeddings(),
                persist_directory=self.persist_directory,
            )

        return self._vector_store

    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store and persist.

        Args:
            documents (List[Document])
        """
        if not documents:
            logger.warning("No documents provided to vector store.")
            return

        vector_store = self.get_vector_store()
        vector_store.add_documents(documents)
        vector_store.persist()

        logger.info(
            f"Added {len(documents)} documents to Chroma collection "
            f"'{self.collection_name}'"
        )

    def reset_collection(self):
        """
        Delete and recreate the vector store collection.
        USE WITH CAUTION.
        """
        logger.warning(
            f"Resetting Chroma collection '{self.collection_name}'"
        )

        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.model_loader.load_embeddings(),
            persist_directory=self.persist_directory,
            client_settings={"allow_reset": True},
        )
        self._vector_store.delete_collection()
        self._vector_store.persist()

        logger.info("Vector store collection reset completed.")
