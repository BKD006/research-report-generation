import os
import hashlib
from typing import List, Optional, Set

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.utils.model_loader import ModelLoader
from src.utils.config_loader import load_config
from src.logger import GLOBAL_LOGGER

logger = GLOBAL_LOGGER.bind(module="VectorStoreManager")


class VectorStoreManager:
    """
    FAISS-backed persistent vector store manager.

    Supports:
    - Load existing FAISS index from disk
    - Create a new index on first ingestion
    - Add documents with auto-embedding
    - Add documents with explicit embeddings (EmbeddingService)
    - Stable chunk ids for multi-file ingestion
    - Optional deduplication (idempotent ingestion)
    """

    def __init__(self):
        config = load_config()
        faiss_cfg = config.get("faiss", {})

        self.persist_directory = faiss_cfg.get("persist_directory", "./faiss_db")
        self.embedding_fn = ModelLoader().load_embeddings()
        self._vectorstore: Optional[FAISS] = None

    # ==================================================
    # LOAD OR CREATE
    # ==================================================

    def get_or_create(self) -> Optional[FAISS]:
        """
        Loads FAISS vectorstore from disk if it exists.
        If not, keeps it as None (created on first add).
        """
        if self._vectorstore is not None:
            return self._vectorstore

        os.makedirs(self.persist_directory, exist_ok=True)

        faiss_index_path = os.path.join(self.persist_directory, "index.faiss")
        if os.path.exists(faiss_index_path):
            logger.info(f"Loading existing FAISS vectorstore from {self.persist_directory}")
            self._vectorstore = FAISS.load_local(
                self.persist_directory,
                embeddings=self.embedding_fn,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.info("FAISS index does not exist yet. Will create on first add.")
            self._vectorstore = None

        return self._vectorstore

    # ==================================================
    # ADD DOCUMENTS (AUTO-EMBED)
    # ==================================================

    def add_documents(
        self,
        documents: List[Document],
        *,
        deduplicate: bool = True,
    ) -> FAISS:
        """
        Add documents to FAISS and embed internally using embedding_fn.

        Args:
            documents: List[Document]
            deduplicate: Skip chunks already present (idempotent ingestion)

        Returns:
            FAISS vectorstore
        """
        if not documents:
            raise ValueError("No documents provided for ingestion.")

        # Ensure we load existing store if present
        self.get_or_create()

        # Assign stable ids
        ids = [self._make_chunk_uid(doc) for doc in documents]

        # Optional deduplication
        if deduplicate and self._vectorstore is not None:
            existing = self._existing_ids()
            keep_idx = [i for i, _id in enumerate(ids) if _id not in existing]

            if not keep_idx:
                logger.info("All chunks already exist. Nothing to add.")
                return self._vectorstore

            documents = [documents[i] for i in keep_idx]
            ids = [ids[i] for i in keep_idx]

            logger.info(f"Deduplication enabled. Adding {len(documents)} new chunks.")

        # Attach chunk_uid metadata
        for doc, chunk_uid in zip(documents, ids):
            doc.metadata["chunk_uid"] = chunk_uid

        # Create vs append
        if self._vectorstore is None:
            logger.info("Creating FAISS index from first batch (auto-embed).")
            self._vectorstore = FAISS.from_documents(
                documents,
                embedding=self.embedding_fn,
                ids=ids,
            )
        else:
            logger.info("Appending documents to existing FAISS index (auto-embed).")
            self._vectorstore.add_documents(
                documents,
                ids=ids,
            )

        self._persist()
        logger.info(f"Added {len(documents)} documents to FAISS.")
        return self._vectorstore

    # ==================================================
    # ADD DOCUMENTS WITH EXPLICIT EMBEDDINGS
    # ==================================================

    def add_documents_with_embeddings(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        *,
        deduplicate: bool = True,
    ) -> FAISS:
        """
        Store documents into FAISS using precomputed embeddings.

        Args:
            documents: List[Document]
            embeddings: List[List[float]]
            deduplicate: Skip chunks already present (idempotent ingestion)

        Returns:
            FAISS vectorstore
        """
        if not documents or not embeddings:
            raise ValueError("Documents and embeddings are required.")

        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings length mismatch.")

        # Ensure we load existing store if present
        self.get_or_create()

        # Assign stable ids
        ids = [self._make_chunk_uid(doc) for doc in documents]

        # Optional deduplication
        if deduplicate and self._vectorstore is not None:
            existing = self._existing_ids()
            keep_idx = [i for i, _id in enumerate(ids) if _id not in existing]

            if not keep_idx:
                logger.info("All chunks already exist. Nothing to add.")
                return self._vectorstore

            documents = [documents[i] for i in keep_idx]
            embeddings = [embeddings[i] for i in keep_idx]
            ids = [ids[i] for i in keep_idx]

            logger.info(f"Deduplication enabled. Adding {len(documents)} new chunks.")

        # Attach chunk_uid metadata
        for doc, chunk_uid in zip(documents, ids):
            doc.metadata["chunk_uid"] = chunk_uid

        # Required LangChain format: List[(text, embedding)]
        text_embeddings = [(doc.page_content, emb) for doc, emb in zip(documents, embeddings)]
        metadatas = [doc.metadata for doc in documents]

        if self._vectorstore is None:
            logger.info("Creating FAISS index from first embedded batch.")
            self._vectorstore = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=self.embedding_fn,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            logger.info("Appending embeddings to existing FAISS index.")
            self._vectorstore.add_embeddings(
                text_embeddings=text_embeddings,
                metadatas=metadatas,
                ids=ids,
            )

        self._persist()
        logger.info(f"Added {len(documents)} embedded chunks to FAISS.")
        return self._vectorstore

    # ==================================================
    # PERSISTENCE
    # ==================================================

    def _persist(self):
        if self._vectorstore is None:
            return
        self._vectorstore.save_local(self.persist_directory)
        logger.info("FAISS vectorstore persisted.")

    # ==================================================
    # HELPERS
    # ==================================================

    def _existing_ids(self) -> Set[str]:
        """
        Return the current ids in docstore (if any).
        """
        try:
            # docstore is usually InMemoryDocstore with `_dict`
            docstore_dict = self._vectorstore.docstore._dict  # type: ignore
            return set(docstore_dict.keys())
        except Exception:
            return set()

    def _make_chunk_uid(self, doc: Document) -> str:
        """
        Build a stable unique id for each chunk.

        Uses:
        - file_name / source
        - page (if exists)
        - chunk_id (if exists)
        - content hash
        """
        file_name = str(doc.metadata.get("file_name", doc.metadata.get("source", "unknown")))
        page = str(doc.metadata.get("page", doc.metadata.get("page_number", "")))
        chunk_id = str(doc.metadata.get("chunk_id", ""))

        content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:12]
        return f"{file_name}::p{page}::c{chunk_id}::{content_hash}"

    # ==================================================
    # UTILITY
    # ==================================================

    def count(self) -> int:
        """
        Number of vectors stored in FAISS.
        """
        if self._vectorstore is None:
            self.get_or_create()
        if self._vectorstore is None:
            return 0
        return self._vectorstore.index.ntotal
