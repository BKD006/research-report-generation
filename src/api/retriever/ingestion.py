import os
from typing import List

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.config_loader import load_config
from src.utils.model_loader import ModelLoader
from src.logger import GLOBAL_LOGGER
logger= GLOBAL_LOGGER.bind(module="AutonomousReportGenerator")


class DocumentIngestor:
    """
    Handles ingestion of internal documents into ChromaDB.

    Responsibilities:
      - Load raw documents from filesystem
      - Chunk documents with metadata preservation
      - Embed and persist into Chroma vector store
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

        splitter_cfg = self.config.get("chunking", {})
        self.chunk_size = splitter_cfg.get("chunk_size", 800)
        self.chunk_overlap = splitter_cfg.get("chunk_overlap", 120)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.model_loader.load_embeddings(),
            persist_directory=self.persist_directory,
        )

    def ingest_directory(self, base_path: str):
        """
        Ingest all supported documents from a directory.

        Expected structure:
            base_path/
              ├── adrs/
              ├── rfcs/
              ├── tickets/
              ├── postmortems/
              └── research_notes/

        Args:
            base_path (str): Root directory of documents
        """
        logger.info(f"Starting ingestion from: {base_path}")

        documents = self._load_documents(base_path)
        chunks = self._chunk_documents(documents)

        if not chunks:
            logger.warning("No documents found to ingest.")
            return

        self.vector_store.add_documents(chunks)
        self.vector_store.persist()

        logger.info(
            f"Ingested {len(chunks)} chunks into Chroma collection "
            f"'{self.collection_name}'"
        )

    def _load_documents(self, base_path: str) -> List[Document]:
        """
        Load documents from filesystem and attach metadata.

        Returns:
            List[Document]
        """
        supported_ext = {".md", ".txt"}
        loaded_docs: List[Document] = []

        for root, _, files in os.walk(base_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in supported_ext:
                    continue

                file_path = os.path.join(root, file)
                doc_type = os.path.basename(root)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    metadata = {
                        "source": file_path,
                        "doc_type": doc_type,
                        "file_name": file,
                    }

                    loaded_docs.append(
                        Document(
                            page_content=content,
                            metadata=metadata,
                        )
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to load document {file_path}: {e}"
                    )

        logger.info(f"Loaded {len(loaded_docs)} raw documents.")
        return loaded_docs

    def _chunk_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        """
        Chunk documents while preserving metadata.

        Args:
            documents (List[Document])

        Returns:
            List[Document]
        """
        chunks: List[Document] = []

        for doc in documents:
            split_docs = self.text_splitter.split_documents([doc])

            for idx, split_doc in enumerate(split_docs):
                split_doc.metadata.update(
                    {
                        "chunk_id": idx,
                        "total_chunks": len(split_docs),
                    }
                )
                chunks.append(split_doc)

        logger.info(f"Generated {len(chunks)} document chunks.")
        return chunks
