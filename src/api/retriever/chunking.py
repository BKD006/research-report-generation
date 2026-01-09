from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config_loader import load_config
from src.logger import GLOBAL_LOGGER
logger= GLOBAL_LOGGER.bind(module="AutonomousReportGenerator")

class DocumentChunker:
    """
    Handles document chunking logic for RAG pipelines.

    Responsibilities:
      - Split documents into overlapping chunks
      - Preserve and enrich metadata
      - Ensure consistent chunk structure across ingestion
    """

    def __init__(self):
        self.config = load_config()

        chunking_cfg = self.config.get("chunking", {})
        self.chunk_size = chunking_cfg.get("chunk_size", 800)
        self.chunk_overlap = chunking_cfg.get("chunk_overlap", 120)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        """
        Split documents into chunks while preserving metadata.

        Args:
            documents (List[Document]): Raw LangChain documents

        Returns:
            List[Document]: Chunked documents
        """
        if not documents:
            logger.warning("No documents provided for chunking.")
            return []

        all_chunks: List[Document] = []

        for doc in documents:
            split_docs = self.text_splitter.split_documents([doc])

            total_chunks = len(split_docs)
            for idx, chunk in enumerate(split_docs):
                chunk.metadata.update(
                    {
                        "chunk_id": idx,
                        "total_chunks": total_chunks,
                    }
                )
                all_chunks.append(chunk)

        logger.info(
            f"Chunked {len(documents)} documents into "
            f"{len(all_chunks)} chunks."
        )
        return all_chunks
