from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config_loader import load_config
from src.logger import GLOBAL_LOGGER

logger = GLOBAL_LOGGER.bind(module="DocumentChunker")


class DocumentChunker:
    """
    Splits documents into overlapping chunks
    while preserving metadata.
    """

    def __init__(self):
        config = load_config()
        chunk_cfg = config.get("chunking", {})

        self.chunk_size = chunk_cfg.get("chunk_size", 512)
        self.chunk_overlap = chunk_cfg.get("chunk_overlap", 64)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        if not documents:
            logger.warning("No documents provided for chunking.")
            return []

        chunks: List[Document] = []

        for doc in documents:
            split_docs = self.splitter.split_documents([doc])
            total = len(split_docs)

            for idx, chunk in enumerate(split_docs):
                chunk.metadata.update(
                    {
                        "chunk_id": idx,
                        "total_chunks": total,
                    }
                )
                chunks.append(chunk)

        logger.info(
            f"Chunked {len(documents)} documents into {len(chunks)} chunks."
        )
        return chunks
