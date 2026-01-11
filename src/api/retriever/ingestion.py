from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


class DocumentIngestor:
    """
    Loads documents from file paths and returns LangChain Documents
    ready for chunking.

    Supported formats:
      - PDF (.pdf)  [PyPDFLoader fallback to PyMuPDFLoader]
      - Text (.txt, .md)
      - Word (.docx)
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    # ==================================================
    # Public API
    # ==================================================
    def ingest_file(self, file_path: Union[str, Path]) -> List[Document]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        docs = self._load_file(path, ext)

        # Attach metadata
        for doc in docs:
            doc.metadata.update(
                {
                    "source": str(path),
                    "file_name": path.name,
                    "file_type": ext,
                }
            )

        print(f"✅ Loaded {len(docs)} Document objects from {path.name}")
        return docs

    def ingest_files(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        all_docs: List[Document] = []

        for fp in file_paths:
            try:
                docs = self.ingest_file(fp)
                all_docs.extend(docs)
            except Exception as e:
                print(f"⚠️ Skipping {fp}: {e}")

        print(f"\n✅ Total loaded documents: {len(all_docs)}")
        return all_docs

    # ==================================================
    # Internal loading
    # ==================================================
    def _load_file(self, path: Path, ext: str) -> List[Document]:
        if ext == ".pdf":
            return self._load_pdf_with_fallback(path)

        if ext in {".txt", ".md"}:
            return TextLoader(str(path), encoding=self.encoding).load()

        if ext == ".docx":
            return Docx2txtLoader(str(path)).load()

        raise ValueError(f"Unsupported extension: {ext}")

    def _load_pdf_with_fallback(self, path: Path) -> List[Document]:
        """
        Try PyPDFLoader first; fallback to PyMuPDFLoader if:
        - PyPDFLoader throws
        - OR returns 0 docs
        """
        # Try PyPDFLoader
        try:
            docs = PyPDFLoader(str(path)).load()
            if docs:
                return docs
            print(f"⚠️ PyPDFLoader returned 0 docs for {path.name}. Falling back to PyMuPDFLoader...")
        except Exception as e:
            print(f"⚠️ PyPDFLoader failed for {path.name}: {e}. Falling back to PyMuPDFLoader...")

        # Fallback: PyMuPDFLoader
        try:
            docs = PyMuPDFLoader(str(path)).load()
            return docs
        except Exception as e:
            raise RuntimeError(f"Both PyPDFLoader and PyMuPDFLoader failed for {path.name}: {e}")
