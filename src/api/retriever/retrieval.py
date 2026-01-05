import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter

from src.utils.config_loader import load_config
from src.utils.model_loader import ModelLoader


class Retriever:
    """
    Retriever pipeline for querying ChromaDB vector store with contextual compression.

    This class:
      - Loads environment variables (if needed).
      - Initializes embeddings and Chroma vector store.
      - Creates an MMR-based retriever.
      - Applies an LLM-based compression filter to refine retrieved context.
    """

    def __init__(self):
        """Initialize retriever with config and environment setup."""
        print("🔍 Initializing Retriever Pipeline (ChromaDB)...")
        self.model_loader = ModelLoader()
        self.config = load_config()
        self._load_env_variables()

        self.vstore = None
        self.retriever_instance = None

    def load_retriever(self):
        """
        Initialize the vector store retriever with MMR search and LLM-based compression.

        Steps:
          1. Connect to Chroma vector store.
          2. Create an MMR retriever for diverse search results.
          3. Load an LLM-based compressor to refine retrieved chunks.
        """

        # Step 1: Initialize Chroma vector store
        if not self.vstore:
            chroma_config = self.config.get("chroma", {})

            collection_name = chroma_config.get(
                "collection_name", "research_report"
            )
            persist_directory = chroma_config.get(
                "persist_directory", "./chroma_db"
            )

            self.vstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.model_loader.load_embeddings(),
                persist_directory=persist_directory,
            )

            print(
                f"Connected to ChromaDB collection '{collection_name}' "
                f"at '{persist_directory}'"
            )

        # Step 2: Build MMR retriever
        if not self.retriever_instance:
            retriever_config = self.config.get("retriever", {})
            top_k = retriever_config.get("top_k", 3)

            mmr_retriever = self.vstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,
                    "fetch_k": 20,
                    "lambda_mult": 0.7,
                },
            )
            print("MMR retriever initialized successfully.")

            # Step 3: Contextual compression
            llm = self.model_loader.load_llm()
            compressor = LLMChainFilter.from_llm(llm)

            self.retriever_instance = ContextualCompressionRetriever(
                base_retriever=mmr_retriever,
                base_compressor=compressor,
            )
            print("LLM-based compression retriever configured successfully.")

        return self.retriever_instance

    def call_retriever(self, query: str):
        """
        Execute the retriever pipeline for a given query.

        Args:
            query (str): User query text.

        Returns:
            List[Document]: A list of contextually relevant LangChain Document objects.
        """
        retriever = self.load_retriever()
        print(f"Querying retriever for: '{query}'")

        output = retriever.invoke(query)

        print(f"Retrieved {len(output)} relevant documents.")
        return output
