import os
import sys
from dotenv import load_dotenv
from src.utils.config_loader import load_config
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_groq import ChatGroq
from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ResearchAnalystException

load_dotenv()
class APIKeyManager:
    def __init__(self):
        self.api_keys={
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION"),
                }
        
        #Load loaded key statuses without exposing secrets
        for key, val in self.api_keys.items():
            if val:
                log.info(f"{key} loaded successfully from environment")
            else:
                log.warning(f"{key} is missing from environment variables")

    def get(self, key:str):
        """
        Retrieve a specific API key.

        Args:
            key (str): Name of the API key.
        
        Returns:
            str | None: API key value if found
        """
        return self.api_keys.get(key)


class ModelLoader:
    """
    Loads embedding models and LLMs dynamically based on YAML configuration and environment settings.
    """
    def __init__(self):
        """Initialize model loader and load configuration"""
        try:
            self.api_key_mgr= APIKeyManager()
            self.config= load_config()
            log.info("YAML config loaded", config_keys= list(self.config.keys()))
        except Exception as e:
            log.error("Error initializing ModelLoader", error=str(e))
            raise ResearchAnalystException("Failed to initialize ModelLoader", sys)

    def load_embeddings(self):
        """
        Load and return a Amazon Titan embedding model.

        Returns:

        """
        try:
            model_name=self.config["embedding_model"]["model_name"]
            log.info("Load embedding model", model=model_name)
            embeddings= BedrockEmbeddings(model_id=model_name)
            log.info("Embedding model loaded successfully", model=model_name)
            return embeddings
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise ResearchAnalystException("Failed to load embedding model", sys)
        
    def load_llm(self):
        """
        Load and return a chat-based LLM according to the configured provider.
        """
        try:
            llm_block = self.config["llm"]
            provider_key = os.getenv("LLM_PROVIDER", "groq")

            if provider_key not in llm_block:
                log.error("LLM provider not found in configuration", provider=provider_key)
                raise ValueError(f"LLM provider '{provider_key}' not found in configuration")

            llm_config = llm_block[provider_key]
            provider = llm_config.get("provider")
            model_name = llm_config.get("model_name")
            temperature = llm_config.get("temperature", 0.2)
            max_tokens = llm_config.get("max_output_tokens", 2048)

            log.info("Loading LLM", provider=provider, model=model_name)

            if provider == "groq":
                llm = ChatGroq(
                    model=model_name,
                    api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                    temperature=temperature,
                )
            else:
                log.error("Unsupported LLM provider encountered", provider=provider)
                raise ValueError(f"Unsupported LLM provider: {provider}")

            log.info("LLM loaded successfully", provider=provider, model=model_name)
            return llm

        except Exception as e:
            log.error("Error loading LLM", error=str(e))
            raise ResearchAnalystException("Failed to load LLM", sys)

# ----------------------------------------------------------------------
# 🔹 Standalone Testing
# ----------------------------------------------------------------------
# if __name__ == "__main__":
#     try:
#         loader = ModelLoader()

#         # Test embedding model
#         embeddings = loader.load_embeddings()
#         print(f"Embedding Model Loaded: {embeddings}")
#         result = embeddings.embed_query("Hello, how are you?")
#         print(f"Embedding Result: {result[:5]} ...")

#         # Test LLM
#         llm = loader.load_llm()
#         print(f"LLM Loaded: {llm}")
#         result = llm.invoke("Hello, how are you?")
#         print(f"LLM Result: {result.content[:200]}")

#         log.info("ModelLoader test completed successfully")

#     except ResearchAnalystException as e:
#         log.error("Critical failure in ModelLoader test", error=str(e))