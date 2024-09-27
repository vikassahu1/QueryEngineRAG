from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model
import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(api_key, document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.
    
    Args:
        api_key (str): The API key for Gemini.
        document (list): The documents to be indexed.
    
    Returns:
        - query_engine: A query engine based on the vector store index.
    """
    try:
        logging.info("Initializing Gemini Embedding model")
        
        # Configure Settings object
        Settings.embed_model = GeminiEmbedding(api_key=api_key)
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20
        Settings.llm = Gemini(api_key=api_key)
        
        logging.info("Creating VectorStoreIndex")
        
        index = VectorStoreIndex.from_documents(document)
        
        # Persist the index
        storage_context = index.storage_context
        storage_context.persist("./storage")
        
        logging.info("Creating query engine")
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise customexception(e, sys)