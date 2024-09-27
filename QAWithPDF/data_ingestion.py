from llama_index.core import SimpleDirectoryReader
import sys
from exception import customexception
from logger import logging

def load_data(data_dir):
    """
    Load PDF documents from a specified directory.

    Parameters:
    - data_dir (str): The path to the directory containing PDF files.

    Returns:
    - A list of loaded PDF documents. The specific type of documents may vary.
    """
    try:
        logging.info("Data loading started...")
        loader = SimpleDirectoryReader(data_dir)  # Use the provided directory path
        documents = loader.load_data()
        logging.info("Data loading completed...")
        return documents
    except Exception as e:
        logging.info("Exception in loading data...")
        raise customexception(e, sys)
