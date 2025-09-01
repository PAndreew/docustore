# api/loader.py
import os
import logging
import tarfile
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer
from .storage import get_storage_provider  # Import the factory

# --- Configuration ---
LOCAL_CACHE_DIR = "/data" 

# --- In-Memory Caching for Models and Clients ---
# This dictionary will hold initialized ChromaDB clients to avoid reloading.
chroma_clients = {}

@lru_cache(maxsize=1) # Simple singleton pattern to load model only once
def get_embedding_model():
    """Loads and returns the sentence-transformer model, caching it in memory."""
    logging.info("Loading embedding model 'all-MiniLM-L6-v2' into memory...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def load_knowledge_pack(target: str):
    """
    Ensures a knowledge pack is available locally, using the configured storage
    provider to download it if needed. Returns a ChromaDB collection.
    """
    if target in chroma_clients:
        return chroma_clients[target]

    target_dir = os.path.join(LOCAL_CACHE_DIR, target)
    db_path = os.path.join(target_dir, "db")

    if not os.path.exists(db_path):
        logging.info(f"Knowledge pack for '{target}' not found in local cache. Using storage provider.")
        os.makedirs(target_dir, exist_ok=True)
        
        # Use the factory to get the right provider
        storage_provider = get_storage_provider()
        
        # The provider is only responsible for downloading the archive
        archive_name = f"{target}-knowledge-pack.tar.gz"
        local_archive_path = os.path.join(target_dir, archive_name)
        
        # This now works for local, GCS, S3, etc.
        if not storage_provider.download_pack(target, local_archive_path):
            raise FileNotFoundError(f"Failed to download knowledge pack for '{target}'.")

        logging.info(f"Extracting archive {local_archive_path}...")
        with tarfile.open(local_archive_path, "r:gz") as tar:
            tar.extractall(path=target_dir)

        # Clean up the downloaded archive
        os.remove(local_archive_path)

        # Standardize the directory structure after extraction
        extracted_folder_name = f"{target}-knowledge-pack"
        extracted_folder_path = os.path.join(target_dir, extracted_folder_name)
        if os.path.isdir(extracted_folder_path):
            for item in os.listdir(extracted_folder_path):
                os.rename(os.path.join(extracted_folder_path, item), os.path.join(target_dir, item))
            os.rmdir(extracted_folder_path)

    # Now that files are local, the rest of the logic is the same
    logging.info(f"Loading ChromaDB collection for '{target}' from {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=target)
    
    chroma_clients[target] = collection
    return collection