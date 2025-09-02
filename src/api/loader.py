# api/loader.py
import os
import logging
import tarfile
import shutil
import time
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer
from .storage import get_storage_provider
from .config import get_embedding_model_name

# --- Configuration ---
# The root directory inside the container for storing extracted knowledge packs.
LOCAL_CACHE_DIR = "/data"
# The cache directory for the Hugging Face models, read from the environment.
HF_CACHE_DIR = os.environ.get("SENTENCE_TRANSFORMERS_HOME")

# --- In-Memory Caching for Models and Clients ---
chroma_clients = {}

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Loads and returns the sentence-transformer model specified in the config,
    explicitly using the pre-cached directory.
    """
    model_name = get_embedding_model_name()
    logging.info(f"Loading embedding model '{model_name}' from cache path '{HF_CACHE_DIR}'...")
    
    # IMPROVEMENT: Be explicit about the cache folder to ensure it uses the
    # files baked into the Docker image. This removes ambiguity.
    model = SentenceTransformer(model_name, cache_folder=HF_CACHE_DIR)
    return model

def load_knowledge_pack(target: str):
    """
    Ensures a knowledge pack is available locally, using the configured storage
    provider to download and extract it if needed. This function is concurrency-safe.
    Returns a ChromaDB collection.
    """
    if target in chroma_clients:
        return chroma_clients[target]

    target_dir = os.path.join(LOCAL_CACHE_DIR, target)
    db_path = os.path.join(target_dir, "db")
    lock_path = os.path.join(target_dir, ".lock")

    if not os.path.exists(db_path):
        # IMPROVEMENT: Simple file-based locking to prevent race conditions on cold starts.
        if os.path.exists(lock_path):
            logging.info(f"Download for '{target}' is already in progress. Waiting...")
            # Wait for up to 30 seconds for the lock to be released.
            for _ in range(30):
                if not os.path.exists(lock_path):
                    break
                time.sleep(1)
            else:
                raise TimeoutError(f"Timed out waiting for lock on '{target}' to be released.")
        
        try:
            # Create the lock
            os.makedirs(target_dir, exist_ok=True)
            with open(lock_path, 'w') as f:
                f.write('locked')

            logging.info(f"Knowledge pack for '{target}' not found in local cache. Using storage provider.")
            
            storage_provider = get_storage_provider()
            archive_name = f"{target}-knowledge-pack.tar.gz"
            local_archive_path = os.path.join(target_dir, archive_name)
            
            if not storage_provider.download_pack(target, local_archive_path):
                raise FileNotFoundError(f"Failed to download knowledge pack for '{target}'.")

            logging.info(f"Extracting archive {local_archive_path}...")
            with tarfile.open(local_archive_path, "r:gz") as tar:
                tar.extractall(path=target_dir)

            os.remove(local_archive_path)

            extracted_folder_path = os.path.join(target_dir, f"{target}-knowledge-pack")
            if os.path.isdir(extracted_folder_path):
                # IMPROVEMENT: Use shutil.copytree and rmtree for a more robust move.
                # This works like 'mv' by copying and then removing the source.
                for item_name in os.listdir(extracted_folder_path):
                    src_path = os.path.join(extracted_folder_path, item_name)
                    dest_path = os.path.join(target_dir, item_name)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_path, dest_path)
                shutil.rmtree(extracted_folder_path)

        finally:
            # Always ensure the lock is removed, even if an error occurs.
            if os.path.exists(lock_path):
                os.remove(lock_path)

    logging.info(f"Loading ChromaDB collection for '{target}' from {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=target)
    
    chroma_clients[target] = collection
    return collection