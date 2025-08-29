# api/loader.py
import os
import logging
import tarfile
from functools import lru_cache

import chromadb
from google.cloud import storage
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# These would ideally be set via environment variables in Cloud Run
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "your-knowledge-packs-bucket")
LOCAL_CACHE_DIR = "/data" # A writable directory inside the container

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
    Ensures a knowledge pack is available locally, downloading from GCS if needed.
    Returns an initialized ChromaDB collection object.
    """
    if target in chroma_clients:
        return chroma_clients[target]

    target_dir = os.path.join(LOCAL_CACHE_DIR, target)
    db_path = os.path.join(target_dir, "db")

    if not os.path.exists(db_path):
        logging.info(f"Knowledge pack for '{target}' not found in local cache. Downloading from GCS...")
        os.makedirs(target_dir, exist_ok=True)
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        archive_name = f"{target}-knowledge-pack.tar.gz"
        blob = bucket.blob(archive_name)
        
        if not blob.exists():
            raise FileNotFoundError(f"Archive {archive_name} not found in GCS bucket {GCS_BUCKET_NAME}.")

        local_archive_path = os.path.join(target_dir, archive_name)
        blob.download_to_filename(local_archive_path)
        
        logging.info(f"Extracting archive {local_archive_path}...")
        with tarfile.open(local_archive_path, "r:gz") as tar:
            # Extract to the parent directory of where the tar is, e.g., /data/
            tar.extractall(path=target_dir)
        
        # The tarball contains a dir like 'langchain-knowledge-pack', we need to move its contents up
        extracted_folder = os.path.join(target_dir, f"{target}-knowledge-pack")
        for item in os.listdir(extracted_folder):
             os.rename(os.path.join(extracted_folder, item), os.path.join(target_dir, item))
        os.rmdir(extracted_folder)
        os.remove(local_archive_path)

    # Now that the files are guaranteed to be local, load ChromaDB
    logging.info(f"Loading ChromaDB collection for '{target}' from {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=target)
    
    # Cache the client for future requests
    chroma_clients[target] = collection
    return collection