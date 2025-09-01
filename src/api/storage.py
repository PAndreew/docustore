# api/storage.py
import os
import logging
from abc import ABC, abstractmethod

class BaseStorageProvider(ABC):
    """Abstract base class for all storage providers."""
    @abstractmethod
    def download_pack(self, target_name: str, destination_dir: str) -> bool:
        """
        Downloads and extracts a knowledge pack for a given target.

        Args:
            target_name: The name of the target (e.g., 'langchain').
            destination_dir: The local directory to save and extract the pack into.

        Returns:
            True if successful, False otherwise.
        """
        pass

class LocalStorageProvider(BaseStorageProvider):
    """'Downloads' a pack from another local directory (e.g., a mounted volume)."""
    def __init__(self):
        self.packs_base_path = os.environ.get("LOCAL_PACKS_PATH", "./repository")
        logging.info(f"LocalStorageProvider initialized. Looking for packs in: {self.packs_base_path}")

    def download_pack(self, target_name: str, destination_dir: str) -> bool:
        import shutil
        source_path = os.path.join(self.packs_base_path, f"{target_name}-knowledge-pack.tar.gz")
        if not os.path.exists(source_path):
            logging.error(f"Local pack not found at {source_path}")
            return False
        
        # In a local setup, we can just extract directly.
        logging.info(f"Extracting local pack from {source_path} to {destination_dir}")
        with tarfile.open(source_path, "r:gz") as tar:
            tar.extractall(path=destination_dir)
        return True

class GCSStorageProvider(BaseStorageProvider):
    """Downloads packs from a Google Cloud Storage bucket."""
    def __init__(self):
        try:
            from google.cloud import storage
            self.storage_client = storage.Client()
            self.bucket_name = os.environ.get("GCS_BUCKET_NAME")
            if not self.bucket_name:
                raise ValueError("GCS_BUCKET_NAME environment variable is not set.")
            self.bucket = self.storage_client.bucket(self.bucket_name)
            logging.info(f"GCSStorageProvider initialized for bucket: {self.bucket_name}")
        except ImportError:
            raise ImportError("Please install 'google-cloud-storage' (`uv pip install \"knowledge-pack-api[gcp]\"`)")

    def download_pack(self, target_name: str, destination_dir: str) -> bool:
        archive_name = f"{target_name}-knowledge-pack.tar.gz"
        blob = self.bucket.blob(archive_name)
        if not blob.exists():
            logging.error(f"Archive {archive_name} not found in GCS bucket {self.bucket_name}.")
            return False
        
        local_archive_path = os.path.join(destination_dir, archive_name)
        blob.download_to_filename(local_archive_path)
        # The extraction logic will be handled by the loader
        return True

# ... You could add S3StorageProvider and HTTPStorageProvider here similarly ...

def get_storage_provider() -> BaseStorageProvider:
    """Factory function to get the configured storage provider."""
    provider_name = os.environ.get("STORAGE_PROVIDER", "local").lower()
    if provider_name == "local":
        return LocalStorageProvider()
    elif provider_name == "gcp":
        return GCSStorageProvider()
    # elif provider_name == "s3":
    #     return S3StorageProvider()
    else:
        raise ValueError(f"Unsupported STORAGE_PROVIDER: {provider_name}")