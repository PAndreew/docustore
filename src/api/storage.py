# api/storage.py
import os
import logging
from abc import ABC, abstractmethod
import shutil


class BaseStorageProvider(ABC):
    """Abstract base class for all storage providers."""
    @abstractmethod
    def download_pack(self, target_name: str, destination_archive_path: str) -> bool:
        """
        Downloads a knowledge pack archive for a given target to a specific file path.

        Args:
            target_name: The name of the target (e.g., 'langchain').
            destination_archive_path: The full local file path to save the .tar.gz file.

        Returns:
            True if successful, False otherwise.
        """
        pass

class LocalStorageProvider(BaseStorageProvider):
    """'Downloads' a pack by copying it from another local directory."""
    def __init__(self):
        self.packs_base_path = os.environ.get("LOCAL_PACKS_PATH", "./repository")
        logging.info(f"LocalStorageProvider initialized. Looking for packs in: {self.packs_base_path}")

    # CORRECTED: The argument name now matches the base class.
    def download_pack(self, target_name: str, destination_archive_path: str) -> bool:
        source_path = os.path.join(self.packs_base_path, f"{target_name}-knowledge-pack.tar.gz")
        if not os.path.exists(source_path):
            logging.error(f"Local pack not found at {source_path}")
            return False
        
        try:
            shutil.copyfile(source_path, destination_archive_path)
            logging.info(f"Copied local pack from {source_path} to {destination_archive_path}")
            return True
        except IOError as e:
            logging.error(f"Failed to copy local pack: {e}")
            return False

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

    # CORRECTED: The argument name now matches the base class.
    def download_pack(self, target_name: str, destination_archive_path: str) -> bool:
        archive_name = f"{target_name}-knowledge-pack.tar.gz"
        blob = self.bucket.blob(archive_name)
        if not blob.exists():
            logging.error(f"Archive {archive_name} not found in GCS bucket {self.bucket_name}.")
            return False
        
        try:
            blob.download_to_filename(destination_archive_path)
            logging.info(f"Downloaded GCS pack to {destination_archive_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to download GCS pack: {e}")
            return False

def get_storage_provider() -> BaseStorageProvider:
    """Factory function to get the configured storage provider."""
    provider_name = os.environ.get("STORAGE_PROVIDER", "local").lower()
    if provider_name == "local":
        return LocalStorageProvider()
    elif provider_name == "gcp":
        return GCSStorageProvider()
    else:
        raise ValueError(f"Unsupported STORAGE_PROVIDER: {provider_name}")