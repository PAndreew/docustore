# api/config.py
import tomli
from functools import lru_cache

@lru_cache(maxsize=1)
def get_api_config():
    """Loads and caches the main config.toml file."""
    with open("config.toml", "rb") as f:
        return tomli.load(f)

def get_embedding_model_name():
    """Returns the embedding model name from the config."""
    config = get_api_config()
    return config.get("vector_store", {}).get("embedding_model_name", "all-MiniLM-L6-v2")