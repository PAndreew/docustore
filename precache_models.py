# precache_models.py
import os
import tomli 

def main():
    """
    Downloads and caches the sentence-transformer model defined in the config.
    This script is intended to be run during the Docker build process.
    """
    print("--- Starting model pre-caching ---")

    # Load configuration to find the model name
    try:
        with open("config.toml", "rb") as f:
            config = tomli.load(f)
        
        # Get the model name from the config file
        model_name = config.get("vector_store", {}).get("embedding_model_name")
        
        if not model_name:
            raise ValueError("`embedding_model_name` not found in config.toml under [vector_store]")

    except (FileNotFoundError, tomli.TOMLDecodeError, ValueError) as e:
        print(f"ERROR: Could not load model configuration from config.toml. Reason: {e}")
        # Exit with an error code to fail the Docker build if config is bad
        import sys
        sys.exit(1)

    # We now have a list of one model to cache
    models_to_cache = [model_name]
    
    from sentence_transformers import SentenceTransformer
    CACHE_DIR = os.environ.get("SENTENCE_TRANSFORMERS_HOME", os.path.expanduser('~/.cache/torch/sentence_transformers'))

    for model_name_to_cache in models_to_cache:
        try:
            print(f"Downloading and caching model: {model_name_to_cache}")
            SentenceTransformer(model_name_to_cache, cache_folder=CACHE_DIR)
            print(f"Successfully cached {model_name_to_cache}.")
        except Exception as e:
            print(f"ERROR: Could not cache model {model_name_to_cache}. Reason: {e}")
            import sys
            sys.exit(1)

    print("--- Model pre-caching finished ---")

if __name__ == "__main__":
    main()