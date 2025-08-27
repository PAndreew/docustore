# pipeline/config.py
import logging
from typing import Any, Dict
import tomli

def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Loads the pipeline configuration from a TOML file."""
    logging.info("Loading configuration from %s", config_path)
    try:
        with open(config_path, "rb") as f:
            return tomli.load(f)
    except FileNotFoundError:
        logging.error("Configuration file not found at %s", config_path)
        raise
    except tomli.TOMLDecodeError as e:
        logging.error("Error decoding TOML file: %s", e)
        raise