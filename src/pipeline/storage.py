# pipeline/storage.py
import json
import logging
import os
from typing import Any, Dict, List
import networkx as nx

def save_to_cache(data: List[Dict[str, Any]], cache_path: str) -> None:
    """Saves scraped data to a JSON cache file."""
    logging.info("Saving scraped data to cache: %s", cache_path)
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info("Successfully cached %d documents.", len(data))
    except IOError as e:
        logging.error("Failed to save cache file: %s", e)

def load_from_cache(cache_path: str) -> List[Dict[str, Any]]:
    """Loads scraped data from a JSON cache file if it exists."""
    if os.path.exists(cache_path):
        logging.info("Loading scraped data from cache: %s", cache_path)
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logging.error("Failed to load or parse cache file: %s", e)
    return []

def save_graph(graph: nx.DiGraph, output_dir: str, target_name: str) -> None:
    """Saves the knowledge graph to a GraphML file."""
    logging.info("Saving knowledge graph to directory: %s", output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{target_name}_graph.graphml")
        nx.write_graphml(graph, file_path)
        logging.info("Graph successfully saved to %s", file_path)
    except Exception as e:
        logging.error("An unexpected error occurred while saving the graph: %s", e)