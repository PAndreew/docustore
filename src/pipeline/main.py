# main.py
import logging
import os
import argparse
from typing import Any, Dict

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)

# Import pipeline modules
from pipeline.config import load_config
from pipeline.scraper import scrape_documentation
from pipeline.storage import save_to_cache, load_from_cache, save_graph
from pipeline.graph_creator import create_knowledge_graph

# --- Stage 1: Scraping and Caching ---
def scrape_and_cache_target(target_name: str, config: Dict[str, Any], force: bool = False):
    """
    Handles the scraping and caching stage for a specific target.
    
    Args:
        target_name: The key of the target in the config file.
        config: The loaded configuration dictionary.
        force: If True, re-scrapes even if a cache file exists.
    """
    logging.info("--- Starting Scrape Stage for Target: %s ---", target_name)
    target_config = config.get("targets", {}).get(target_name)
    if not target_config:
        logging.error("Target '%s' not found in config.toml.", target_name)
        return

    pipeline_config = config.get("pipeline", {})
    cache_dir = pipeline_config.get("cache_dir", ".cache")
    cache_file = os.path.join(cache_dir, f"{target_name}_scrape_data.json")

    if os.path.exists(cache_file) and not force:
        logging.info("Cache file already exists for '%s'. Use --force to re-scrape. Skipping.", target_name)
        return

    page_limit = target_config.get("page_limit", 1000)
    target_url = target_config.get("url")

    scraped_docs = scrape_documentation(url=target_url, limit=page_limit)
    if scraped_docs:
        save_to_cache(scraped_docs, cache_file)
    else:
        logging.warning("Scraping returned no documents for '%s'. Nothing to cache.", target_name)

# --- Stage 2: Knowledge Graph Creation from Cache ---
def process_target_from_cache(target_name: str, config: Dict[str, Any]):
    """
    Handles the graph creation and saving stage by loading data from the cache.
    
    Args:
        target_name: The key of the target in the config file.
        config: The loaded configuration dictionary.
    """
    logging.info("--- Starting Processing Stage for Target: %s ---", target_name)
    target_config = config.get("targets", {}).get(target_name)
    if not target_config:
        logging.error("Target '%s' not found in config.toml.", target_name)
        return

    pipeline_config = config.get("pipeline", {})
    kg_config = config.get("knowledge_graph", {})
    cache_dir = pipeline_config.get("cache_dir", ".cache")
    output_dir = pipeline_config.get("output_dir", "repository")
    cache_file = os.path.join(cache_dir, f"{target_name}_scrape_data.json")

    # Load data exclusively from the cache
    docs_from_cache = load_from_cache(cache_file)
    if not docs_from_cache:
        logging.error("No cache file found for '%s'. Please run the 'scrape' command first.", target_name)
        return

    # Generate Knowledge Graph from the loaded documents
    knowledge_graph = create_knowledge_graph(
        documents=docs_from_cache,
        technical_entities=kg_config.get("technical_entities", []),
        enabled_spacy_entities=kg_config.get("enabled_spacy_entities", []),
        associate_descriptions=kg_config.get("associate_descriptions", False),
    )

    # Save the generated graph
    if knowledge_graph.number_of_nodes() > 0:
        save_graph(knowledge_graph, output_dir=output_dir, target_name=target_name)
    else:
        logging.warning("Knowledge graph for '%s' is empty. Nothing to save.", target_name)

# --- Full Pipeline Orchestrator ---
def run_full_pipeline(target_name: str, config: Dict[str, Any]):
    """
    Runs the entire pipeline sequentially: scrape -> cache -> load -> process.
    """
    logging.info("--- Starting Full Pipeline for Target: %s ---", target_name)
    scrape_and_cache_target(target_name, config, force=False) # force=False to use existing cache
    process_target_from_cache(target_name, config)
    logging.info("--- Full Pipeline Finished for Target: %s ---", target_name)

# --- GCP Entrypoint ---
def gcp_entrypoint(event, context):
    """
    Cloud Function entry point. Runs the full pipeline.
    """
    target = event.get('attributes', {}).get('target')
    if not target:
        logging.error("Pub/Sub message must have a 'target' attribute.")
        return

    logging.info("GCP Function triggered for target: %s", target)
    config = load_config()
    run_full_pipeline(target, config) # The GCP function will always run the full sequence

# --- Local Execution with Sub-commands ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Scrape command ---
    parser_scrape = subparsers.add_parser("scrape", help="Scrape a target website and save the results to cache.")
    parser_scrape.add_argument("target", help="The name of the target to scrape (e.g., 'langchain' or 'all').")
    parser_scrape.add_argument("--force", action="store_true", help="Force re-scraping even if cache exists.")

    # --- Process command ---
    parser_process = subparsers.add_parser("process", help="Process cached data to generate and save a knowledge graph.")
    parser_process.add_argument("target", help="The name of the target to process from cache (e.g., 'langchain' or 'all').")
    
    # --- Run command (all-in-one) ---
    parser_run = subparsers.add_parser("run", help="Run the entire pipeline: scrape (if needed) and process.")
    parser_run.add_argument("target", help="The name of the target to run the full pipeline for (e.g., 'langchain' or 'all').")

    args = parser.parse_args()
    config = load_config()
    targets_to_run = config.get("targets", {}).keys() if args.target.lower() == 'all' else [args.target]

    for target_name in targets_to_run:
        if args.command == "scrape":
            scrape_and_cache_target(target_name, config, args.force)
        elif args.command == "process":
            process_target_from_cache(target_name, config)
        elif args.command == "run":
            run_full_pipeline(target_name, config)