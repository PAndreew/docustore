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

def run_pipeline_for_target(target_name: str, config: Dict[str, Any]):
    """
    Executes the full data pipeline for a single specified target.

    Args:
        target_name: The key of the target in the config file (e.g., 'langchain').
        config: The loaded configuration dictionary.
    """
    logging.info("--- Starting Pipeline for Target: %s ---", target_name)

    target_config = config.get("targets", {}).get(target_name)
    if not target_config:
        logging.error("Target '%s' not found in config.toml.", target_name)
        return

    # Extract configs with defaults
    pipeline_config = config.get("pipeline", {})
    kg_config = config.get("knowledge_graph", {})

    cache_dir = pipeline_config.get("cache_dir", ".cache")
    output_dir = pipeline_config.get("output_dir", "repository")
    page_limit = target_config.get("page_limit", 1000)
    target_url = target_config.get("url")
    cache_file = os.path.join(cache_dir, f"{target_name}_scrape_data.json")

    # --- Step 1: Scrape (or Load from Cache) ---
    scraped_docs = load_from_cache(cache_file)
    if not scraped_docs:
        logging.info("No cache found for '%s', starting web scrape...", target_name)
        scraped_docs = scrape_documentation(url=target_url, limit=page_limit)
        if scraped_docs:
            save_to_cache(scraped_docs, cache_file)

    if not scraped_docs:
        logging.warning("Scraping returned no documents for '%s'. Exiting.", target_name)
        return

    # --- Step 2: Generate Knowledge Graph ---
    # knowledge_graph = create_knowledge_graph(
    #     documents=scraped_docs,
    #     technical_entities=kg_config.get("technical_entities", []),
    #     enabled_spacy_entities=kg_config.get("enabled_spacy_entities", []),
    #     associate_descriptions=kg_config.get("associate_descriptions", False),
    # )

    # # --- Step 3: Save Graph ---
    # if knowledge_graph.number_of_nodes() > 0:
    #     save_graph(knowledge_graph, output_dir=output_dir, target_name=target_name)
    # else:
    #     logging.warning("Knowledge graph for '%s' is empty. Nothing to save.", target_name)
    
    logging.info("--- Pipeline Finished for Target: %s ---", target_name)

# --- GCP Entrypoint ---
def gcp_entrypoint(event, context):
    """
    Cloud Function entry point.
    Triggered by Pub/Sub, expecting a message with a 'target' attribute.
    
    Example Pub/Sub message body (can be empty):
    gcloud pubsub topics publish my-topic --message "" --attribute target=langchain
    """
    target = event.get('attributes', {}).get('target')
    if not target:
        logging.error("Pub/Sub message must have a 'target' attribute.")
        return

    logging.info("GCP Function triggered for target: %s", target)
    config = load_config()
    run_pipeline_for_target(target, config)

# --- Local Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Pipeline CLI")
    parser.add_argument(
        "target", 
        help="The name of the target to process (e.g., 'langchain' or 'firecrawl'). Use 'all' to run for all targets."
    )
    args = parser.parse_args()

    config = load_config()
    
    if args.target.lower() == 'all':
        targets = config.get("targets", {}).keys()
        if not targets:
            logging.warning("No targets found in the configuration file.")
        for target_name in targets:
            run_pipeline_for_target(target_name, config)
    else:
        run_pipeline_for_target(args.target, config)