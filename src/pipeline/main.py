# main.py
import logging
import os
import argparse
import tarfile
from typing import Any, Dict

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)

# --- Import pipeline modules ---
from pipeline.config import load_config
from pipeline.scraper import scrape_documentation
from pipeline.storage import save_to_cache, load_from_cache
# Switch from graph_creator to vector_processor
from pipeline.vector_processor import process_and_embed

# --- Stage 1: Scraping and Caching ---
def scrape_and_cache_target(target_name: str, config: Dict[str, Any], force: bool = False):
    """Handles the scraping and caching stage for a specific target."""
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

# --- Stage 2: Vector Processing from Cache ---
def process_target_from_cache(target_name: str, config: Dict[str, Any]):
    """
    Loads cached data, processes it into chunks, creates vector embeddings,
    and saves the resulting Knowledge Pack.
    """
    logging.info("--- Starting Vector Processing Stage for Target: %s ---", target_name)
    pipeline_config = config.get("pipeline", {})
    vector_store_config = config.get("vector_store", {})
    cache_dir = pipeline_config.get("cache_dir", ".cache")
    output_dir = pipeline_config.get("output_dir", "repository")
    cache_file = os.path.join(cache_dir, f"{target_name}_scrape_data.json")

    # Load data exclusively from the cache
    docs_from_cache = load_from_cache(cache_file)
    if not docs_from_cache:
        logging.error("No cache file found for '%s'. Please run the 'scrape' command first.", target_name)
        return

    # Call the vector processor
    embedding_model = vector_store_config.get("embedding_model_name", "all-MiniLM-L6-v2")
    process_and_embed(
        documents=docs_from_cache,
        output_dir=output_dir,
        target_name=target_name,
        model_name=embedding_model
    )

# --- Stage 3: Packaging ---
def package_target(target_name: str, config: Dict[str, Any]):
    """Packages the processed output for a target into a distributable .tar.gz archive."""
    logging.info("--- Creating Knowledge Pack for Target: %s ---", target_name)
    pipeline_config = config.get("pipeline", {})
    output_dir = pipeline_config.get("output_dir", "repository")
    
    source_dir = os.path.join(output_dir, target_name)
    if not os.path.isdir(source_dir):
        logging.error("Processed output not found for '%s' at %s. Run 'process' first.", target_name, source_dir)
        return

    package_name = f"{target_name}-knowledge-pack"
    archive_name = f"{package_name}.tar.gz"
    archive_path = os.path.join(output_dir, archive_name)

    try:
        logging.info("Compressing %s into %s...", source_dir, archive_path)
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(source_dir, arcname=package_name)
        logging.info("Successfully created Knowledge Pack: %s", archive_path)
    except Exception as e:
        logging.error("Failed to create package: %s", e)

# --- Full Pipeline Orchestrator ---
def run_full_pipeline(target_name: str, config: Dict[str, Any]):
    """Runs the entire pipeline sequentially: scrape -> cache -> process."""
    logging.info("--- Starting Full Pipeline for Target: %s ---", target_name)
    scrape_and_cache_target(target_name, config, force=False)
    process_target_from_cache(target_name, config)
    logging.info("--- Full Pipeline Finished for Target: %s ---", target_name)

# --- GCP Entrypoint ---
def gcp_entrypoint(event, context):
    """Cloud Function entry point. Runs the full scrape and process pipeline."""
    target = event.get('attributes', {}).get('target')
    if not target:
        logging.error("Pub/Sub message must have a 'target' attribute.")
        return
    config = load_config()
    run_full_pipeline(target, config)

# --- Local Execution with Sub-commands ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Pack Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Scrape command
    parser_scrape = subparsers.add_parser("scrape", help="Scrape a target website and save the results to cache.")
    parser_scrape.add_argument("target", help="The name of the target to scrape (e.g., 'langchain' or 'all').")
    parser_scrape.add_argument("--force", action="store_true", help="Force re-scraping even if cache exists.")

    # Process command
    parser_process = subparsers.add_parser("process", help="Process cached data to generate a vector store knowledge pack.")
    parser_process.add_argument("target", help="The name of the target to process from cache (e.g., 'langchain' or 'all').")
    
    # Package command
    parser_package = subparsers.add_parser("package", help="Compress a processed knowledge pack into a .tar.gz archive.")
    parser_package.add_argument("target", help="The name of the target to package (e.g., 'langchain' or 'all').")

    # Run command (all-in-one)
    parser_run = subparsers.add_parser("run", help="Run the entire pipeline: scrape (if needed) and process.")
    parser_run.add_argument("target", help="The name of the target to run the full pipeline for (e.g., 'langchain' or 'all').")

    args = parser.parse_args()
    config = load_config()
    
    if args.target.lower() == 'all':
        targets_to_run = config.get("targets", {}).keys()
    else:
        targets_to_run = [args.target]

    for target_name in targets_to_run:
        if args.command == "scrape":
            scrape_and_cache_target(target_name, config, args.force)
        elif args.command == "process":
            process_target_from_cache(target_name, config)
        elif args.command == "package":
            package_target(target_name, config)
        elif args.command == "run":
            run_full_pipeline(target_name, config)