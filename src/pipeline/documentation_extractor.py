import logging
import os
import re
import json
import sys
import tomli  # Use tomli for TOML parsing
from typing import Any, Dict, List

import networkx as nx
import spacy
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from markdown_it import MarkdownIt

# --- 1. Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)


# --- 2. Enhanced Data Cleaning Module (No changes needed) ---
class EnhancedDataCleaner:
    """
    A sophisticated data cleaner for documentation repositories that handles code blocks,
    removes boilerplate content, and normalizes text for NLP processing.
    """

    def __init__(self):
        # ... (previous code is unchanged)
        self.code_block_pattern = re.compile(r"```.*?```", re.DOTALL)
        self.inline_code_pattern = re.compile(r"`[^`]*`")
        self.boilerplate_patterns = [
            re.compile(r"on this page", re.IGNORECASE),
            re.compile(r"table of contents", re.IGNORECASE),
            re.compile(r"was this page helpful\?", re.IGNORECASE),
            re.compile(r"edit this page on.*", re.IGNORECASE),
            re.compile(r"next\s*→", re.IGNORECASE),
            re.compile(r"←\s*previous", re.IGNORECASE),
            re.compile(r"© \d{4}.* All rights reserved.", re.IGNORECASE),
        ]
        self.markdown_link_pattern = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
        self.extra_whitespace_pattern = re.compile(r"[ \t]+")
        self.extra_newlines_pattern = re.compile(r"\n{3,}")
        self.md = MarkdownIt()

    def _remove_html_and_markdown(self, text: str) -> str:
        try:
            text = self.markdown_link_pattern.sub(r"\1", text)
            html = self.md.render(text)
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text()
        except Exception as e:
            logging.error("Error removing HTML/Markdown: %s", e)
            return text

    def clean(self, markdown_content: str) -> str:
        if not isinstance(markdown_content, str):
            logging.warning("Markdown content was not a string. Returning empty.")
            return ""
        logging.debug("Starting cleaning process...")
        text = self.code_block_pattern.sub("", markdown_content)
        text = self.inline_code_pattern.sub("", text)
        text = self._remove_html_and_markdown(text)
        for pattern in self.boilerplate_patterns:
            text = pattern.sub("", text)
        text = text.lower()
        text = self.extra_whitespace_pattern.sub(" ", text)
        text = self.extra_newlines_pattern.sub("\n\n", text)
        logging.debug("Cleaning process finished.")
        return text.strip()


# --- 3. Pipeline Modules ---

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


def save_to_cache(data: List[Dict[str, Any]], cache_path: str) -> None:
    """Saves scraped data to a JSON file to avoid re-scraping."""
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
                data = json.load(f)
                logging.info("Successfully loaded %d documents from cache.", len(data))
                return data
        except (IOError, json.JSONDecodeError) as e:
            logging.error("Failed to load or parse cache file, will re-scrape: %s", e)
    return []


def scrape_documentation(url: str, limit: int) -> List[Dict[str, Any]]:
    """Scrapes a documentation website using the Firecrawl API."""
    # ... (function content is the same, just removed hardcoded values)
    logging.info("Starting scrape for URL: %s", url)
    load_dotenv()
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        logging.error("FIRECRAWL_API_KEY not found in environment variables.")
        raise ValueError("FIRECRAWL_API_KEY is not set.")

    try:
        app = FirecrawlApp(api_key=api_key)
        scraped_data = app.crawl(
            url,
            params={
                "crawlerOptions": {"limit": limit},
                "pageOptions": {"onlyMainContent": True},
            },
        )
        logging.info("Successfully scraped %d documents.", len(scraped_data))
        return scraped_data
    except Exception as e:
        logging.error("An error occurred during scraping: %s", e)
        return []


def create_knowledge_graph(documents: List[Dict[str, Any]]) -> nx.DiGraph:
    """Creates a knowledge graph from cleaned documentation text."""
    # ... (function content is unchanged)
    logging.info("Starting knowledge graph generation...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logging.error("spaCy model 'en_core_web_sm' not found.")
        logging.error("Please run: python -m spacy download en_core_web_sm")
        raise

    G = nx.DiGraph()
    total_docs = len(documents)
    for i, doc_data in enumerate(documents):
        text = doc_data.get("cleaned_text", "")
        if not text:
            continue

        if (i + 1) % 50 == 0: # Log progress every 50 documents
            logging.info("Processing document %d/%d...", i + 1, total_docs)
            
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "PERSON", "GPE", "WORK_OF_ART"]:
                if not G.has_node(ent.text):
                    G.add_node(ent.text, label=ent.label_)

    logging.info("Graph generation complete. Found %d nodes.", G.number_of_nodes())
    return G


def save_graph(graph: nx.DiGraph, output_dir: str, target_name: str) -> None:
    """Saves the knowledge graph to a GraphML file, named after the target."""
    logging.info("Saving knowledge graph to directory: %s", output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{target_name}_graph.graphml")
        nx.write_graphml(graph, file_path)
        logging.info("Graph successfully saved to %s", file_path)
    except Exception as e:
        logging.error("An unexpected error occurred while saving the graph: %s", e)


# --- 4. Main Execution Block ---
def main():
    """Main function to run the data pipeline for a specified target."""
    logging.info("--- Starting Documentation to Knowledge Graph Pipeline ---")
    
    try:
        # --- Configuration & Setup ---
        config = load_config()
        if len(sys.argv) < 2:
            logging.error("Usage: python -m pipeline.run_pipeline <target_name>")
            logging.info("Available targets are: %s", ", ".join(config.get('targets', {}).keys()))
            sys.exit(1)
            
        target_name = sys.argv[1]
        target_config = config.get("targets", {}).get(target_name)
        if not target_config:
            logging.error("Target '%s' not found in config.toml.", target_name)
            sys.exit(1)
        
        pipeline_config = config.get("pipeline", {})
        cache_dir = pipeline_config.get("cache_dir", ".cache")
        output_dir = pipeline_config.get("output_dir", "repository")
        page_limit = target_config.get("page_limit", 1000)
        target_url = target_config.get("url")

        cache_file = os.path.join(cache_dir, f"{target_name}_scrape_data.json")

        # --- Step 1: Scrape (or Load from Cache) ---
        scraped_docs = load_from_cache(cache_file)
        if not scraped_docs:
            logging.info("No cache found, starting web scrape...")
            scraped_docs = scrape_documentation(url=target_url, limit=page_limit)
            if scraped_docs:
                save_to_cache(scraped_docs, cache_file)

        if not scraped_docs:
            logging.warning("Scraping returned no documents. Exiting pipeline.")
            return

        # --- Step 2: Clean ---
        logging.info("Starting data cleaning process...")
        cleaner = EnhancedDataCleaner()
        for doc in scraped_docs:
            doc["cleaned_text"] = cleaner.clean(doc.get("markdown", ""))

        # --- Step 3: Generate Graph ---
        knowledge_graph = create_knowledge_graph(scraped_docs)

        # --- Step 4: Save Graph ---
        if knowledge_graph.number_of_nodes() > 0:
            save_graph(knowledge_graph, output_dir=output_dir, target_name=target_name)
        else:
            logging.warning("Knowledge graph is empty. Nothing to save.")

    except Exception as e:
        logging.critical("A critical error occurred in the pipeline: %s", e, exc_info=True)
    finally:
        logging.info("--- Pipeline Execution Finished ---")


if __name__ == "__main__":
    main()