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


def chunk_document_by_headers(markdown_content: str) -> List[Dict[str, str]]:
    """
    Splits a markdown document into chunks based on its headers.

    Args:
        markdown_content: The raw markdown text of a single page.

    Returns:
        A list of chunks, where each chunk is a dictionary with a 'header' and 'content'.
    """
    # Regex to find markdown headers (e.g., #, ##, ###)
    header_pattern = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
    chunks = []
    
    last_end = 0
    
    # Find all header matches
    matches = list(header_pattern.finditer(markdown_content))
    
    # Handle content before the first header
    if matches and matches[0].start() > 0:
        chunks.append({
            "header": "Introduction",
            "content": markdown_content[0:matches[0].start()].strip()
        })

    # Create a chunk for each header and its content
    for i, match in enumerate(matches):
        start = match.start()
        header_text = match.group(2).strip()
        
        # The content of this chunk is from the end of its header to the start of the next one
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_content)
        
        content = markdown_content[content_start:content_end].strip()
        
        chunks.append({"header": header_text, "content": content})
        
    # If no headers are found, treat the whole document as one chunk
    if not chunks and markdown_content:
        chunks.append({"header": "General", "content": markdown_content})
        
    return chunks

def create_knowledge_graph(
    documents: List[Dict[str, Any]],
    technical_entities: List[str],
    enabled_spacy_entities: List[str],
    associate_descriptions: bool,
) -> nx.DiGraph:
    """
    Creates a technical knowledge graph, associating entities with their descriptive text.
    """
    logging.info("Starting technical knowledge graph generation...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # ... (error handling is the same)
        raise

    G = nx.DiGraph()
    cleaner = EnhancedDataCleaner() # We'll need a cleaner instance here

    patterns = {
        "CLASS": re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]+)"),
        "FUNCTION": re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]+)"),
    }
    enabled_patterns = {k: v for k, v in patterns.items() if k in technical_entities}
    
    total_docs = len(documents)
    for i, doc_data in enumerate(documents):
        raw_markdown = doc_data.get("markdown", "")
        if not raw_markdown:
            continue
            
        if (i + 1) % 50 == 0:
            logging.info("Processing document %d/%d...", i + 1, total_docs)

        # Step 1: Split document into semantic chunks
        chunks = chunk_document_by_headers(raw_markdown)

        for chunk in chunks:
            # Step 2: Clean the text content of this specific chunk
            cleaned_chunk_content = cleaner.clean(chunk["content"])
            if not cleaned_chunk_content:
                continue

            # Step 3: Extract entities from this chunk
            found_entities = set()

            # Rule-based extraction
            for label, pattern in enabled_patterns.items():
                for match in pattern.finditer(cleaned_chunk_content):
                    found_entities.add((match.group(1), label))
            
            # spaCy-based extraction
            if enabled_spacy_entities:
                doc = nlp(cleaned_chunk_content)
                for ent in doc.ents:
                    if ent.label_ in enabled_spacy_entities:
                        found_entities.add((ent.text, ent.label_))
            
            # Step 4: Add entities to graph and associate description
            for entity_text, label in found_entities:
                if not G.has_node(entity_text):
                    node_attrs = {"label": label}
                    if associate_descriptions:
                        # Add the cleaned text of the chunk as the description
                        node_attrs["description"] = cleaned_chunk_content
                        logging.debug("Found entity '%s' and associated its description.", entity_text)
                    G.add_node(entity_text, **node_attrs)

    logging.info(
        "Graph generation complete. Found %d nodes.", G.number_of_nodes()
    )
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