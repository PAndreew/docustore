import logging
import os
import re
from typing import Any, Dict, List

import networkx as nx
import spacy
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from markdown_it import MarkdownIt

# --- 1. Logging Configuration ---
# Configure logging to output to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)


# --- 2. Enhanced Data Cleaning Module ---
class EnhancedDataCleaner:
    """
    A sophisticated data cleaner for documentation repositories that handles code blocks,
    removes boilerplate content, and normalizes text for NLP processing.
    """

    def __init__(self):
        # Pre-compile regex patterns for efficiency
        # Matches multi-line code blocks (e.g., ```python...```)
        self.code_block_pattern = re.compile(r"```.*?```", re.DOTALL)
        # Matches inline code snippets (e.g., `my_function()`)
        self.inline_code_pattern = re.compile(r"`[^`]*`")
        # Matches common boilerplate text patterns found in documentation
        self.boilerplate_patterns = [
            re.compile(r"on this page", re.IGNORECASE),
            re.compile(r"table of contents", re.IGNORECASE),
            re.compile(r"was this page helpful\?", re.IGNORECASE),
            re.compile(r"edit this page on.*", re.IGNORECASE),
            re.compile(r"next\s*→", re.IGNORECASE),
            re.compile(r"←\s*previous", re.IGNORECASE),
            re.compile(r"© \d{4}.* All rights reserved.", re.IGNORECASE),
        ]
        # Matches markdown links like [text](url) to extract the text
        self.markdown_link_pattern = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
        # Matches multiple spaces and newlines
        self.extra_whitespace_pattern = re.compile(r"[ \t]+")
        self.extra_newlines_pattern = re.compile(r"\n{3,}")
        self.md = MarkdownIt()

    def _remove_html_and_markdown(self, text: str) -> str:
        """Converts markdown to HTML and then strips all tags for a plain text representation."""
        try:
            # First, replace markdown links with just their text
            text = self.markdown_link_pattern.sub(r"\1", text)
            # Convert the remaining markdown to HTML
            html = self.md.render(text)
            # Use BeautifulSoup to parse the HTML and get the text
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text()
        except Exception as e:
            logging.error("Error removing HTML/Markdown: %s", e)
            return text  # Return original text on failure

    def clean(self, markdown_content: str) -> str:
        """
        Executes the full multi-stage cleaning pipeline on raw markdown text.

        Args:
            markdown_content (str): The raw markdown from the scraper.

        Returns:
            str: The cleaned, high-quality text ready for NLP.
        """
        if not isinstance(markdown_content, str):
            logging.warning("Markdown content was not a string. Returning empty.")
            return ""

        logging.debug("Starting cleaning process...")
        # Stage 1: Handle code blocks (remove them to focus on narrative)
        text = self.code_block_pattern.sub("", markdown_content)
        text = self.inline_code_pattern.sub("", text)

        # Stage 2: Remove HTML and basic markdown syntax
        text = self._remove_html_and_markdown(text)

        # Stage 3: Remove boilerplate sections
        for pattern in self.boilerplate_patterns:
            text = pattern.sub("", text)

        # Stage 4: Text Normalization
        text = text.lower()  # Convert to lowercase
        text = self.extra_whitespace_pattern.sub(" ", text)  # Normalize spaces
        text = self.extra_newlines_pattern.sub("\n\n", text)  # Normalize newlines

        logging.debug("Cleaning process finished.")
        return text.strip()


# --- 3. Pipeline Modules (Functions) ---


def scrape_documentation(url: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Scrapes a documentation website using the Firecrawl API.

    Args:
        url (str): The starting URL of the documentation to scrape.
        limit (int): The maximum number of pages to crawl.

    Returns:
        A list of scraped page data dictionaries.
    """
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
    """
    Creates a knowledge graph from the cleaned documentation text using spaCy for NER.

    Args:
        documents: A list of documents, each containing 'cleaned_text'.

    Returns:
        A NetworkX DiGraph representing the knowledge graph.
    """
    logging.info("Starting knowledge graph generation...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logging.error("spaCy model 'en_core_web_sm' not found.")
        logging.error("Please run: python -m spacy download en_core_web_sm")
        raise

    G = nx.DiGraph()
    for i, doc_data in enumerate(documents):
        text = doc_data.get("cleaned_text", "")
        if not text:
            continue

        logging.info("Processing document %d/%d...", i + 1, len(documents))
        doc = nlp(text)

        # Add entities as nodes
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "PRODUCT", "GPE", "WORK_OF_ART"]:
                if not G.has_node(ent.text):
                    G.add_node(ent.text, label=ent.label_)

    logging.info("Graph generation complete. Found %d nodes.", G.number_of_nodes())
    # Relationship extraction logic can be added here as the module evolves.
    return G


def save_graph(graph: nx.DiGraph, output_dir: str = "graph_data") -> None:
    """
    Saves the knowledge graph to a GraphML file.

    Args:
        graph: The NetworkX graph to save.
        output_dir: The directory to save the graph file in.
    """
    logging.info("Saving knowledge graph to directory: %s", output_dir)
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = os.path.join(output_dir, "documentation_graph.graphml")
        nx.write_graphml(graph, file_path)
        logging.info("Graph successfully saved to %s", file_path)
    except OSError as e:
        logging.error("Failed to save graph due to an I/O error: %s", e)
    except Exception as e:
        logging.error("An unexpected error occurred while saving the graph: %s", e)


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    TARGET_URL = (
        "https://docs.firecrawl.dev/"  # Change to your target documentation URL
    )
    PAGE_LIMIT = 500  # Adjust as needed
    OUTPUT_DIR = "repository"

    logging.info("--- Starting Documentation to Knowledge Graph Pipeline ---")

    try:
        # Step 1: Scrape
        scraped_docs = scrape_documentation(url=TARGET_URL, limit=PAGE_LIMIT)

        if not scraped_docs:
            logging.warning("Scraping returned no documents. Exiting pipeline.")
        else:
            # Step 2: Clean
            logging.info("Starting data cleaning process...")
            cleaner = EnhancedDataCleaner()
            for doc in scraped_docs:
                doc["cleaned_text"] = cleaner.clean(doc.get("markdown", ""))

            # Step 3: Generate Graph
            knowledge_graph = create_knowledge_graph(scraped_docs)

            # Step 4: Save Graph
            if knowledge_graph.number_of_nodes() > 0:
                save_graph(knowledge_graph, output_dir=OUTPUT_DIR)
            else:
                logging.warning("Knowledge graph is empty. Nothing to save.")

    except Exception as e:
        logging.critical("A critical error occurred in the pipeline: %s", e, exc_info=True)

    logging.info("--- Pipeline Execution Finished ---")
