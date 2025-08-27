# pipeline/scraper.py
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from firecrawl import Firecrawl

def scrape_documentation(url: str, limit: int) -> List[Dict[str, Any]]:
    """
    Scrapes a documentation website using the Firecrawl API.

    This function initiates a crawl job, waits for it to complete, and then
    formats the results into a list of dictionaries, each containing the
    markdown content and metadata of a scraped page.

    Args:
        url: The starting URL for the crawl.
        limit: The maximum number of pages to crawl.

    Returns:
        A list of dictionaries, where each dictionary represents a scraped document.
    """
    logging.info("Starting scrape for URL: %s with limit: %d", url, limit)
    load_dotenv()
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        logging.error("FIRECRAWL_API_KEY not found in environment variables.")
        raise ValueError("FIRECRAWL_API_KEY is not set.")

    try:
        firecrawl = Firecrawl(api_key=api_key)

        # Call the crawl method. This is a blocking call.
        crawl_result = firecrawl.crawl(
            url=url,
            limit=limit,
            scrape_options={
                'formats': [
                    'markdown',
                    {'type': 'json',
                     'schema': {'type': 'object', 'properties': {'title': {'type': 'string'}}}
                     }
                ],
                'proxy': 'auto',
                'maxAge': 600000,
                'onlyMainContent': True
            }
        )

        if not crawl_result or not crawl_result.data:
            logging.warning("Crawl finished but returned no data for URL: %s", url)
            return []

        # CORRECTED: Use the snake_case attribute 'credits_used'
        logging.info("Crawl successful. Credits used: %d. Found %d documents.",
                     crawl_result.credits_used, len(crawl_result.data))

        # Process the results into the expected format (list of dicts)
        processed_data = []
        for doc in crawl_result.data:
            # CORRECTED AND ENHANCED: Convert the DocumentMetadata object to a dictionary
            # This is critical for JSON serialization (caching) and consistent data handling.
            # The firecrawl-py models have a .dict() method for this purpose.
            metadata_as_dict = doc.metadata.dict() if doc.metadata else {}

            processed_data.append({
                "markdown": doc.markdown,
                "metadata": metadata_as_dict
            })

        return processed_data

    except Exception as e:
        logging.error("An unexpected error occurred during the Firecrawl API call: %s", e, exc_info=True)
        return []
    

            