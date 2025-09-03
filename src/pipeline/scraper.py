import logging
import os
from collections import deque
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import hrequests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from markdownify import markdownify
from readability import Document as ReadabilityDocument

# Conditional import for Firecrawl, allowing the module to work without it
try:
    from firecrawl import Firecrawl
except ImportError:
    logging.warning("Firecrawl library not found. Firecrawl fallback will not be available.")
    Firecrawl = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _is_same_domain(base_url: str, new_url: str) -> bool:
    """Checks if two URLs belong to the same domain."""
    return urlparse(base_url).netloc == urlparse(new_url).netloc

def _get_main_content_markdown(html_content: str, url: str) -> Optional[str]:
    """
    Extracts the main content from HTML using readability and converts it to Markdown.
    """
    try:
        # Using readability to get the main article content
        doc = ReadabilityDocument(html_content)
        main_html = doc.summary()

        if main_html:
            # Clean up the HTML before converting to markdown
            soup = BeautifulSoup(main_html, 'lxml')
            # Remove script, style, and other non-content tags that might be in summary
            for unwanted_tag in soup(['script', 'style', 'noscript', 'img', 'iframe', 'svg', 'canvas', 'picture', 'source']):
                unwanted_tag.decompose()
            
            # Convert the cleaned HTML to markdown
            markdown_content = markdownify(str(soup), heading_style="ATX", strong_em_symbol="*", default_title=doc.title())
            return markdown_content.strip()
        return None
    except Exception as e:
        logging.error("Failed to extract main content or convert to markdown for %s: %s", url, e, exc_info=True)
        return None

def _scrape_with_hrequests(start_url: str, limit: int) -> List[Dict[str, Any]]:
    """
    Scrapes a documentation website using hrequests and BeautifulSoup.
    Prioritizes main content extraction using readability.
    """
    logging.info("Attempting primary scrape with hrequests for URL: %s (limit: %d)", start_url, limit)
    session = hrequests.Session()
    # Use a common User-Agent to mimic a browser and avoid being blocked
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36'
    })

    queue = deque([start_url])
    visited_urls = set()
    results: List[Dict[str, Any]] = []

    base_domain = urlparse(start_url).netloc

    while queue and len(results) < limit:
        current_url = queue.popleft()

        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)
        logging.info("Scraping page: %s (Documents found: %d/%d)", current_url, len(results), limit)

        try:
            response = session.get(current_url, timeout=15) # Increased timeout
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            html_content = response.text

            markdown = _get_main_content_markdown(html_content, current_url)
            if markdown:
                # Extract title from the original full HTML for better accuracy
                soup_full = BeautifulSoup(html_content, 'lxml')
                title_tag = soup_full.find('title')
                title = title_tag.get_text(strip=True) if title_tag else "No Title"

                results.append({
                    "markdown": markdown,
                    "metadata": {
                        "url": current_url,
                        "title": title
                    }
                })
            else:
                logging.warning("No main content extracted for %s. This page will not be included in results.", current_url)

            # Find links to crawl further
            soup_links = BeautifulSoup(html_content, 'lxml')
            for link in soup_links.find_all('a', href=True):
                href = link['href']
                # Resolve relative URLs and remove URL fragments
                absolute_url = urljoin(current_url, href).split('#')[0]

                # Only follow links on the same domain and not yet visited
                if _is_same_domain(start_url, absolute_url) and absolute_url not in visited_urls and absolute_url not in queue:
                    queue.append(absolute_url)

        except hrequests.exceptions.HTTPStatusError as e:
            logging.warning("HTTP error scraping %s: %s", current_url, e)
        except hrequests.exceptions.RequestException as e:
            logging.warning("Request error scraping %s: %s", current_url, e)
        except Exception as e:
            logging.error("An unexpected error occurred while processing %s: %s", current_url, e, exc_info=True)

    logging.info("Hrequests scraping completed. Found %d documents.", len(results))
    return results

def _scrape_with_firecrawl(url: str, limit: int) -> List[Dict[str, Any]]:
    """
    Scrapes a documentation website using the Firecrawl API.
    This function acts as a fallback if the primary hrequests method fails.
    """
    if Firecrawl is None:
        logging.error("Firecrawl library not available. Cannot use Firecrawl fallback.")
        return []

    logging.info("Attempting Firecrawl fallback scrape for URL: %s (limit: %d)", url, limit)
    load_dotenv()
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        logging.error("FIRECRAWL_API_KEY not found in environment variables. Cannot use Firecrawl fallback.")
        return []

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
                'proxy': 'auto', # Firecrawl handles proxy automatically
                'maxAge': 600000, # Cache results for 10 minutes
                'onlyMainContent': True
            }
        )

        if not crawl_result or not crawl_result.data:
            logging.warning("Firecrawl finished but returned no data for URL: %s", url)
            return []

        logging.info("Firecrawl successful. Credits used: %d. Found %d documents.",
                     crawl_result.credits_used, len(crawl_result.data))

        # Process the results into the expected format (list of dicts)
        processed_data = []
        for doc in crawl_result.data:
            # Convert the DocumentMetadata object to a dictionary for consistent data handling.
            metadata_as_dict = doc.metadata.dict() if doc.metadata else {}
            processed_data.append({
                "markdown": doc.markdown,
                "metadata": metadata_as_dict
            })
        return processed_data

    except Exception as e:
        logging.error("An unexpected error occurred during the Firecrawl API call: %s", e, exc_info=True)
        return []

def scrape_documentation(url: str, limit: int = 10, use_firecrawl_fallback: bool = True) -> List[Dict[str, Any]]:
    """
    Scrapes a documentation website.
    Attempts to use hrequests and readability-lxml first.
    If that fails to return any results and `use_firecrawl_fallback` is True,
    it falls back to using the Firecrawl API.

    Args:
        url: The starting URL for the crawl.
        limit: The maximum number of pages to crawl.
        use_firecrawl_fallback: If True, attempts to use Firecrawl as a fallback
                                if the primary scraping method yields no results.

    Returns:
        A list of dictionaries, where each dictionary represents a scraped document.
    """
    logging.info("Initiating documentation scrape for URL: %s (limit: %d)", url, limit)

    # Attempt primary scraping method using hrequests and readability
    hrequests_results = _scrape_with_hrequests(url, limit)

    if hrequests_results:
        logging.info("Primary scraping with hrequests successfully obtained %d documents.", len(hrequests_results))
        return hrequests_results
    elif use_firecrawl_fallback:
        logging.warning("Primary hrequests scraping yielded no results. Attempting Firecrawl fallback.")
        # Pass the original limit to Firecrawl
        return _scrape_with_firecrawl(url, limit)
    else:
        logging.warning("Primary hrequests scraping yielded no results and Firecrawl fallback is disabled. Returning empty list.")
        return []