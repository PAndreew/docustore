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

def _extract_rich_metadata(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """
    Extracts rich metadata from a parsed BeautifulSoup object.
    
    Args:
        soup: A BeautifulSoup object of the HTML page.
        url: The URL of the page, used for resolving relative links.

    Returns:
        A dictionary containing extracted metadata.
    """
    metadata = {
        "source_url": url,
        "title": None,
        "description": None,
        "language": None,
        "keywords": None,
        "og_title": None,
        "og_description": None,
        "og_image": None,
        "og_url": None,
        "favicon": None
    }

    # Extract title
    if soup.title and soup.title.string:
        metadata["title"] = soup.title.string.strip()

    # Extract language from <html> tag
    if soup.html and soup.html.get('lang'):
        metadata["language"] = soup.html.get('lang')

    # Extract meta tags (description, keywords, Open Graph)
    for tag in soup.find_all('meta'):
        if tag.get('name') == 'description':
            metadata['description'] = tag.get('content')
        elif tag.get('name') == 'keywords':
            metadata['keywords'] = tag.get('content')
        elif tag.get('property') == 'og:title':
            metadata['og_title'] = tag.get('content')
        elif tag.get('property') == 'og:description':
            metadata['og_description'] = tag.get('content')
        elif tag.get('property') == 'og:image':
            metadata['og_image'] = tag.get('content')
        elif tag.get('property') == 'og:url':
            metadata['og_url'] = tag.get('content')

    # Fallback to OG title if main title is missing
    if not metadata["title"] and metadata["og_title"]:
        metadata["title"] = metadata["og_title"]
        
    # Extract favicon
    favicon_link = soup.find('link', rel=lambda rel: rel and 'icon' in rel)
    if favicon_link and favicon_link.get('href'):
        # Resolve relative URL for the favicon
        metadata['favicon'] = urljoin(url, favicon_link['href'])
        
    # Clean up None values
    return {k: v for k, v in metadata.items() if v is not None}


def _get_main_content_markdown(html_content: str, url: str) -> Optional[str]:
    """
    Extracts the main content from HTML using readability and converts it to Markdown.
    """
    try:
        doc = ReadabilityDocument(html_content)
        main_html = doc.summary()

        if main_html:
            soup = BeautifulSoup(main_html, 'lxml')
            for unwanted_tag in soup(['script', 'style', 'noscript', 'img', 'iframe', 'svg', 'canvas', 'picture', 'source']):
                unwanted_tag.decompose()
            
            markdown_content = markdownify(str(soup), heading_style="ATX", strong_em_symbol="*", default_title=doc.title())
            return markdown_content.strip()
        return None
    except Exception as e:
        logging.error("Failed to extract main content or convert to markdown for %s: %s", url, e, exc_info=True)
        return None

def _scrape_with_hrequests(start_url: str, limit: int) -> List[Dict[str, Any]]:
    """
    Scrapes a documentation website using hrequests and BeautifulSoup.
    Prioritizes main content extraction using readability and gathers rich metadata.
    """
    logging.info("Attempting primary scrape with hrequests for URL: %s (limit: %d)", start_url, limit)
    session = hrequests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36'
    })

    queue = deque([start_url])
    visited_urls = set()
    results: List[Dict[str, Any]] = []

    while queue and len(results) < limit:
        current_url = queue.popleft()

        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)
        logging.info("Scraping page: %s (Documents found: %d/%d)", current_url, len(results), limit)

        try:
            response = session.get(current_url, timeout=15)
            
            if not response.ok:
                logging.warning(
                    "Request to %s failed with status code %d: %s",
                    current_url, response.status_code, response.reason
                )
                continue

            html_content = response.text
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract rich metadata first
            metadata = _extract_rich_metadata(soup, current_url)

            # Extract main content as markdown
            markdown = _get_main_content_markdown(html_content, current_url)
            
            if markdown:
                results.append({
                    "markdown": markdown,
                    "metadata": metadata
                })
            else:
                logging.warning("No main content extracted for %s. This page will not be included in results.", current_url)

            # Find and queue new links from the same domain
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(current_url, href).split('#')[0]

                if _is_same_domain(start_url, absolute_url) and absolute_url not in visited_urls and absolute_url not in queue:
                    queue.append(absolute_url)

        except hrequests.exceptions.ClientException as e:
            logging.warning("Request failed for %s: %s", current_url, e)
        except Exception as e:
            logging.error("An unexpected error occurred while processing %s: %s", current_url, e, exc_info=True)

    logging.info("Hrequests scraping completed. Found %d documents.", len(results))
    return results

def _scrape_with_firecrawl(url: str, limit: int) -> List[Dict[str, Any]]:
    """
    Scrapes a documentation website using the Firecrawl API. (Fallback method)
    """
    # ... (This function remains unchanged)
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
        crawl_result = firecrawl.crawl(
            url=url,
            limit=limit,
            scrape_options={'formats': ['markdown'], 'onlyMainContent': True}
        )
        if not crawl_result or not crawl_result.data:
            logging.warning("Firecrawl finished but returned no data for URL: %s", url)
            return []

        logging.info("Firecrawl successful. Credits used: %d. Found %d documents.",
                     crawl_result.credits_used, len(crawl_result.data))

        processed_data = []
        for doc in crawl_result.data:
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
    Scrapes a documentation website, prioritizing hrequests and falling back to Firecrawl.
    """
    # ... (This function remains unchanged)
    logging.info("Initiating documentation scrape for URL: %s (limit: %d)", url, limit)

    hrequests_results = _scrape_with_hrequests(url, limit)

    if hrequests_results:
        logging.info("Primary scraping with hrequests successfully obtained %d documents.", len(hrequests_results))
        return hrequests_results
    elif use_firecrawl_fallback:
        logging.warning("Primary hrequests scraping yielded no results. Attempting Firecrawl fallback.")
        return _scrape_with_firecrawl(url, limit)
    else:
        logging.warning("Primary hrequests scraping yielded no results and Firecrawl fallback is disabled. Returning empty list.")
        return []