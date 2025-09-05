# mcp_server.py
import os
import logging
import requests
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from a .env file if it exists
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Get the URL for the Knowledge API from environment variables, with a default for local dev
KNOWLEDGE_API_URL = os.environ.get("KNOWLEDGE_API_URL", "http://localhost:8080/query")

# --- MCP Server Setup ---
# Initialize the MCP server with a name that describes its capability.
# This name is visible to LLMs that discover it.
mcp = FastMCP("Software Documentation Knowledge Base")

@mcp.tool()
def query_knowledge_base(target: str, query_text: str) -> str:
    """
    Queries a specific software framework's knowledge base to get up-to-date, grounded information.

    Use this tool when a user asks a question about a specific, modern software framework,
    SDK, or technical library.

    Args:
        target (str): The name of the software framework to query (e.g., 'langchain', 'firecrawl', 'mintlify').
        query_text (str): The natural language question you want to ask about the target framework.
    
    Returns:
        A string containing the most relevant information found, including source URLs.
        Returns an error message if the knowledge base for the target cannot be found or if the API is unavailable.
    """
    logging.info(f"MCP Tool: Received query for target='{target}', query='{query_text}'")
    
    payload = {
        "target": target,
        "query_text": query_text,
        "n_results": 3  # A good default for providing concise context to an LLM
    }

    try:
        response = requests.post(KNOWLEDGE_API_URL, json=payload, timeout=30)
        
        # Raise an exception for HTTP error codes (4xx or 5xx)
        response.raise_for_status()
        
        data = response.json()
        
        # --- Format the output for the LLM ---
        # The goal is to create a clean, readable string that the LLM can easily parse.
        docs = data.get('documents', [[]])[0]
        metadatas = data.get('metadatas', [[]])[0]

        if not docs:
            return f"No relevant information found in the '{target}' knowledge base for your query."

        formatted_results = [
            f"Source: {meta.get('source_url', 'N/A')}\nContent: {doc}"
            for doc, meta in zip(docs, metadatas)
        ]
        
        return "\n\n---\n\n".join(formatted_results)

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(f"Knowledge pack for target '{target}' not found via API.")
            return f"Error: The knowledge base for the target '{target}' does not exist. Please try a different target."
        logging.error(f"API returned a server error: {e.response.text}")
        return f"Error: The Knowledge API returned a server error: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not connect to the Knowledge API at {KNOWLEDGE_API_URL}. Details: {e}")
        return f"Error: Could not connect to the Knowledge API. Please ensure the API server is running and accessible."

# --- Run the MCP Server ---
if __name__ == '__main__':
    logging.info(f"Starting MCP server for '{mcp.name}'...")
    logging.info("This server exposes tools for LLMs to use.")
    logging.info(f"Make sure the Knowledge API is running and accessible at: {KNOWLEDGE_API_URL}")
    mcp.run()