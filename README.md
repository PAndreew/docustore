    
# docustore - Automated Knowledge Pack Builder

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**docustore** is an automated pipeline that scrapes software documentation, processes the content, and creates self-contained, downloadable **Knowledge Packs**. The goal is to provide developers and LLMs with up-to-date, grounded information, especially for frameworks that are rapidly changing or less popular.

Instead of just providing raw text, this project chunks, cleans, and embeds the documentation into a portable vector store. The resulting "Knowledge Pack" can be used locally for offline-first Retrieval-Augmented Generation (RAG) applications, ensuring that AI-powered developer tools have access to accurate, versioned information.

## Core Features

-   **Automated Content Scraping**: Uses [Firecrawl](https://firecrawl.dev/) to efficiently scrape and convert web pages into clean Markdown.
-   **Intelligent Text Chunking**: Splits documents into semantic chunks based on headers for optimal context in RAG systems.
-   **Vector Embeddings**: Converts text chunks into vector embeddings using state-of-the-art `sentence-transformer` models.
-   **Persistent Vector Store**: Uses [ChromaDB](https://www.trychroma.com/) to create a portable, file-based vector database.
-   **Downloadable Knowledge Packs**: Packages the entire processed output into a single, versioned `.tar.gz` archive, perfect for distribution and offline use.
-   **Highly Configurable**: Add new documentation targets easily in the `config.toml` file.
-   **Modern Python Tooling**: Uses `pyproject.toml` with [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.
-   **Modular & Testable**: The pipeline is broken into distinct, independently runnable stages: `scrape`, `process`, and `package`.
-   **Cloud-Ready**: Includes a GCP Cloud Function entry point for easy, scheduled, serverless pipeline execution.

## Architecture Overview

The pipeline operates in decoupled stages for flexibility and efficient debugging.

1.  **Scrape**: A target URL is scraped, and the raw Markdown content for each page is downloaded.
2.  **Cache**: The scraped data is saved to a local cache (`.cache/`) as a JSON file. This prevents re-scraping and allows the processing stage to be run multiple times on a stable dataset.
3.  **Process**: The cached data is loaded. The pipeline then:
    a.  Splits documents into clean, semantic text chunks.
    b.  Generates vector embeddings for each chunk.
    c.  Saves the embeddings, source text, and metadata into a persistent ChromaDB vector store.
    d.  The output is a self-contained directory for the target (e.g., `repository/langchain/`).
4.  **Package**: This final step compresses the processed directory into a single, distributable `langchain-knowledge-pack.tar.gz` archive.

## Getting Started

Follow these instructions to set up and run the pipeline on your local machine.

### Prerequisites

-   Git
-   Python 3.9+
-   [uv](https://github.com/astral-sh/uv) (a fast Python package installer and resolver)

### 1. Clone the Repository

```bash
git clone git@github.com:PAndreew/docustore.git
cd docustore
```

  

### 2. Create and Activate a Virtual Environment

Using a virtual environment is crucial for isolating project dependencies. uv makes this simple.
```Bash
    
# Create a virtual environment named .venv
uv venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows (Command Prompt/PowerShell):
.venv\Scripts\activate
```
  

### 3. Install Dependencies

With the virtual environment active, install all project dependencies from pyproject.toml.
```Bash
    
# Install the project and all its dependencies
uv pip install -e .
```

  

### 4. Configure Your Environment

Create a file named .env in the project root to store your Firecrawl API key.
```Env
# .env
FIRECRAWL_API_KEY="your_firecrawl_api_key_here"
```
  

Pipeline Configuration (config.toml) - this file controls the pipeline. Define your scraping targets here.

```Toml
    # config.toml
    [pipeline]
    output_dir = "repository"
    cache_dir = ".cache"

    [vector_store]
    embedding_model_name = "all-MiniLM-L6-v2"

    [targets.langchain]
    url = "https://python.langchain.com/docs/get_started/introduction"
    page_limit = 200
```
      

## Running the Pipeline

The pipeline is controlled via main.py with four distinct commands.

### scrape

Scrapes a target website and saves the raw data to the local cache.
```Bash
    
# Scrape the 'langchain' target
python -m pipeline.main scrape langchain
```
  

### process

Loads cached data and generates a Knowledge Pack directory containing the vector store and source files.
```Bash

    
# Process the cached 'langchain' data
python -m pipeline.main process langchain
```
  

### package

Compresses a processed Knowledge Pack directory into a single .tar.gz file for distribution.
```Bash
    
# Create the downloadable archive for 'langchain'
python -m pipeline.main package langchain
```
  

### run

An all-in-one command that runs the scrape and process stages sequentially.
```Bash

    
# Run the entire pipeline for the 'langchain' target
python -m pipeline.main run langchain
```
  

## Using a Knowledge Pack

The primary output of this project is a portable Knowledge Pack that you can use to build powerful RAG applications.

    Download a pack (e.g., langchain-knowledge-pack.tar.gz) from the project's releases.

    Extract the archive: tar -xzf langchain-knowledge-pack.tar.gz

    Use it in your Python code:

```Python

import chromadb

# Point the ChromaDB client to the extracted database directory
db_path = "./langchain-knowledge-pack/db"
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="langchain")

# Ask a question!
results = collection.query(
    query_texts=["how do I use memory in an agent?"],
    n_results=5
)

# results will contain the most relevant text chunks
for doc in results['documents']:
    print(doc, "\n---\n")
```
  

## License & Acknowledgements

This project's source code is licensed under the MIT License. See the LICENSE file for details.
