# docustore - an Automated Knowledge Graph Builder

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

NOTE! This repo is under development and the pipeline is not fully functional, yet...

**docustore** is an automated pipeline for scraping software documentation websites and transforming their content into a structured, queryable knowledge graph (`.graphml`). This project uses a hybrid approach, combining rule-based extraction with advanced Machine Learning models to build a rich, semantic representation of technical knowledge.

The core of this project is to create a dataset of nodes (entities), edges (relationships/triples), and their associated source text chunks. This structure is ideal for advanced Retrieval-Augmented Generation (RAG) workflows, enabling more accurate and context-aware answers to complex questions about software frameworks.

## Core Features

-   **Automated Content Scraping**: Uses [Firecrawl](https://firecrawl.dev/) to efficiently scrape and convert web pages into clean Markdown.
-   **Hybrid Entity Extraction**: Combines fast, precise regex-based rules with flexible NER from [spaCy](https://spacy.io/) to identify key entities.
-   **Advanced Triplet Extraction**: Leverages the state-of-the-art `sciphi/triplex` LLM to extract knowledge triples (Subject-Predicate-Object), forming the relational backbone of the graph.
-   **Decoupled Knowledge Store**: Generates both a structured graph (`.graphml`) and a separate store of the unstructured source text (`.json`), linking them together for powerful GraphRAG applications.
-   **Highly Configurable**: Each documentation target can have its own custom rules, entity types, and predicates defined in a simple `config.toml` file.
-   **Modular & Testable**: The pipeline is broken into distinct, independently runnable stages: `scrape`, `process`, and `run`.
-   **Cloud-Ready**: Includes a GCP Cloud Function entry point for easy, scheduled, serverless deployment.

## Architecture Overview

The pipeline operates in distinct, decoupled stages, allowing for flexibility and efficient debugging.

1.  **Scrape**: A target URL is scraped, and the raw Markdown content for each page is downloaded.
2.  **Cache**: The scraped data is saved to a local cache (`.cache/`) as a JSON file. This prevents re-scraping and allows the processing stage to be run multiple times on a stable dataset.
3.  **Process**: The cached data is loaded. For each document:
    a.  The text is split into semantic chunks based on headers.
    b.  A hybrid extraction process identifies entities and relationships (triples) within each chunk.
    c.  A `networkx` graph is built, linking entities together. Crucially, each node in the graph stores a list of IDs pointing to the source chunks where it was found.
4.  **Save**: The final output is saved to the `repository/` directory as two files:
    *   `_graph.graphml`: The structured knowledge graph.
    *   `_chunks.json`: The unstructured source text chunks, indexed by ID.

## Getting Started

Follow these instructions to set up and run the pipeline on your local machine.

### Prerequisites

-   Git
-   Python 3.9+
-   [uv](https://github.com/astral-sh/uv) (for fast package management)
-   **An NVIDIA GPU with CUDA is highly recommended** for running the Triplex model. CPU execution will be extremely slow.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name> 
```
### 2.a Create and activate a virtual environment

This project uses uv for fast and reliable dependency management.
```
# Create a virtual environment named .venv in the current directory
uv venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows (Command Prompt/PowerShell):
.venv\Scripts\activate
```
### 2.b Install dependencies


```
# Install the project in editable mode, including all dependencies
uv pip install -e .
```
### 3. Configure Your Environment
Configuration is handled by two main files.

A. API Keys (.env file)
Create a file named .env in the project root to store your Firecrawl API key.

**.env**
```
FIRECRAWL_API_KEY="your_firecrawl_api_key_here"
```
B. Pipeline Configuration (config.toml)
This is the main control file for the pipeline. Modify config.toml to define your scraping targets and the "ontology" (entities and predicates) for the knowledge extraction.

**config.toml**
```
[triplet_extraction]
# Set to false to disable the slow, expensive ML step for quick tests
enabled = true 
model_name = "sciphi/triplex" 

[targets.langchain]
url = "https://python.langchain.com/docs/get_started/introduction"
page_limit = 1000
# Regex patterns for high-precision extraction
patterns = [
    { label = "CLASS",    pattern = "\\bclass\\s+([A-Za-z_][A-Za-z0-9_]+)" }
]
# The "ontology" for the Triplex model to understand this domain
entity_types = ["CLASS", "FUNCTION", "PARAMETER", "MODULE", "CONCEPT"]
predicates = ["HAS_METHOD", "RETURNS", "USES_PARAMETER", "IMPORTS"]
```

## Running the Pipeline

The pipeline is controlled via `main.py` with three distinct commands.

### `scrape`

Scrapes a target website and saves the raw data to the local cache.

```bash
# Scrape the 'langchain' target defined in config.toml
python -m pipeline.main scrape langchain

# Force re-scraping even if a cache file exists
python -m pipeline.main scrape langchain --force

# Scrape all targets defined in the config
python -m pipeline.main scrape all
```
### `process`
Loads data from the cache, builds the knowledge graph using the hybrid ML approach, and saves the output. This command will fail if the cache for the target does not exist.
```Bash
# Process the cached data for the 'langchain' target
python -m pipeline.main process langchain

# Process all existing caches
python -m pipeline.main process all
```
### `run`

An all-in-one command that runs the full pipeline sequentially: it will scrape if a cache is missing, and then immediately process the data.
```Bash
# Run the entire pipeline for the 'langchain' target
python main.py run langchain
```
### Cloud Deployment (Google Cloud Function)
The main.py script includes a gcp_entrypoint function designed for serverless deployment. You can deploy it with a command similar to the following:
```Bash
gcloud functions deploy knowledge-graph-pipeline \
  --runtime python311 \
  --trigger-topic your-trigger-topic \
  --entry-point gcp_entrypoint \
  --region your-gcp-region \
  --source . \
  --set-env-vars FIRECRAWL_API_KEY="your-secret-key"
  ```
You can then trigger the function by publishing a message to the Pub/Sub topic with a target attribute (e.g., target=langchain).
# License & Acknowledgements
## Project License
This project's source code is licensed under the MIT License. See the LICENSE file for details. You are free to use, modify, and distribute this code.
## Model License and Attribution
This project's knowledge extraction capabilities are powered by the sciphi/triplex model. The use of this model is subject to its own license, and we are grateful to the creators for making it available.
### Triplex: a SOTA LLM for knowledge graph construction

**Model License**

The model weights are licensed under CC BY-NC-SA 4.0.

**Usage Guidelines**

Non-Commercial Use: 

This license permits use for research, personal projects, and open-source, non-commercial applications like this one.

Commercial Use: 

For commercial usage, please refer to the licensing terms on the model card and contact the creators at founders@sciphi.ai.
Attribution: You must give appropriate credit to the model's authors. If you use or adapt this pipeline, please include the following citation in your work:
```Bibtex
@misc{pimpalgaonkar2024triplex,
  author = {Pimpalgaonkar, Shreyas and Tremelling, Nolan and Colegrove, Owen},
  title = {Triplex: a SOTA LLM for knowledge graph construction},
  year = {2024},
  url = {https://huggingface.co/sciphi/triplex}
}
```