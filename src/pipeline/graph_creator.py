# pipeline/graph_creator.py
import logging
import re
from typing import Any, Dict, List

import networkx as nx
import spacy
from .cleaner import EnhancedDataCleaner, chunk_document_by_headers

def create_knowledge_graph(
    documents: List[Dict[str, Any]],
    technical_entities: List[str],
    enabled_spacy_entities: List[str],
    associate_descriptions: bool,
) -> nx.DiGraph:
    """Creates a technical knowledge graph from cleaned document chunks."""
    logging.info("Starting technical knowledge graph generation...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logging.error("Spacy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
        raise

    G = nx.DiGraph()
    cleaner = EnhancedDataCleaner()
    patterns = {
        "CLASS": re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]+)"),
        "FUNCTION": re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]+)"),
    }
    enabled_patterns = {k: v for k, v in patterns.items() if k in technical_entities}

    for i, doc_data in enumerate(documents):
        raw_markdown = doc_data.get("markdown", "")
        if not raw_markdown:
            continue

        if (i + 1) % 50 == 0:
            logging.info("Processing document %d/%d...", i + 1, len(documents))

        chunks = chunk_document_by_headers(raw_markdown)
        for chunk in chunks:
            cleaned_chunk_content = cleaner.clean(chunk["content"])
            if not cleaned_chunk_content:
                continue

            found_entities = set()
            # Rule-based extraction
            for label, pattern in enabled_patterns.items():
                for match in pattern.finditer(chunk['content']): # Use original content for regex
                    found_entities.add((match.group(1), label))

            # spaCy-based extraction on cleaned content
            if enabled_spacy_entities:
                doc_nlp = nlp(cleaned_chunk_content)
                for ent in doc_nlp.ents:
                    if ent.label_ in enabled_spacy_entities:
                        found_entities.add((ent.text, ent.label_))

            for entity_text, label in found_entities:
                if not G.has_node(entity_text):
                    node_attrs = {"label": label}
                    if associate_descriptions:
                        node_attrs["description"] = cleaned_chunk_content
                    G.add_node(entity_text, **node_attrs)

    logging.info("Graph generation complete. Found %d nodes.", G.number_of_nodes())
    return G