import logging
import re
import hashlib
import json
from typing import Any, Dict, List, Tuple, Set
from dataclasses import dataclass, field

import networkx as nx
import spacy
from .cleaner import EnhancedDataCleaner, chunk_document_by_headers

# Attempt to import Hugging Face and PyTorch libraries.
# This allows the rest of the pipeline to function even if they aren't installed,
# as long as triplet extraction is disabled in the config.
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Define dummy types if imports fail, so type hints don't break
    AutoModelForCausalLM = type('AutoModelForCausalLM', (), {})
    AutoTokenizer = type('AutoTokenizer', (), {})


@dataclass
class KnowledgeGraphData:
    """
    A container for the structured and unstructured data extracted by the pipeline.
    This decouples the graph from the source text, which is essential for hybrid
    Retrieval-Augmented Generation (RAG) systems.
    """
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    # The chunks dict stores the original text and metadata for each chunk.
    # Key: chunk_id (a stable hash), Value: { "text": "...", "source_url": "..." }
    chunks: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class TripletExtractor:
    """A wrapper for the sciphi/triplex model to handle loading, prompting, and parsing."""

    def __init__(self, model_name: str):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Please install 'transformers', 'torch', and 'accelerate' to use the TripletExtractor.")
        
        self.model_name = model_name
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"TripletExtractor initialized for model '{model_name}' on device '{self.device}'.")

    def _load_model(self):
        """
        Lazy-loads the model and tokenizer to avoid startup overhead and memory usage
        if the feature is not used.
        """
        if self.model is None or self.tokenizer is None:
            logging.info(f"Loading Triplex model '{self.model_name}'... This may take a moment.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
            ).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            logging.info("Triplex model loaded successfully.")

    def _parse_output(self, raw_output: str) -> List[Tuple[str, str, str]]:
        """
        Robustly parses the JSON output from the model's response string.
        It looks for a ```json ... ``` block and decodes its content.
        """
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        if not json_match:
            logging.warning("Could not find a JSON block in the model's output.")
            return []
        
        try:
            data = json.loads(json_match.group(1))
            # The model is expected to return a list of dictionaries under the "triples" key
            return [
                (triple['subject'], triple['predicate'], triple['object'])
                for triple in data.get('triples', [])
                if 'subject' in triple and 'predicate' in triple and 'object' in triple
            ]
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from model output: {e}\nRaw output part: {json_match.group(1)}")
            return []
        except (TypeError, KeyError) as e:
            logging.error(f"Unexpected JSON structure in model output: {e}\nParsed data: {data}")
            return []

    def extract(self, text: str, entity_types: List[str], predicates: List[str]) -> List[Tuple[str, str, str]]:
        """
        Formats the prompt with the specified ontology, runs the model inference,
        and parses the resulting list of triples.
        """
        self._load_model() # Ensure model is loaded before inference

        prompt_template = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.

**Entity Types:**
{entity_types}

**Predicates:**
{predicates}

**Text:**
{text}
"""
        message = prompt_template.format(
            entity_types=json.dumps({"entity_types": entity_types}),
            predicates=json.dumps({"predicates": predicates}),
            text=text
        )
        messages = [{'role': 'user', 'content': message}]
        
        # The model expects a specific chat template format for inference
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=False)
        
        # Decode the full output, skipping special tokens like <|endoftext|>
        raw_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # The model's response often includes the original prompt. We only care about the part after it.
        # The `apply_chat_template` adds roles like '<|user|>\n...' and '<|assistant|>\n...'.
        # We split on the assistant role to get only the model's generation.
        response_parts = raw_output.split('<|assistant|>')
        response_text = response_parts[1].strip() if len(response_parts) > 1 else ""

        return self._parse_output(response_text)

def create_knowledge_graph(
    documents: List[Dict[str, Any]],
    kg_config: Dict[str, Any],
    target_config: Dict[str, Any]
) -> KnowledgeGraphData:
    """
    Orchestrates the creation of a sophisticated knowledge graph using a hybrid
    rule-based and machine learning approach.
    """
    logging.info("Starting hybrid knowledge graph generation...")
    nlp = spacy.load("en_core_web_sm")
    kg_data = KnowledgeGraphData()
    cleaner = EnhancedDataCleaner()

    # --- Initialize the Triplet Extractor (if enabled in the config) ---
    triplet_config = kg_config.get("triplet_extraction", {})
    triplets_enabled = triplet_config.get("enabled", False)
    extractor = None
    if triplets_enabled:
        try:
            model_name = triplet_config.get("model_name", "sciphi/triplex")
            extractor = TripletExtractor(model_name)
        except Exception as e:
            logging.error(f"Failed to initialize TripletExtractor: {e}. Disabling triplet extraction for this run.")
            triplets_enabled = False

    # --- Pre-compile regex patterns for efficiency ---
    target_patterns = target_config.get("patterns", [])
    enabled_patterns = [
        {"label": p["label"], "pattern": re.compile(p["pattern"])}
        for p in target_patterns
    ]
    enabled_spacy_entities = kg_config.get("enabled_spacy_entities", [])

    for doc_data in documents:
        raw_markdown = doc_data.get("markdown", "")
        source_url = doc_data.get("metadata", {}).get("source_url", "unknown")
        if not raw_markdown:
            continue

        chunks = chunk_document_by_headers(raw_markdown)

        for i, chunk in enumerate(chunks):
            chunk_content = chunk["content"]
            if not chunk_content or len(chunk_content.split()) < 5: # Skip very short chunks
                continue

            # 1. Create a stable, unique ID for the chunk and store it
            chunk_identifier = f"{source_url}-{i}-{chunk['header']}"
            chunk_id = hashlib.sha256(chunk_identifier.encode()).hexdigest()
            kg_data.chunks[chunk_id] = {"text": chunk_content, "source_url": source_url, "header": chunk['header']}

            # 2. Perform Extractions
            rule_entities: Set[Tuple[str, str]] = set()
            spacy_entities: Set[Tuple[str, str]] = set()
            
            # --- Precision Pass (Rules) ---
            for item in enabled_patterns:
                for match in item["pattern"].finditer(chunk_content):
                    rule_entities.add((match.group(1).strip(), item["label"]))

            # --- Semantic Pass (ML) ---
            text_for_ml = cleaner._remove_html_and_markdown(chunk_content)
            
            # spaCy for basic Named Entity Recognition
            doc_nlp = nlp(text_for_ml)
            for ent in doc_nlp.ents:
                if ent.label_ in enabled_spacy_entities:
                    spacy_entities.add((ent.text.strip(), ent.label_))

            # Advanced LLM for triple extraction (if enabled and configured)
            triples: List[Tuple[str, str, str]] = []
            if triplets_enabled and extractor:
                entity_types = target_config.get("entity_types", [])
                predicates = target_config.get("predicates", [])
                if entity_types and predicates:
                    triples = extractor.extract(text_for_ml, entity_types, predicates)

            # 3. Merge, Enrich, and Build Graph
            all_entities = rule_entities.union(spacy_entities)
            # Add subjects and objects from the extracted triples as nodes
            for subj, _, obj in triples:
                all_entities.add((subj.strip(), "ENTITY")) # Use a generic label
                all_entities.add((obj.strip(), "ENTITY"))

            if not all_entities and not triples:
                continue

            # Add/update nodes in the graph, linking them to their source chunk
            for entity_text, label in all_entities:
                if kg_data.graph.has_node(entity_text):
                    node_data = kg_data.graph.nodes[entity_text]
                    if chunk_id not in node_data.get("chunk_ids", []):
                        node_data["chunk_ids"].append(chunk_id)
                else:
                    kg_data.graph.add_node(
                        entity_text,
                        label=label,
                        chunk_ids=[chunk_id]
                    )

            # Add edges (relationships) from the extracted triples
            for subj, pred, obj in triples:
                subj_clean, obj_clean = subj.strip(), obj.strip()
                if kg_data.graph.has_node(subj_clean) and kg_data.graph.has_node(obj_clean):
                    kg_data.graph.add_edge(subj_clean, obj_clean, label=pred.strip())

    node_count = kg_data.graph.number_of_nodes()
    edge_count = kg_data.graph.number_of_edges()
    logging.info(
        f"Graph generation complete. Found {node_count} nodes, {edge_count} edges, "
        f"and {len(kg_data.chunks)} text chunks."
    )
    return kg_data