# pipeline/vector_processor.py
import logging
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any

import chromadb
from .cleaner import EnhancedDataCleaner, chunk_document_by_headers

# You'll need a sentence-transformer model
# Add `sentence-transformers` to your pyproject.toml
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def process_and_embed(
    documents: List[Dict[str, Any]],
    output_dir: str,
    target_name: str,
    model_name: str = 'all-MiniLM-L6-v2'
):
    """
    Chunks documents, creates vector embeddings, and saves them to a
    persistent ChromaDB collection. It produces a self-contained "Knowledge Pack"
    directory containing the vector store, raw text chunks, and metadata.

    Args:
        documents: A list of scraped document data (dictionaries).
        output_dir: The root directory to save the output.
        target_name: The name of the target being processed (e.g., 'langchain').
        model_name: The name of the sentence-transformer model to use for embeddings.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("Please install `sentence-transformers` to use the vector processor.")

    logging.info(f"Starting vector processing for target '{target_name}'...")
    cleaner = EnhancedDataCleaner()
    
    # Initialize the embedding model. This will download it on the first run.
    logging.info(f"Loading embedding model '{model_name}'...")
    embedding_model = SentenceTransformer(model_name)
    
    # Define the base directory for this target's Knowledge Pack
    target_output_dir = os.path.join(output_dir, target_name)
    os.makedirs(target_output_dir, exist_ok=True)
    
    # 1. Setup persistent ChromaDB client in the target's output directory
    db_path = os.path.join(target_output_dir, "db")
    client = chromadb.PersistentClient(path=db_path)
    
    # Create or get the collection. This allows for incremental updates if needed.
    collection = client.get_or_create_collection(
        name=target_name,
        metadata={"embedding_model": model_name}
    )
    
    all_chunks = {}
    chunks_to_embed = []
    chunk_ids_to_embed = []
    metadatas_to_embed = []

    # --- Start Chunking and Processing ---
    for doc_data in documents:
        raw_markdown = doc_data.get("markdown", "")
        metadata = doc_data.get("metadata", {})
        source_url = metadata.get("source_url", "unknown")

        if not raw_markdown:
            continue

        # Split the document into chunks based on markdown headers
        chunks = chunk_document_by_headers(raw_markdown)

        for i, chunk in enumerate(chunks):
            chunk_content = chunk["content"]
            if not chunk_content:
                continue

            # Create a stable, unique ID for this chunk
            identifier = f"{source_url}-{i}"
            chunk_id = hashlib.sha256(identifier.encode()).hexdigest()

            # Clean the text specifically for the embedding process
            cleaned_text = cleaner.clean(chunk_content)
            if not cleaned_text:
                continue
            
            # Store the original, uncleaned text for full-context retrieval
            all_chunks[chunk_id] = {
                "text": chunk_content,
                "source_url": source_url,
                "header": chunk["header"]
            }
            
            # Accumulate data for batch embedding
            chunks_to_embed.append(cleaned_text)
            chunk_ids_to_embed.append(chunk_id)
            metadatas_to_embed.append({"source_url": source_url})

    # --- Batch Embedding and Storage ---
    if chunks_to_embed:
        logging.info(f"Embedding {len(chunks_to_embed)} chunks in batches...")
        
        # 2. Embed all collected chunks in a single, efficient batch operation
        embeddings = embedding_model.encode(
            chunks_to_embed, 
            show_progress_bar=True,
            normalize_embeddings=True # Important for cosine similarity
        )
        
        # 3. Upsert the data into the ChromaDB collection
        # 'upsert' will add new items and update existing ones with the same ID
        collection.upsert(
            ids=chunk_ids_to_embed,
            embeddings=embeddings.tolist(),
            documents=chunks_to_embed, # Store cleaned text for Chroma's keyword search
            metadatas=metadatas_to_embed
        )
        logging.info(f"Upserted {len(chunk_ids_to_embed)} chunks into ChromaDB collection '{target_name}'.")

    # --- Save the Final Artifacts for the Knowledge Pack ---
    # 4. Save the raw text chunks file
    chunks_path = os.path.join(target_output_dir, "chunks.json")
    try:
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved raw text chunks to {chunks_path}.")
    except IOError as e:
        logging.error(f"Failed to save chunks file: {e}")

    # 5. Save the metadata file
    metadata_path = os.path.join(target_output_dir, "metadata.json")
    pack_metadata = {
      "framework": target_name,
      "version": datetime.utcnow().strftime("%Y.%m.%d"),
      "documents_scraped": len(documents),
      "chunks_embedded": len(chunk_ids_to_embed),
      "embedding_model": model_name
    }
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(pack_metadata, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved knowledge pack metadata to {metadata_path}.")
    except IOError as e:
        logging.error(f"Failed to save metadata file: {e}")

    logging.info(f"Vector processing for target '{target_name}' complete.")