# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

from .loader import load_knowledge_pack, get_embedding_model

# Configure logging
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Knowledge Pack Query API")

class QueryRequest(BaseModel):
    target: str
    query_text: str
    n_results: int = 5

class QueryResponse(BaseModel):
    documents: List[List[str]]
    metadatas: List[List[Dict[str, Any]]]
    distances: List[List[float]]

@app.post("/query", response_model=QueryResponse)
def query_knowledge_pack(request: QueryRequest):
    """
    Receives a query and a target, loads the corresponding knowledge pack (if not cached),
    and returns the results from the vector store.
    """
    try:
        logging.info(f"Received query for target: '{request.target}'")
        
        # 1. Load the ChromaDB collection for the target.
        # This function handles the caching logic (download from GCS if needed).
        collection = load_knowledge_pack(request.target)
        
        # 2. Load the embedding model (this is also cached in memory).
        embedding_model = get_embedding_model()
        
        # 3. Query the collection.
        results = collection.query(
            query_texts=[request.query_text],
            n_results=request.n_results,
            # We don't need to provide the embeddings here because ChromaDB can
            # use the embedding function associated with the collection,
            # but for clarity and control, we could embed here and pass `query_embeddings`.
        )
        
        # FastAPI will automatically serialize this Pydantic model to JSON
        return QueryResponse(
            documents=results.get('documents', []),
            metadatas=results.get('metadatas', []),
            distances=results.get('distances', [])
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail=f"Knowledge pack for target '{request.target}' not found. It may not have been built or has a different name."
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")