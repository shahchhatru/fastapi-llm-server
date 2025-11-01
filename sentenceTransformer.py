from sentence_transformers import SentenceTransformer, util
from chromadb.config import Settings
import numpy as np
from typing import List, Tuple
import uuid
import chromadb

# Load the multilingual model
model = SentenceTransformer('amorfati/custom-hindi-emb-model')

def compare_sentences(sentence1: str, sentence2: str, model: SentenceTransformer) -> float:
    """
    Compare similarity between two sentences

    Args:
        sentence1: First sentence (can be Nepali or English)
        sentence2: Second sentence (can be Nepali or English)
        model: Loaded SentenceTransformer model

    Returns:
        Cosine similarity score (0-1)
    """
    embeddings = model.encode([sentence1, sentence2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity



# Initialize ChromaDB client with persistent storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get a collection
collection_name = "multilingual_sentences"

# Delete collection if it exists (for fresh start)
try:
    chroma_client.delete_collection(name=collection_name)
    print(f"Deleted existing collection '{collection_name}'")
except:
    pass

# Create new collection
collection = chroma_client.create_collection(
    name=collection_name,
    metadata={"description": "Multilingual sentence embeddings"}
)

#print(f"Collection '{collection_name}' created successfully")

def store_sentences(sentences: List[str], model: SentenceTransformer,
                   collection, metadata_list: List[dict] = None):
    """
    Store sentences and their embeddings in ChromaDB

    Args:
        sentences: List of sentences to store
        model: SentenceTransformer model
        collection: ChromaDB collection
        metadata_list: Optional metadata for each sentence
    """
    # Generate embeddings
    embeddings = model.encode(sentences, convert_to_numpy=True)

    # Generate unique IDs
    ids = [str(uuid.uuid4()) for _ in range(len(sentences))]

    # Prepare metadata
    if metadata_list is None:
        metadata_list = [{"index": i} for i in range(len(sentences))]

    # Add to collection
    collection.add(
        embeddings=embeddings.tolist(),
        documents=sentences,
        ids=ids,
        metadatas=metadata_list
    )

    #print(f"Stored {len(sentences)} sentences in the database")
    return ids


def find_similar_sentences(query: str, model: SentenceTransformer,
                          collection, n_results: int = 5):
    """
    Find similar sentences from the database

    Args:
        query: Query sentence
        model: SentenceTransformer model
        collection: ChromaDB collection
        n_results: Number of results to return

    Returns:
        Query results with similar sentences
    """
    # Generate query embedding
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Query the database
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )

    return results

    



