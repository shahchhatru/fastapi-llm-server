from sentence_transformers import SentenceTransformer, util
from chromadb.config import Settings
import numpy as np

import uuid
import chromadb
import psycopg2
from psycopg2 import pool
import logging

from typing import List, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Load Multiple Models
# =========================

MODEL_NE = SentenceTransformer("syubraj/sentence_similarity_nepali_v2")
MODEL_EN = SentenceTransformer("BAAI/bge-base-en-v1.5")
MODEL_MIX = SentenceTransformer("BAAI/bge-m3")

MODEL_MAP = {
    "ne": MODEL_NE,
    "en": MODEL_EN,
    "ne-en": MODEL_MIX
}


def get_model(language_type: str) -> SentenceTransformer:
    lang = (language_type or "ne-en").strip().lower()
    return MODEL_MAP.get(lang, MODEL_MIX)


# =========================
# ChromaDB Setup
# =========================

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "multilingual_sentences"

try:
    chroma_client.delete_collection(name=collection_name)
except Exception:
    pass

collection = chroma_client.create_collection(
    name=collection_name,
    metadata={"description": "Multilingual sentence embeddings"}
)


# =========================
# Helpers
# =========================

def normalize_metadata(md: dict) -> dict:
    cleaned = {}
    for k, v in md.items():
        if isinstance(v, str):
            cleaned[k] = v.strip().lower()
        else:
            cleaned[k] = v
    return cleaned


# =========================
# Store Sentences
# =========================

def store_sentences(sentences: List[str],
                    language_type: str,
                    collection,
                    metadata_list: List[dict] = None):

    model = get_model(language_type)

    embeddings = model.encode(sentences, convert_to_numpy=True)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    ids = [str(uuid.uuid4()) for _ in range(len(sentences))]

    if metadata_list is None:
        metadata_list = [{} for _ in range(len(sentences))]

    cleaned_metadata = []
    for md in metadata_list:
        md_clean = normalize_metadata(md)
        md_clean["language_type"] = (language_type or "ne-en").strip().lower()
        cleaned_metadata.append(md_clean)

    collection.add(
        embeddings=normalized_embeddings.tolist(),
        documents=sentences,
        ids=ids,
        metadatas=cleaned_metadata
    )

    return ids


# =========================
# Find Similar Sentences
# =========================

def find_similar_sentences(query: str,
                           language_type: str,
                           collection,
                           n_results: int = 5,
                           where: Optional[dict] = None):

    model = get_model(language_type)
    lang = (language_type or "ne-en").strip().lower()

    # BGE models need prefix for best retrieval
    if lang in ["en", "ne-en"]:
        query_text = f"Represent this sentence for searching relevant passages: {query}"
    else:
        query_text = query

    query_embedding = model.encode([query_text], convert_to_numpy=True)

    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    normalized_query = query_embedding / query_norm

    results = collection.query(
        query_embeddings=normalized_query.tolist(),
        n_results=n_results,
        where=where
    )

    return results
