from sentence_transformers import SentenceTransformer, util
from chromadb.config import Settings
import numpy as np

import uuid
import chromadb
import psycopg2
from psycopg2 import pool
import logging

from typing import List, Optional , Any
import json

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





def normalize_list(value: Any) -> Optional[List[str]]:
    """
    Normalize Postgres JSON / scalar into List[str]
    """
    if value is None:
        return None

    # psycopg2 may return dict/list for JSONB
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if v is not None]

    if isinstance(value, dict):
        return [str(v).strip().lower() for v in value.values()]

    return [str(value).strip().lower()]



def fetch_and_store_projects_from_postgres(
    host: str,
    database: str,
    user: str,
    password: str,
    port: int = 5432,
    offset: int = 0,
    limit: int = 1000,
    province: Optional[str] = "karnali",
    query: Optional[str] = None
) -> dict:
    """
    Fetch English + Nepali project names and store in ChromaDB
    with shared metadata
    """

    connection = None
    cursor = None

    try:
        logger.info(f"Connecting to PostgreSQL at {host}:{port}/{database}")
        connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        cursor = connection.cursor()
        logger.info("✓ Successfully connected to PostgreSQL")

        if query is None:
            query = f"""
            SELECT 
                g.id,
                g.fiscal_year,
                g.project_name_in_english,
                g.project_name_in_nepali,
                pd.general_information->'location'->'district'       AS districts,
                pd.general_information->'location'->'municipalities' AS municipalities,
                pd.general_information->'location'->'wards'          AS wards
            FROM project_details pd
            INNER JOIN gates g ON pd.gate_id = g.id
            WHERE g.project_name_in_english IS NOT NULL
               OR g.project_name_in_nepali IS NOT NULL
            LIMIT {limit} OFFSET {offset}
            """

        logger.info("Executing query")
        cursor.execute(query)
        rows = cursor.fetchall()
        logger.info(f"✓ Fetched {len(rows)} projects from database")

        if not rows:
            return {
                "success": True,
                "message": "No projects found in database",
                "count": 0,
                "stored_ids": []
            }

        sentences_en = []
        metadata_en = []

        sentences_ne = []
        metadata_ne = []

        for row in rows:
            (
                project_id,
                fiscal_year,
                name_english,
                name_nepali,
                districts,
                municipalities,
                wards
            ) = row

            base_metadata = {
                "project_id": str(project_id),
                "fiscal_year": str(fiscal_year) if fiscal_year else "",
                "province": province.lower() if province else "",
                "districts": normalize_list(districts) if districts else "",
                "municipalities": normalize_list(municipalities) if municipalities else "",
                "wards": normalize_list(wards) if wards else "",
                "source": "postgresql",
            }

            # --------------------
            # English
            # --------------------
            if name_english:
                english_text = " ".join(name_english.strip().split())
                if english_text:
                    sentences_en.append(english_text)
                    metadata_en.append({
                        **base_metadata,
                        "language": "en",
                        "project_name_in_english": english_text
                    })

            # --------------------
            # Nepali
            # --------------------
            if name_nepali:
                nepali_text = " ".join(name_nepali.strip().split())
                if nepali_text:
                    sentences_ne.append(nepali_text)
                    metadata_ne.append({
                        **base_metadata,
                        "language": "ne",
                        "project_name_in_nepali": nepali_text
                    })

        stored_ids = []

        if sentences_en:
            logger.info(f"Storing {len(sentences_en)} English embeddings")
            stored_ids.extend(
                store_sentences(
                    sentences=sentences_en,
                    language_type="en",
                    collection=collection,
                    metadata_list=metadata_en
                )
            )

        if sentences_ne:
            logger.info(f"Storing {len(sentences_ne)} Nepali embeddings")
            stored_ids.extend(
                store_sentences(
                    sentences=sentences_ne,
                    language_type="ne",
                    collection=collection,
                    metadata_list=metadata_ne
                )
            )

        logger.info(f"✓ Successfully stored {len(stored_ids)} embeddings")

        return {
            "success": True,
            "message": f"Successfully stored {len(stored_ids)} embeddings",
            "count": len(stored_ids),
            "stored_ids": stored_ids
        }

    except psycopg2.Error as e:
        logger.error(f"✗ PostgreSQL error: {e}")
        return {
            "success": False,
            "message": str(e),
            "count": 0,
            "stored_ids": []
        }

    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return {
            "success": False,
            "message": str(e),
            "count": 0,
            "stored_ids": []
        }

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            logger.info("✓ PostgreSQL connection closed")
