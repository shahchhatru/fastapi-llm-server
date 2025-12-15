from sentence_transformers import SentenceTransformer, util
from chromadb.config import Settings
import numpy as np
from typing import List, Tuple, Optional
import uuid
import chromadb
import psycopg2
from psycopg2 import pool
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the multilingual model
#model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = SentenceTransformer("syubraj/sentence_similarity_nepali_v2")


def compare_sentences(sentence1: str, sentence2: str,
                      model: SentenceTransformer) -> float:
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
    metadata={"description": "Multilingual sentence embeddings"})


def store_sentences(sentences: List[str],
                    model: SentenceTransformer,
                    collection,
                    metadata_list: List[dict] = None):
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

    # Normalize embeddings to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Generate unique IDs
    ids = [str(uuid.uuid4()) for _ in range(len(sentences))]

    # Prepare metadata
    if metadata_list is None:
        metadata_list = [{"index": i} for i in range(len(sentences))]

    # Add to collection
    collection.add(embeddings=normalized_embeddings.tolist(),
                   documents=sentences,
                   ids=ids,
                   metadatas=metadata_list)

    return ids


def find_similar_sentences(query: str,
                           model: SentenceTransformer,
                           collection,
                           n_results: int = 5):
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
    results = collection.query(query_embeddings=query_embedding.tolist(),
                               n_results=n_results)

    return results


def fetch_and_store_projects_from_postgres(
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5432,
        table_name: str = "projects",
        province: Optional[str] = "local",
        query: Optional[str] = None) -> dict:
    """
    Connect to PostgreSQL database, fetch project data, generate embeddings, and store in ChromaDB
    
    Args:
        host: PostgreSQL host address
        database: Database name
        user: Database user
        password: Database password
        port: Database port (default: 5432)
        table_name: Name of the table containing projects (default: 'projects')
        query: Optional custom SQL query. If not provided, uses default query
    
    Returns:
        Dictionary with success status, count of stored projects, and details
    """
    connection = None
    cursor = None

    try:
        # Establish connection to PostgreSQL
        logger.info(f"Connecting to PostgreSQL at {host}:{port}/{database}")
        connection = psycopg2.connect(host=host,
                                      database=database,
                                      user=user,
                                      password=password,
                                      port=port)
        cursor = connection.cursor()
        logger.info("✓ Successfully connected to PostgreSQL")

        # Default query if not provided
        if query is None:
            query = f"""
            SELECT 
                id,
                project_name_in_english,
                project_name_in_nepali
            FROM {table_name}
            WHERE project_name_in_english IS NOT NULL 
               OR project_name_in_nepali IS NOT NULL
            """

        # Execute query
        logger.info(f"Executing query: {query}")
        cursor.execute(query)

        # Fetch all results
        rows = cursor.fetchall()
        logger.info(f"✓ Fetched {len(rows)} projects from database")

        if len(rows) == 0:
            return {
                "success": True,
                "message": "No projects found in database",
                "count": 0,
                "stored_ids": []
            }

        # Process rows and create concatenated sentences
        sentences_to_store = []
        metadata_list = []

        for row in rows:
            project_id = row[0]
            name_english = row[1] if row[1] else ""
            name_nepali = row[2] if row[2] else ""

            # Concatenate project names with separator
            # Format: "English Name | Nepali Name | ID: project_id"
            concatenated_text = f"{name_english}".strip()
            nepali_text = f"{name_nepali}".strip()
            # Clean up multiple spaces and separators
            concatenated_text = " ".join(concatenated_text.split())
            nepali_text_concatenated = " ".join(nepali_text.split())
            sentences_to_store.append(concatenated_text)
            sentences_to_store.append(nepali_text_concatenated)
            # Prepare metadata for each project
            metadata = {
                "project_id":
                str(project_id) if str(project_id) else "",
                "source":
                "postgresql",
                "concatenated_text":
                concatenated_text if str(concatenated_text) else "",
                "province":
                province if province else "",
                "language":
                "en"
            }
            metadata_ne = {
                "project_id":
                str(project_id) if str(project_id) else "",
                "source":
                "postgresql",
                "concatenated_text":
                nepali_text_concatenated
                if str(nepali_text_concatenated) else "",
                "province":
                province if province else "",
                "language":
                "ne"
            }
            metadata_list.append(metadata)
            metadata_list.append(metadata_ne)

        # Generate embeddings and store in ChromaDB
        logger.info(
            f"Generating embeddings for {len(sentences_to_store)} projects...")
        logger.info(f"{sentences_to_store}")
        logger.info(f"{metadata_list}")
        stored_ids = store_sentences(sentences=sentences_to_store,
                                     model=model,
                                     collection=collection,
                                     metadata_list=metadata_list)

        logger.info(
            f"✓ Successfully stored {len(stored_ids)} projects in ChromaDB")

        return {
            "success": True,
            "message": f"Successfully stored {len(stored_ids)} projects",
            "count": len(stored_ids),
            "stored_ids": stored_ids,
            "sample_data": {
                "first_project": {
                    "text":
                    sentences_to_store[0] if sentences_to_store else None,
                    "metadata": metadata_list[0] if metadata_list else None
                }
            }
        }

    except psycopg2.Error as e:
        error_msg = f"PostgreSQL error: {str(e)}"
        logger.error(f"✗ {error_msg}")
        return {
            "success": False,
            "message": error_msg,
            "count": 0,
            "stored_ids": []
        }

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"✗ {error_msg}")
        return {
            "success": False,
            "message": error_msg,
            "count": 0,
            "stored_ids": []
        }

    finally:
        # Clean up database connections
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            logger.info("✓ PostgreSQL connection closed")
