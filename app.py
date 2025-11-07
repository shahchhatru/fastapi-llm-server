from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import aio_pika
import asyncio
import json
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from typing import Optional
import uuid
import os

# Import functions from sentenceTransformers
from sentenceTransformer import store_sentences, find_similar_sentences, model, collection

# Pydantic models
class StoreRequest(BaseModel):
    sentence: str
    project_id: int
    metadata: Optional[dict] = None

class SearchRequest(BaseModel):
    query: str
    n_results: int = 5

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

# Global variables
rabbitmq_connection = None
rabbitmq_channel = None
QUEUE_NAME = "search_queue"

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to RabbitMQ
    global rabbitmq_connection, rabbitmq_channel
    
    try:
       rabbitmq_connection = await aio_pika.connect_robust(os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq/"))
       
       rabbitmq_channel = await rabbitmq_connection.channel()
        
        # Declare queues
       await rabbitmq_channel.declare_queue(QUEUE_NAME, durable=True)
        
       print("✓ Connected to RabbitMQ")
        
        # Start background consumer
       asyncio.create_task(consume_search_requests())
        
    except Exception as e:
        print(f"✗ Failed to connect to RabbitMQ: {e}")
    
    yield
    
    # Shutdown
    if rabbitmq_connection:
        await rabbitmq_connection.close()
        print("✓ Closed RabbitMQ connection")

# Create FastAPI instance
app = FastAPI(lifespan=lifespan)

# Background consumer for search requests
async def consume_search_requests():
    """Consume messages from RabbitMQ and perform similarity search"""
    global rabbitmq_channel
    
    if not rabbitmq_channel:
        return
    
    queue = await rabbitmq_channel.declare_queue(QUEUE_NAME, durable=True)
    
    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                try:
                    body = json.loads(message.body.decode())
                    query = body.get("query")
                    n_results = body.get("n_results", 5)
                    
                    print(f"📨 Received search request: {query}")
                    
                    # Perform similarity search
                    results = find_similar_sentences(
                        query=query,
                        model=model,
                        collection=collection,
                        n_results=n_results
                    )
                    
                    # Process results
                    similar_sentences = []
                    if results['documents'] and len(results['documents']) > 0:
                        for i, doc in enumerate(results['documents'][0]):
                            similar_sentences.append({
                                "sentence": doc,
                                "distance": results['distances'][0][i] if results['distances'] else None,
                                "metadata": results['metadatas'][0][i] if results['metadatas'] else None
                            })
                    
                    output = {
                        "query": query,
                        "results": similar_sentences
                    }
                    
                    print(f"✓ Found {len(similar_sentences)} similar sentences")
                    print(f"📤 Results: {json.dumps(output, indent=2)}")
                    
                except Exception as e:
                    print(f"✗ Error processing message: {str(e)}")

# Route 1: Store sentence embedding in ChromaDB
@app.post("/store", response_model=APIResponse)
async def store_sentence(request: StoreRequest):
    """
    Generate embedding for a sentence and store it in ChromaDB
    """
    try:
        # Prepare metadata
        metadata = request.metadata or {}
        metadata["original_sentence"] = request.sentence
        metadata["project_id"] = request.project_id
        
        # Store in ChromaDB
        ids = store_sentences(
            sentences=[request.sentence],
            model=model,
            collection=collection,
            metadata_list=[metadata]
        )
        
        return APIResponse(
            success=True,
            message="Sentence stored successfully",
            data={
                "sentence": request.sentence,
                "id": ids[0],
                "metadata": metadata
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=APIResponse(
                success=False,
                message=f"Failed to store sentence: {str(e)}",
                data=None
            ).dict()
        )

# Route 2: Publish search request to RabbitMQ
@app.post("/search", response_model=APIResponse)
async def search_similar(request: SearchRequest):
    """
    Publish a search request to RabbitMQ queue
    The consumer will process it and find similar sentences
    """
    global rabbitmq_channel
    
    if not rabbitmq_channel:
        raise HTTPException(
            status_code=503,
            detail=APIResponse(
                success=False,
                message="RabbitMQ not connected",
                data=None
            ).dict()
        )
    
    try:
        # Publish search request to queue
        message_body = {
            "query": request.query,
            "n_results": request.n_results
        }
        
        await rabbitmq_channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message_body).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=QUEUE_NAME,
        )
        
        return APIResponse(
            success=True,
            message="Search request published to queue",
            data={
                "query": request.query,
                "n_results": request.n_results,
                "status": "success"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=APIResponse(
                success=False,
                message=f"Failed to publish search request: {str(e)}",
                data=None
            ).dict()
        )

# Alternative: Direct search without RabbitMQ
@app.post("/search/direct", response_model=APIResponse)
async def search_similar_direct(request: SearchRequest):
    """
    Directly search for similar sentences without using RabbitMQ
    """
    try:
        results = find_similar_sentences(
            query=request.query,
            model=model,
            collection=collection,
            n_results=request.n_results
        )
        
        # Process results
        similar_sentences = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                similar_sentences.append({
                    "sentence": doc,
                    "distance": results['distances'][0][i] if results['distances'] else None,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else None
                })
        
        return APIResponse(
            success=True,
            message="Similar sentences found",
            data={
                "query": request.query,
                "count": len(similar_sentences),
                "results": similar_sentences
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=APIResponse(
                success=False,
                message=f"Search failed: {str(e)}",
                data=None
            ).dict()
        )

# Health check
@app.get("/", response_model=APIResponse)
def health_check():
    return APIResponse(
        success=True,
        message="FastAPI + RabbitMQ + ChromaDB Server",
        data={
            "rabbitmq_connected": rabbitmq_channel is not None,
            "model_loaded": model is not None,
            "collection_name": collection.name
        }
    )

# Get collection stats
@app.get("/stats", response_model=APIResponse)
def get_stats():
    try:
        count = collection.count()
        return APIResponse(
            success=True,
            message="Collection statistics",
            data={
                "collection_name": collection.name,
                "total_documents": count
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=APIResponse(
                success=False,
                message=f"Failed to get stats: {str(e)}",
                data=None
            ).dict()
        )