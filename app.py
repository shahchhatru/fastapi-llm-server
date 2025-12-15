from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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
from sentenceTransformer import (store_sentences, find_similar_sentences,
                                 model, collection,
                                 fetch_and_store_projects_from_postgres)


class ConnectionManager:

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[job_id] = websocket

    def disconnect(self, job_id: str):
        self.active_connections.pop(job_id, None)

    async def send_result(self, job_id: str, message: dict):
        ws = self.active_connections.get(job_id)
        if ws:
            await ws.send_json(message)
            await ws.close()
            self.disconnect(job_id)


manager = ConnectionManager()


# Pydantic models
class StoreRequest(BaseModel):
    sentence: str
    project_id: int
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


class PostgresImportRequest(BaseModel):
    host: str
    database: str
    user: str
    password: str
    port: int = 5432
    table_name: str = "projects"
    province: Optional[str] = None
    custom_query: Optional[str] = None


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
        rabbitmq_connection = await aio_pika.connect_robust(
            os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq/"))
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=False,  # must be False to use wildcard origins
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)


# Background consumer for search requests
async def consume_search_requests():
    queue = await rabbitmq_channel.declare_queue(QUEUE_NAME, durable=True)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                body = json.loads(message.body.decode())

                job_id = body["job_id"]
                query = body["query"]
                n_results = body.get("n_results", 5)

                results = find_similar_sentences(query=query,
                                                 model=model,
                                                 collection=collection,
                                                 n_results=n_results)

                similar_sentences = []
                if results["documents"]:
                    for i, doc in enumerate(results["documents"][0]):
                        similar_sentences.append({
                            "sentence":
                            doc,
                            "distance":
                            results["distances"][0][i],
                            "metadata":
                            results["metadatas"][0][i],
                        })

                payload = {
                    "job_id": job_id,
                    "status": "completed",
                    "query": query,
                    "count": len(similar_sentences),
                    "results": similar_sentences
                }

                # 🔔 PUSH RESULT TO FRONTEND
                await manager.send_result(job_id, payload)


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
        ids = store_sentences(sentences=[request.sentence],
                              model=model,
                              collection=collection,
                              metadata_list=[metadata])

        return APIResponse(success=True,
                           message="Sentence stored successfully",
                           data={
                               "sentence": request.sentence,
                               "id": ids[0],
                               "metadata": metadata
                           })

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=APIResponse(
                                success=False,
                                message=f"Failed to store sentence: {str(e)}",
                                data=None).dict())


# Route 2: Publish search request to RabbitMQ
@app.post("/search")
async def search_similar(request: SearchRequest):
    if not rabbitmq_channel:
        raise HTTPException(status_code=503, detail="RabbitMQ not connected")

    job_id = str(uuid.uuid4())

    await rabbitmq_channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps({
                "job_id": job_id,
                "query": request.query,
                "n_results": request.n_results
            }).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=QUEUE_NAME,
    )

    return {"success": True, "job_id": job_id, "status": "queued"}


@app.websocket("/ws/search/{job_id}")
async def websocket_search(websocket: WebSocket, job_id: str):
    await manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        manager.disconnect(job_id)


# Route 3: Direct search without RabbitMQ
@app.post("/search/direct", response_model=APIResponse)
async def search_similar_direct(request: SearchRequest):
    """
    Directly search for similar sentences without using RabbitMQ
    """
    try:
        results = find_similar_sentences(query=request.query,
                                         model=model,
                                         collection=collection,
                                         n_results=request.n_results)

        # Process results
        similar_sentences = []
        if results['documents'] and len(results['documents']) > 0:
            for i, doc in enumerate(results['documents'][0]):
                similar_sentences.append({
                    "sentence":
                    doc,
                    "distance":
                    results['distances'][0][i]
                    if results['distances'] else None,
                    "metadata":
                    results['metadatas'][0][i]
                    if results['metadatas'] else None
                })

        return APIResponse(success=True,
                           message="Similar sentences found",
                           data={
                               "query": request.query,
                               "count": len(similar_sentences),
                               "results": similar_sentences
                           })

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=APIResponse(
                                success=False,
                                message=f"Search failed: {str(e)}",
                                data=None).dict())


# Route 4: Import projects from PostgreSQL
@app.post("/import/postgres", response_model=APIResponse)
async def import_from_postgres(request: PostgresImportRequest):
    """
    Connect to PostgreSQL database and import project data into ChromaDB
    
    This endpoint fetches project data from PostgreSQL, concatenates 
    project_name_in_english, project_name_in_nepali, and project_id,
    generates embeddings, and stores them in ChromaDB for similarity search.
    """
    try:
        # Call the import function (runs synchronously)
        result = await asyncio.to_thread(
            fetch_and_store_projects_from_postgres,
            host=request.host,
            database=request.database,
            user=request.user,
            password=request.password,
            port=request.port,
            table_name=request.table_name,
            province=request.province,
            query=request.custom_query)

        if result["success"]:
            return APIResponse(
                success=True,
                message=result["message"],
                data={
                    "projects_imported": result["count"],
                    "stored_ids":
                    result["stored_ids"][:5],  # Return first 5 IDs
                    "total_ids": len(result["stored_ids"]),
                    "sample_data": result.get("sample_data")
                })
        else:
            raise HTTPException(status_code=500,
                                detail=APIResponse(success=False,
                                                   message=result["message"],
                                                   data=None).dict())

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=APIResponse(
                success=False,
                message=f"Failed to import from PostgreSQL: {str(e)}",
                data=None).dict())


# Health check
@app.get("/", response_model=APIResponse)
def health_check():
    return APIResponse(
        success=True,
        message="FastAPI + RabbitMQ + ChromaDB + PostgreSQL Server",
        data={
            "rabbitmq_connected": rabbitmq_channel is not None,
            "model_loaded": model is not None,
            "collection_name": collection.name
        })


# Get collection stats
@app.get("/stats", response_model=APIResponse)
def get_stats():
    try:
        count = collection.count()
        return APIResponse(success=True,
                           message="Collection statistics",
                           data={
                               "collection_name": collection.name,
                               "total_documents": count
                           })
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=APIResponse(
                                success=False,
                                message=f"Failed to get stats: {str(e)}",
                                data=None).dict())
