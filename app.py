from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aio_pika
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional
import uuid
import os

from sentenceTransformer import (
    store_sentences,
    find_similar_sentences,
    collection
)

# =========================
# WebSocket Manager
# =========================

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

# =========================
# Helpers
# =========================

def normalize_str(value: Optional[str]) -> Optional[str]:
    if value:
        return value.strip().lower()
    return None


def build_where_filter(ward, municipality, language_type):
    where = {}

    if ward:
        where["ward"] = normalize_str(ward)

    if municipality:
        where["municipality"] = normalize_str(municipality)

    # IMPORTANT: prevent cross-model embedding mismatch
    if language_type:
        where["language_type"] = normalize_str(language_type)

    return where if where else None


# =========================
# Pydantic Models
# =========================

class StoreRequest(BaseModel):
    sentence: str
    project_id: int
    language_type: Optional[str] = "ne-en"   # ne | en | ne-en
    ward: Optional[str] = None
    municipality: Optional[str] = None
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    language_type: Optional[str] = "ne-en"
    ward: Optional[str] = None
    municipality: Optional[str] = None


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None


# =========================
# RabbitMQ Globals
# =========================

rabbitmq_connection = None
rabbitmq_channel = None
QUEUE_NAME = "search_queue"


# =========================
# Lifespan
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rabbitmq_connection, rabbitmq_channel

    try:
        rabbitmq_connection = await aio_pika.connect_robust(
            os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq/")
        )
        rabbitmq_channel = await rabbitmq_connection.channel()
        await rabbitmq_channel.declare_queue(QUEUE_NAME, durable=True)

        asyncio.create_task(consume_search_requests())
        print("✓ Connected to RabbitMQ")

    except Exception as e:
        print(f"✗ RabbitMQ connection failed: {e}")

    yield

    if rabbitmq_connection:
        await rabbitmq_connection.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Store Endpoint
# =========================

@app.post("/store", response_model=APIResponse)
async def store_sentence(request: StoreRequest):
    try:
        metadata = request.metadata or {}
        metadata["project_id"] = request.project_id

        if request.ward:
            metadata["ward"] = normalize_str(request.ward)

        if request.municipality:
            metadata["municipality"] = normalize_str(request.municipality)

        ids = store_sentences(
            sentences=[request.sentence],
            language_type=request.language_type,
            collection=collection,
            metadata_list=[metadata]
        )

        return APIResponse(
            success=True,
            message="Sentence stored successfully",
            data={"id": ids[0], "metadata": metadata}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Direct Search
# =========================

@app.post("/search/direct", response_model=APIResponse)
async def search_similar_direct(request: SearchRequest):
    try:
        where_filter = build_where_filter(
            request.ward,
            request.municipality,
            request.language_type
        )

        results = find_similar_sentences(
            query=request.query,
            language_type=request.language_type,
            collection=collection,
            n_results=request.n_results,
            where=where_filter
        )

        similar_sentences = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                similar_sentences.append({
                    "sentence": doc,
                    "distance": results["distances"][0][i],
                    "metadata": results["metadatas"][0][i],
                })

        return APIResponse(
            success=True,
            message="Similar sentences found",
            data={
                "filters": where_filter,
                "count": len(similar_sentences),
                "results": similar_sentences
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# RabbitMQ Producer
# =========================

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
                "n_results": request.n_results,
                "language_type": request.language_type,
                "ward": request.ward,
                "municipality": request.municipality
            }).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=QUEUE_NAME,
    )

    return {"success": True, "job_id": job_id, "status": "queued"}


# =========================
# RabbitMQ Consumer
# =========================

async def consume_search_requests():
    queue = await rabbitmq_channel.declare_queue(QUEUE_NAME, durable=True)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                body = json.loads(message.body.decode())

                job_id = body["job_id"]
                query = body["query"]
                n_results = body.get("n_results", 5)
                language_type = body.get("language_type", "ne-en")

                where_filter = build_where_filter(
                    body.get("ward"),
                    body.get("municipality"),
                    language_type
                )

                results = find_similar_sentences(
                    query=query,
                    language_type=language_type,
                    collection=collection,
                    n_results=n_results,
                    where=where_filter
                )

                similar_sentences = []
                if results["documents"]:
                    for i, doc in enumerate(results["documents"][0]):
                        similar_sentences.append({
                            "sentence": doc,
                            "distance": results["distances"][0][i],
                            "metadata": results["metadatas"][0][i],
                        })

                payload = {
                    "job_id": job_id,
                    "status": "completed",
                    "filters": where_filter,
                    "count": len(similar_sentences),
                    "results": similar_sentences
                }

                await manager.send_result(job_id, payload)


# =========================
# WebSocket
# =========================

@app.websocket("/ws/search/{job_id}")
async def websocket_search(websocket: WebSocket, job_id: str):
    await manager.connect(job_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(job_id)


# =========================
# Health + Stats
# =========================

@app.get("/", response_model=APIResponse)
def health_check():
    return APIResponse(
        success=True,
        message="FastAPI + RabbitMQ + ChromaDB (Multi-Model + Geo + Language)",
        data={"collection": collection.name}
    )


@app.get("/stats", response_model=APIResponse)
def get_stats():
    try:
        count = collection.count()
        return APIResponse(
            success=True,
            message="Collection statistics",
            data={"collection": collection.name, "total_documents": count}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
