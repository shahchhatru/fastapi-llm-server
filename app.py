from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aio_pika
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional, Any, List, Union
import uuid
import os

from sentenceTransformer import (
    store_sentences,
    find_similar_sentences,
    fetch_and_store_projects_from_postgres,
    collection,
    chroma_client,
    collection_name
)

# =====================================================
# WebSocket Manager
# =====================================================

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

# =====================================================
# Normalization Helpers (CRITICAL)
# =====================================================

def normalize_scalar(value: Any) -> Optional[str]:
    """Normalize a single value to lowercase string."""
    if value is None:
        return None
    return str(value).strip().lower()


def normalize_list(value: Union[str, int, List[Any], None]) -> Optional[List[str]]:
    """
    Normalize input to a list of lowercase strings.
    Handles: None, single values (str/int), or lists.
    """
    if value is None:
        return None

    if isinstance(value, list):
        return [normalize_scalar(v) for v in value if v is not None]

    return [normalize_scalar(value)]


def build_where_filter(
    ward=None,
    municipality=None,
    district=None,
    fiscal_year=None,
    province=None,
    language_type=None
):
    """
    Build ChromaDB where filter that handles both single values and lists.
    
    Key insight: ChromaDB metadata stores lists like ["1", "2", "3"].
    When filtering, we need to check if ANY of our search values exist in those lists.
    
    Use $contains to check if a value exists in a metadata list.
    Use $or when searching for multiple values (match any).
    """
    conditions = []

    # Handle wards - check if any ward value exists in the metadata wards list
    wards = normalize_list(ward)
    if wards:
        if len(wards) == 1:
            # Single ward: check if it's contained in the wards list
            conditions.append({"wards": {"$contains": wards[0]}})
        else:
            # Multiple wards: match if ANY ward is in the wards list
            ward_conditions = [{"wards": {"$contains": w}} for w in wards]
            conditions.append({"$or": ward_conditions})

    # Handle municipalities
    municipalities = normalize_list(municipality)
    if municipalities:
        if len(municipalities) == 1:
            conditions.append({"municipalities": {"$contains": municipalities[0]}})
        else:
            muni_conditions = [{"municipalities": {"$contains": m}} for m in municipalities]
            conditions.append({"$or": muni_conditions})

    # Handle districts
    districts = normalize_list(district)
    if districts:
        if len(districts) == 1:
            conditions.append({"districts": {"$contains": districts[0]}})
        else:
            district_conditions = [{"districts": {"$contains": d}} for d in districts]
            conditions.append({"$or": district_conditions})

    # Handle scalar fields (non-list metadata)
    if fiscal_year:
        conditions.append({"fiscal_year": str(fiscal_year)})

    if province:
        conditions.append({"province": normalize_scalar(province)})

    if language_type:
        conditions.append({"language": normalize_scalar(language_type)})

    # Return None if no filters, single condition if only one, $and for multiple
    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


# =====================================================
# Pydantic Models
# =====================================================

class StoreRequest(BaseModel):
    sentence: str
    project_id: int
    language_type: Optional[str] = "en"
    ward: Optional[Union[str, int, list]] = None
    municipality: Optional[Union[str, int, list]] = None
    district: Optional[Union[str, int, list]] = None
    fiscal_year: Optional[str] = None
    province: Optional[str] = None
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    language_type: Optional[str] = "en"
    ward: Optional[Union[str, int, list]] = None
    municipality: Optional[Union[str, int, list]] = None
    district: Optional[Union[str, int, list]] = None
    fiscal_year: Optional[str] = None
    province: Optional[str] = None


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None


class PostgresImportRequest(BaseModel):
    host: str
    database: str
    user: str
    password: str
    port: int = 5432
    offset: int = 0
    limit: int = 1000
    province: Optional[str] = None
    custom_query: Optional[str] = None


class DeleteByFilterRequest(BaseModel):
    fiscal_year: Optional[str] = None
    district: Optional[str] = None
    province: Optional[str] = None

# =====================================================
# RabbitMQ Globals
# =====================================================

rabbitmq_connection = None
rabbitmq_channel = None
QUEUE_NAME = "search_queue"

# =====================================================
# Lifespan
# =====================================================

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

# =====================================================
# Store Endpoint
# =====================================================

@app.post("/store", response_model=APIResponse)
async def store_sentence(request: StoreRequest):
    try:
        metadata = request.metadata or {}
        metadata["project_id"] = str(request.project_id)
        metadata["language"] = normalize_scalar(request.language_type)

        # Store wards, municipalities, and districts as lists for consistent querying
        if request.ward:
            metadata["wards"] = normalize_list(request.ward)

        if request.municipality:
            metadata["municipalities"] = normalize_list(request.municipality)

        if request.district:
            metadata["districts"] = normalize_list(request.district)

        # Store scalar metadata
        if request.fiscal_year:
            metadata["fiscal_year"] = str(request.fiscal_year)

        if request.province:
            metadata["province"] = normalize_scalar(request.province)

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

# =====================================================
# Direct Search
# =====================================================

@app.post("/search/direct", response_model=APIResponse)
async def search_similar_direct(request: SearchRequest):
    try:
        where_filter = build_where_filter(
            request.ward,
            request.municipality,
            request.district,
            request.fiscal_year,
            request.province,
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

# =====================================================
# RabbitMQ Producer
# =====================================================

@app.post("/search")
async def search_similar(request: SearchRequest):
    if not rabbitmq_channel:
        raise HTTPException(status_code=503, detail="RabbitMQ not connected")

    job_id = str(uuid.uuid4())

    await rabbitmq_channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(request.dict() | {"job_id": job_id}).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        ),
        routing_key=QUEUE_NAME,
    )

    return {"success": True, "job_id": job_id, "status": "queued"}

# =====================================================
# RabbitMQ Consumer
# =====================================================

async def consume_search_requests():
    queue = await rabbitmq_channel.declare_queue(QUEUE_NAME, durable=True)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                body = json.loads(message.body.decode())

                where_filter = build_where_filter(
                    body.get("ward"),
                    body.get("municipality"),
                    body.get("district"),
                    body.get("fiscal_year"),
                    body.get("province"),
                    body.get("language_type"),
                )

                results = find_similar_sentences(
                    query=body["query"],
                    language_type=body.get("language_type", "en"),
                    collection=collection,
                    n_results=body.get("n_results", 5),
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

                await manager.send_result(body["job_id"], {
                    "job_id": body["job_id"],
                    "status": "completed",
                    "filters": where_filter,
                    "count": len(similar_sentences),
                    "results": similar_sentences
                })

# =====================================================
# VectorDB Maintenance
# =====================================================
@app.delete("/vectordb/clear", response_model=APIResponse)
def clear_vector_db():
    global collection

    chroma_client.delete_collection(name=collection_name)

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"description": "Multilingual sentence embeddings"}
    )

    return APIResponse(
        success=True,
        message="VectorDB cleared successfully",
        data={"collection": collection_name}
    )

    

@app.delete("/vectordb/project/{project_id}", response_model=APIResponse)
def delete_project_embeddings(project_id: int):
    collection.delete(where={"project_id": str(project_id)})
    return APIResponse(
        success=True,
        message="Project embeddings deleted",
        data={"project_id": project_id}
    )


@app.post("/vectordb/delete/filter", response_model=APIResponse)
def delete_by_filter(request: DeleteByFilterRequest):
    where = {}

    if request.fiscal_year:
        where["fiscal_year"] = request.fiscal_year
    if request.district:
        # For deletion, use $contains to match any district in the list
        where["districts"] = {"$contains": normalize_scalar(request.district)}
    if request.province:
        where["province"] = normalize_scalar(request.province)

    if not where:
        raise HTTPException(status_code=400, detail="At least one filter required")

    collection.delete(where=where)
    return APIResponse(success=True, message="Deleted by filter", data=where)

# =====================================================
# PostgreSQL Import
# =====================================================

@app.post("/import/postgres", response_model=APIResponse)
async def import_from_postgres(request: PostgresImportRequest):
    result = await asyncio.to_thread(
        fetch_and_store_projects_from_postgres,
        host=request.host,
        database=request.database,
        user=request.user,
        password=request.password,
        port=request.port,
        offset=request.offset,
        limit=request.limit,
        province=request.province,
        query=request.custom_query,
    )

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])

    return APIResponse(
        success=True,
        message=result["message"],
        data=result
    )

# =====================================================
# Health
# =====================================================

@app.get("/", response_model=APIResponse)
def health_check():
    return APIResponse(
        success=True,
        message="FastAPI + RabbitMQ + ChromaDB + Robust Metadata",
        data={"collection": collection.name}
    )