## Instruction to create environment

```
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Install Dependencies

```
pip install -r requirements.txt
```

## Start RabbitMQ

```
# Run RabbitMQ with management plugin
docker run -d --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3-management

# Access management UI at: http://localhost:15672
# Default credentials: guest/guest
```
# FastAPI + ChromaDB + RabbitMQ + PostgreSQL Integration

A semantic search API that generates multilingual embeddings for sentences and provides similarity search capabilities. Now includes PostgreSQL integration to automatically import and index project data.

## Features

- 🔍 **Semantic Search**: Find similar sentences using multilingual embeddings
- 💾 **Vector Database**: Store and retrieve embeddings using ChromaDB
- 🐰 **Message Queue**: Asynchronous search processing with RabbitMQ
- 🗄️ **PostgreSQL Integration**: Import and index project data from PostgreSQL databases
- 🌐 **Multilingual Support**: Works with Nepali, Hindi, and English text

## Installation

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. Health Check
```bash
GET /
```

**Response:**
```json
{
  "success": true,
  "message": "FastAPI + RabbitMQ + ChromaDB + PostgreSQL Server",
  "data": {
    "rabbitmq_connected": true,
    "model_loaded": true,
    "collection_name": "multilingual_sentences"
  }
}
```

### 2. Store a Sentence
```bash
POST /store
Content-Type: application/json

{
  "sentence": "यो एक परीक्षण वाक्य हो",
  "project_id": 123,
  "metadata": {
    "category": "test",
    "language": "nepali"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Sentence stored successfully",
  "data": {
    "sentence": "यो एक परीक्षण वाक्य हो",
    "id": "uuid-here",
    "metadata": {
      "category": "test",
      "language": "nepali",
      "original_sentence": "यो एक परीक्षण वाक्य हो",
      "project_id": 123
    }
  }
}
```

### 3. Search Similar Sentences (Direct)
```bash
POST /search/direct
Content-Type: application/json

{
  "query": "test sentence",
  "n_results": 5
}
```

**Response:**
```json
{
  "success": true,
  "message": "Similar sentences found",
  "data": {
    "query": "test sentence",
    "count": 5,
    "results": [
      {
        "sentence": "यो एक परीक्षण वाक्य हो",
        "distance": 0.234,
        "metadata": {
          "project_id": "123",
          "language": "nepali"
        }
      }
    ]
  }
}
```

### 4. Search via RabbitMQ Queue
```bash
POST /search
Content-Type: application/json

{
  "query": "test sentence",
  "n_results": 5
}
```

**Response:**
```json
{
  "success": true,
  "message": "Search request published to queue",
  "data": {
    "query": "test sentence",
    "n_results": 5,
    "status": "queued"
  }
}
```

### 5. **Import Projects from PostgreSQL** 🆕

This endpoint connects to your PostgreSQL database, fetches project data, and automatically generates embeddings for similarity search.

```bash
POST /import/postgres
Content-Type: application/json

{
  "host": "localhost",
  "database": "mydb",
  "user": "postgres",
  "password": "your_password",
  "port": 5432,
  "table_name": "projects"
}
```

**With Custom Query:**
```bash
POST /import/postgres
Content-Type: application/json

{
  "host": "localhost",
  "database": "mydb",
  "user": "postgres",
  "password": "your_password",
  "port": 5432,
  "table_name": "projects",
  "custom_query": "SELECT project_id, project_name_in_english, project_name_in_nepali FROM projects WHERE status = 'active'"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully stored 150 projects",
  "data": {
    "projects_imported": 150,
    "stored_ids": ["uuid1", "uuid2", "uuid3", "uuid4", "uuid5"],
    "total_ids": 150,
    "sample_data": {
      "first_project": {
        "text": "Road Construction Project | सडक निर्माण परियोजना | ID: 1",
        "metadata": {
          "project_id": "1",
          "project_name_english": "Road Construction Project",
          "project_name_nepali": "सडक निर्माण परियोजना",
          "source": "postgresql",
          "concatenated_text": "Road Construction Project | सडक निर्माण परियोजना | ID: 1"
        }
      }
    }
  }
}
```

### 6. Get Collection Statistics
```bash
GET /stats
```

**Response:**
```json
{
  "success": true,
  "message": "Collection statistics",
  "data": {
    "collection_name": "multilingual_sentences",
    "total_documents": 150
  }
}
```

## PostgreSQL Integration Details

### Expected Database Schema

The function expects a table with the following columns:
- `project_id` (INT or similar)
- `project_name_in_english` (TEXT or VARCHAR)
- `project_name_in_nepali` (TEXT or VARCHAR)

### How It Works

1. **Connection**: Connects to your PostgreSQL database using provided credentials
2. **Query Execution**: Fetches project data (either using default query or custom query)
3. **Text Concatenation**: Combines English name, Nepali name, and project ID:
   - Format: `"{english_name} | {nepali_name} | ID: {project_id}"`
4. **Embedding Generation**: Creates multilingual embeddings using the sentence transformer model
5. **Storage**: Stores embeddings in ChromaDB with metadata for future similarity searches

### Metadata Stored

Each imported project stores the following metadata:
- `project_id`: Original project ID from database
- `project_name_english`: English name of the project
- `project_name_nepali`: Nepali name of the project
- `source`: Set to "postgresql" to identify the source
- `concatenated_text`: The full concatenated text that was embedded

### Example Usage with Python Requests

```python
import requests

# Import projects from PostgreSQL
response = requests.post(
    "http://localhost:8000/import/postgres",
    json={
        "host": "192.168.1.100",
        "database": "projects_db",
        "user": "admin",
        "password": "secure_password",
        "port": 5432,
        "table_name": "projects"
    }
)

print(response.json())

# Now search for similar projects
search_response = requests.post(
    "http://localhost:8000/search/direct",
    json={
        "query": "सडक निर्माण",  # "Road construction" in Nepali
        "n_results": 5
    }
)

print(search_response.json())
```

### Custom SQL Queries

You can provide custom SQL queries to filter or transform data:

```python
custom_query = """
SELECT 
    p.project_id,
    p.project_name_in_english,
    p.project_name_in_nepali
FROM projects p
INNER JOIN project_status ps ON p.project_id = ps.project_id
WHERE ps.status = 'active'
  AND p.created_at >= '2024-01-01'
ORDER BY p.created_at DESC
"""

response = requests.post(
    "http://localhost:8000/import/postgres",
    json={
        "host": "localhost",
        "database": "mydb",
        "user": "postgres",
        "password": "password",
        "custom_query": custom_query
    }
)
```

## Error Handling

The API provides detailed error messages for:
- Database connection failures
- Invalid credentials
- SQL query errors
- Embedding generation issues

Example error response:
```json
{
  "success": false,
  "message": "PostgreSQL error: connection refused",
  "data": null
}
```

## Environment Variables

You can set default RabbitMQ connection:
```bash
export RABBITMQ_URL="amqp://user:password@rabbitmq-host:5672/"
```

## Security Considerations

⚠️ **Important Security Notes:**

1. **Never commit database credentials** to version control
2. Use environment variables or secret management systems for sensitive data
3. Implement authentication on your FastAPI endpoints in production
4. Use SSL/TLS for database connections in production
5. Limit database user permissions to only what's needed (SELECT only)
6. Consider using connection pooling for production deployments

## Model Information

- **Model**: `amorfati/custom-hindi-emb-model`
- **Type**: Multilingual sentence transformer
- **Languages**: Supports Hindi, Nepali, English, and related languages

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ├─────────────────────────────────┐
       │                                 │
       ▼                                 ▼
┌─────────────┐                  ┌─────────────┐
│   FastAPI   │◄────────────────►│ PostgreSQL  │
└──────┬──────┘                  └─────────────┘
       │
       ├────────────────┬────────────────┐
       │                │                │
       ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  ChromaDB   │  │  RabbitMQ   │  │SentenceXformer│
└─────────────┘  └─────────────┘  └─────────────┘
```

## Future Enhancements

- [ ] Add batch import endpoints
- [ ] Implement periodic sync from PostgreSQL
- [ ] Add support for other databases (MySQL, MongoDB)
- [ ] Add authentication and authorization
- [ ] Implement rate limiting
- [ ] Add caching layer
- [ ] Support for incremental updates

## License

MIT

## Support

For issues or questions, please contact the development team.

### Build using docker compose

``` docker compose up --build -d```

🚀 Base URL (replace if needed)

If you’re running locally:
```
BASE_URL="http://localhost:7888"
```

🧠 1. Health check

Check if the server and RabbitMQ are connected:
```
curl -X GET "$BASE_URL/" -H "Content-Type: application/json"
```
💾 2. Store 5 example sentences
Sentence 1
```
curl -X POST "$BASE_URL/store" \
-H "Content-Type: application/json" \
-d '{"sentence": "The cat sat on the mat.", "project_id": 1}'
```
Sentence 2
```
curl -X POST "$BASE_URL/store" \
-H "Content-Type: application/json" \
-d '{"sentence": "Dogs are loyal animals.", "project_id": 1}'
```
Sentence 3
```
curl -X POST "$BASE_URL/store" \
-H "Content-Type: application/json" \
-d '{"sentence": "Artificial intelligence is transforming industries.", "project_id": 1}'
```
Sentence 4
```
curl -X POST "$BASE_URL/store" \
-H "Content-Type: application/json" \
-d '{"sentence": "Machine learning is a subset of artificial intelligence.", "project_id": 1}'
```
Sentence 5
```
curl -X POST "$BASE_URL/store" \
-H "Content-Type: application/json" \
-d '{"sentence": "Cats are independent and curious animals.", "project_id": 1}'
```
🔍 3. Search for similar sentences (directly)

Let’s search for sentences related to “AI is changing the world”:
```
curl -X POST "$BASE_URL/search/direct" \
-H "Content-Type: application/json" \
-d '{"query": "AI is changing the world", "n_results": 3}'
```

This will immediately return the top 3 most similar sentences from ChromaDB.







