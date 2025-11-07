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







