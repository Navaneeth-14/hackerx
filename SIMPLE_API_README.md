# Simple Flask API for RAG System

A simplified Flask API server that provides localhost access to the main.py RAG system functionality.

## Features

- **Localhost Only**: Runs only on 127.0.0.1 (localhost)
- **Simple Authentication**: No authentication required for localhost use
- **Core Operations**: Document upload and query processing
- **GET/POST Operations**: Simple REST endpoints

## API Endpoints

### 1. Health Check
**GET** `http://127.0.0.1:5000/health`

Check if the server is running.

**Response:**
```json
{
  "status": "healthy",
  "rag_system_initialized": true
}
```

### 2. System Status
**GET** `http://127.0.0.1:5000/hackrx/status`

Check if the RAG system is ready.

**Response:**
```json
{
  "status": "ready",
  "message": "RAG system is ready"
}
```

### 3. Document Upload
**POST** `http://127.0.0.1:5000/hackrx/upload`

Upload and process a document.

**Form Data:**
- `file`: The document file to upload

**Supported File Types:**
- PDF (.pdf)
- Text (.txt)
- Word (.docx)
- HTML (.html, .htm)
- Email (.eml, .msg)
- CSV (.csv)
- JSON (.json)

**Response:**
```json
{
  "success": true,
  "message": "Document processed successfully",
  "chunks_processed": 45,
  "processing_time": 2.34,
  "filename": "document.pdf"
}
```

### 4. Query Processing
**POST** `http://127.0.0.1:5000/hackrx/run`

Process questions about uploaded documents.

**Request Body:**
```json
{
  "questions": [
    "What is covered under this policy?",
    "What is the maximum coverage amount?",
    "What documents are required for claims?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    {
      "question": "What is covered under this policy?",
      "answer": "Based on the policy document, the following are covered...",
      "decision": "COVERED",
      "confidence": 0.85,
      "processing_time": 1.23
    }
  ]
}
```

## Usage Examples

### Python Example

```python
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

# 1. Upload a document
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/hackrx/upload", files=files)
    print("Upload response:", response.json())

# 2. Process queries
questions = [
    "What is covered under this policy?",
    "What is the maximum coverage amount?"
]

payload = {"questions": questions}
response = requests.post(
    f"{BASE_URL}/hackrx/run",
    json=payload,
    headers={"Content-Type": "application/json"}
)

answers = response.json()["answers"]
for answer in answers:
    print(f"Q: {answer['question']}")
    print(f"A: {answer['answer']}")
    print(f"Decision: {answer['decision']}")
    print(f"Confidence: {answer['confidence']}")
    print("---")
```

### cURL Examples

**Health Check:**
```bash
curl http://127.0.0.1:5000/health
```

**System Status:**
```bash
curl http://127.0.0.1:5000/hackrx/status
```

**Upload Document:**
```bash
curl -X POST -F "file=@document.pdf" http://127.0.0.1:5000/hackrx/upload
```

**Process Queries:**
```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"questions": ["What is covered under this policy?"]}' \
     http://127.0.0.1:5000/hackrx/run
```

## Running the Server

1. **Start the server:**
   ```bash
   python app.py
   ```

2. **Test the API:**
   ```bash
   python test_api.py
   ```

## Error Responses

- `200`: Success
- `400`: Bad Request (missing parameters, invalid data)
- `500`: Internal Server Error

Error response format:
```json
{
  "error": "Error description"
}
```

## Notes

- **Localhost Only**: The server only accepts connections from localhost (127.0.0.1)
- **No Authentication**: No API keys or authentication required for localhost use
- **Simple Interface**: Focused on core document upload and query processing
- **Automatic Cleanup**: Uploaded files are automatically cleaned up after processing 