# Flask API Server for main.py

A simple Flask API server that provides REST endpoints for the main.py RAG system without duplicating model loading or initialization.

## How It Works

The `app.py` server acts as a lightweight wrapper around `main.py`:
- **No Model Loading**: The server doesn't load models, parsers, or RAG components
- **Subprocess Calls**: Uses `subprocess` to call `main.py` with command line arguments
- **Localhost Only**: Runs only on 127.0.0.1 for security
- **Simple Interface**: Provides basic GET/POST endpoints

## API Endpoints

### 1. Health Check
**GET** `http://127.0.0.1:5000/health`

Check if the Flask server is running.

**Response:**
```json
{
  "status": "healthy",
  "message": "Flask server is running"
}
```

### 2. System Status
**GET** `http://127.0.0.1:5000/hackrx/status`

Check if main.py is ready and working.

**Response:**
```json
{
  "status": "ready",
  "message": "main.py is ready"
}
```

### 3. Document Upload
**POST** `http://127.0.0.1:5000/hackrx/upload`

Upload and process a document using main.py.

**Form Data:**
- `file`: The document file to upload

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

Process questions about uploaded documents using main.py.

**Request Body:**
```json
{
  "questions": [
    "What is covered under this policy?",
    "What is the maximum coverage amount?"
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

## How the Server Works

### Document Upload Process:
1. Flask receives uploaded file
2. Saves file to `uploads/` directory
3. Calls: `python main.py --upload /path/to/file`
4. Parses output from main.py
5. Returns JSON response

### Query Processing Process:
1. Flask receives JSON with questions
2. For each question:
   - Creates temporary file with question
   - Calls: `python main.py --query /path/to/question.txt`
   - Parses structured output from main.py
   - Extracts decision, confidence, justification
3. Returns JSON with all answers

### Status Check Process:
1. Calls: `python main.py --status`
2. Checks if main.py responds successfully
3. Returns status JSON

## Usage Examples

### Python Example

```python
import requests

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

## Command Line Interface

The main.py now supports command line arguments:

```bash
# Process a single query
python main.py --query question.txt

# Upload and process a document
python main.py --upload document.pdf

# Check system status
python main.py --status

# Interactive mode (default)
python main.py
```

## Advantages

1. **No Duplication**: Doesn't load models or initialize RAG system
2. **Lightweight**: Minimal memory footprint
3. **Simple**: Easy to understand and maintain
4. **Secure**: Localhost only
5. **Reliable**: Uses existing main.py functionality

## Error Handling

- **File Not Found**: Returns 400 if file doesn't exist
- **Unsupported Format**: Returns 400 for unsupported file types
- **Processing Errors**: Returns 500 with error details
- **Timeouts**: 60 seconds for queries, 120 seconds for uploads

## Notes

- **main.py Required**: The server requires main.py to be in the same directory
- **Python Path**: Assumes `python` command is available
- **File Cleanup**: Temporary files are automatically cleaned up
- **Upload Directory**: Creates `uploads/` directory if it doesn't exist 