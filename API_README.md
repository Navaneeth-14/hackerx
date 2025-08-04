# Flask API Server for Advanced RAG System

This Flask API server provides REST endpoints for the Advanced RAG System, allowing you to upload documents and process queries programmatically.

## Features

- **Document Upload**: Upload and process various document formats (PDF, TXT, DOCX, HTML, etc.)
- **Query Processing**: Ask questions about uploaded documents
- **OCR Support**: Process scanned documents with OCR
- **System Management**: Check status, validate components, clear data
- **Authentication**: Bearer token authentication
- **Error Handling**: Comprehensive error handling and fallback mechanisms

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the server is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "rag_system_initialized": true,
  "ocr_available": true
}
```

### 2. System Status
**GET** `/hackrx/status`

Get detailed system status and statistics.

**Headers:**
```
Authorization: Bearer your_api_key_here
```

**Response:**
```json
{
  "status": "ready",
  "statistics": {
    "vector_database": {
      "total_chunks": 150,
      "unique_sources": 3,
      "file_types": ["pdf", "txt"]
    },
    "audit_trail": {
      "total_entries": 25,
      "successful_queries": 20,
      "failed_queries": 5
    }
  },
  "ocr_available": true
}
```

### 3. Document Upload
**POST** `/hackrx/upload`

Upload and process a document.

**Headers:**
```
Authorization: Bearer your_api_key_here
```

**Form Data:**
- `file`: The document file to upload
- `use_ocr`: (optional) "true" or "false" to enable OCR for PDFs

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
**POST** `/hackrx/run`

Process questions about uploaded documents.

**Headers:**
```
Authorization: Bearer your_api_key_here
Content-Type: application/json
```

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
      "processing_time": 1.23,
      "amount": 50000.0,
      "waiting_period": "30 days",
      "relevant_clauses": ["Section 3.1", "Section 4.2"],
      "conditions": ["Must be hospitalized", "Pre-authorization required"],
      "exclusions": ["Cosmetic procedures", "Experimental treatments"],
      "required_documents": ["Hospital bills", "Medical reports"]
    }
  ]
}
```

### 5. System Validation
**GET** `/hackrx/validate`

Validate all system components.

**Headers:**
```
Authorization: Bearer your_api_key_here
```

**Response:**
```json
{
  "document_processor": true,
  "vector_database": true,
  "query_parser": true,
  "reasoning_engine": true,
  "all_valid": true,
  "errors": []
}
```

### 6. Clear System
**POST** `/hackrx/clear`

Clear all system data and reset the RAG system.

**Headers:**
```
Authorization: Bearer your_api_key_here
```

**Response:**
```json
{
  "success": true,
  "message": "System cleared successfully"
}
```

## Authentication

All endpoints (except `/health`) require Bearer token authentication:

```
Authorization: Bearer your_api_key_here
```

**Default API Key:** `your_api_key_here`

**Note:** Change this in production for security.

## Error Responses

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (missing parameters, invalid data)
- `401`: Unauthorized (missing or invalid Authorization header)
- `403`: Forbidden (invalid API key)
- `500`: Internal Server Error

Error response format:
```json
{
  "error": "Error description"
}
```

## Usage Examples

### Python Example

```python
import requests
import json

# Configuration
BASE_URL = "http://localhost:5000"
API_KEY = "your_api_key_here"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 1. Upload a document
with open("document.pdf", "rb") as f:
    files = {"file": f}
    data = {"use_ocr": "false"}
    upload_headers = {"Authorization": f"Bearer {API_KEY}"}
    
    response = requests.post(
        f"{BASE_URL}/hackrx/upload",
        files=files,
        data=data,
        headers=upload_headers
    )
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
    headers=HEADERS
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
curl http://localhost:5000/health
```

**System Status:**
```bash
curl -H "Authorization: Bearer your_api_key_here" \
     http://localhost:5000/hackrx/status
```

**Upload Document:**
```bash
curl -X POST \
     -H "Authorization: Bearer your_api_key_here" \
     -F "file=@document.pdf" \
     -F "use_ocr=false" \
     http://localhost:5000/hackrx/upload
```

**Process Queries:**
```bash
curl -X POST \
     -H "Authorization: Bearer your_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{"questions": ["What is covered under this policy?"]}' \
     http://localhost:5000/hackrx/run
```

## Running the Server

1. **Install Dependencies:**
   ```bash
   pip install flask requests
   ```

2. **Start the Server:**
   ```bash
   python app.py
   ```

3. **Test the API:**
   ```bash
   python test_api.py
   ```

## Configuration

### Environment Variables

You can set these environment variables:

- `FLASK_ENV`: Set to `production` for production deployment
- `API_KEY`: Override the default API key
- `PORT`: Override the default port (5000)

### Production Deployment

For production deployment:

1. Change the API key in `app.py`
2. Set `debug=False` in `app.run()`
3. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## Troubleshooting

### Common Issues

1. **RAG System Initialization Failed**
   - Check if all required dependencies are installed
   - Ensure model files are available
   - Check system memory and resources

2. **Document Upload Fails**
   - Verify file format is supported
   - Check file size limits
   - Ensure proper file permissions

3. **Query Processing Errors**
   - Make sure documents are uploaded first
   - Check if the RAG system is properly initialized
   - Verify the question format

4. **Authentication Errors**
   - Ensure the Authorization header is present
   - Verify the API key is correct
   - Check the Bearer token format

### Logs

The server provides detailed logging. Check the console output for:
- RAG system initialization status
- Document processing progress
- Query processing results
- Error messages and stack traces

## Security Considerations

1. **Change the Default API Key**: Update `your_api_key_here` in production
2. **Use HTTPS**: Always use HTTPS in production
3. **Rate Limiting**: Consider implementing rate limiting for production use
4. **Input Validation**: The API includes basic validation, but add more as needed
5. **File Upload Security**: Implement additional file validation for production

## Support

For issues and questions:
1. Check the console logs for error messages
2. Verify all dependencies are installed
3. Test with the provided `test_api.py` script
4. Check the system validation endpoint for component status 