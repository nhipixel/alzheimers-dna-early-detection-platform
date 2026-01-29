# Backend API - Enhanced Documentation

## 🏗️ Architecture Overview

The backend is built with FastAPI and follows a modular architecture:

```
backend/
├── main.py                          # FastAPI application entry point
├── app/
│   ├── config/settings.py          # Application configuration
│   ├── models/
│   │   ├── schemas.py              # Pydantic models
│   │   └── loader.py               # Model loading logic
│   ├── routes/
│   │   ├── predict.py              # Prediction endpoints
│   │   ├── health.py               # Health check endpoints
│   │   ├── analysis.py             # Analysis management
│   │   └── websocket.py            # Real-time WebSocket
│   ├── services/
│   │   └── model_service.py        # Model prediction service
│   ├── middleware/
│   │   ├── logging.py              # Request logging
│   │   └── error_handler.py        # Error handling
│   └── utils/helpers.py            # Utility functions
```

## 🚀 Quick Start

### Installation

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Running the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## 📡 API Endpoints

### Health & Monitoring

#### GET /api/v1/health
Get detailed health status of the backend.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "version": "1.0.0",
  "models": {
    "xgboost": true,
    "pytorch": true
  }
}
```

#### GET /api/v1/models/info
Get information about all loaded models.

**Response:**
```json
{
  "models": {
    "xgboost": {
      "loaded": true,
      "type": "XGBoost",
      "version": "2.0.0"
    },
    "pytorch": {
      "loaded": true,
      "type": "PyTorch CNN",
      "version": "2.0.0"
    }
  }
}
```

#### GET /api/v1/system/info
Get system information.

**Response:**
```json
{
  "platform": "Windows",
  "python_version": "3.10.0",
  "cpu_count": 8,
  "memory_total": "16GB"
}
```

### Predictions

#### POST /api/v1/predict
Make predictions using trained models.

**Request Body:**
```json
{
  "data": [[0.5, 0.3, 0.8, ...]],
  "model_type": "both"
}
```

**Parameters:**
- `data`: Array of feature arrays (methylation values)
- `model_type`: "xgboost", "pytorch", or "both"

**Response:**
```json
{
  "predictions": {
    "xgboost": {
      "class": 1,
      "probabilities": [0.1, 0.8, 0.1],
      "confidence": 0.8
    },
    "pytorch": {
      "class": 1,
      "probabilities": [0.15, 0.75, 0.1],
      "confidence": 0.75
    }
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### Analysis Management

#### GET /api/v1/analyses
List all analyses.

**Response:**
```json
{
  "analyses": [
    {
      "id": "abc123",
      "timestamp": "2024-01-15T10:30:00",
      "status": "completed"
    }
  ],
  "total": 1
}
```

#### GET /api/v1/analyses/{id}
Get a specific analysis by ID.

#### DELETE /api/v1/analyses/{id}
Delete an analysis by ID.

### WebSocket Endpoints

#### WS /api/v1/ws/status
Real-time status updates every 5 seconds.

**Message Format:**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "type": "status_update",
  "data": {
    "backend_status": "online",
    "models": {
      "xgboost": {
        "loaded": true,
        "status": "ready"
      },
      "pytorch": {
        "loaded": true,
        "status": "ready"
      }
    }
  }
}
```

#### WS /api/v1/ws/predictions
Real-time prediction updates (coming soon).

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True
LOG_LEVEL=INFO

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001

# Models
XGBOOST_MODEL_PATH=../model/models/xgboost/model.pkl
PYTORCH_MODEL_PATH=../model/models/pytorch/model.pth

# API
API_V1_PREFIX=/api/v1
APP_NAME=Alzheimers DNA Detection API
APP_VERSION=1.0.0
```

## 🧪 Testing

### Manual Testing with curl

**Health Check:**
```bash
curl http://localhost:8000/api/v1/health
```

**Prediction:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[0.5, 0.3, 0.8]], "model_type": "both"}'
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔒 Middleware

### Request Logging
All requests are logged with:
- Method and path
- Response status code
- Processing time (added to `X-Process-Time` header)

### Error Handling
Automatic error handling for:
- HTTP exceptions
- Unhandled exceptions
- Model loading errors
- Validation errors

## 📊 Model Service

The `ModelService` class handles:
- Loading XGBoost and PyTorch models
- Making predictions
- Managing model state
- Error handling for missing models

**Key Features:**
- Singleton pattern (single instance)
- Lazy loading of models
- Fallback mechanisms
- Comprehensive error logging

## 🔄 Real-time Features

### WebSocket Connection Manager
- Manages multiple WebSocket connections
- Broadcasts status updates
- Auto-reconnection support
- Connection health monitoring

### Status Broadcasting
- Updates every 5 seconds
- Backend status
- Model availability
- System metrics

## 🚦 Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Invalid data format |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Models not loaded |

## 📈 Performance

### Optimization Features
- Async/await for I/O operations
- Connection pooling for WebSockets
- Efficient model loading
- Request caching (planned)

### Monitoring
- Request logging
- Processing time tracking
- Error rate monitoring
- Model performance metrics

## 🔐 Security

### Best Practices
- CORS configuration
- Input validation with Pydantic
- Error message sanitization
- Rate limiting (planned)

## 🛠️ Development

### Running in Development Mode

```bash
python main.py
```

This will start the server with:
- Auto-reload enabled
- Debug logging
- CORS for localhost:3000

### Code Structure

**Routes**: Define API endpoints
**Controllers**: Business logic
**Services**: External service integration
**Models**: Data validation and schemas
**Utils**: Helper functions
**Middleware**: Request/response processing

## 📦 Dependencies

Key dependencies:
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `pydantic`: Data validation
- `xgboost`: XGBoost model support
- `torch`: PyTorch model support
- `numpy`: Numerical operations
- `python-multipart`: File upload support

## 🚀 Deployment

### Production Checklist
- [ ] Set `DEBUG=False`
- [ ] Configure proper CORS origins
- [ ] Set up SSL/TLS
- [ ] Configure logging to file
- [ ] Set up monitoring
- [ ] Enable rate limiting
- [ ] Configure database for analysis storage

### Docker Support (Coming Soon)

## 📝 Changelog

### v1.0.0 (Current)
- Initial release
- Prediction endpoints
- Health monitoring
- WebSocket support
- Batch prediction support
- Real-time status updates
- Enhanced error handling
- Request logging middleware

## 🤝 Contributing

See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - See [LICENSE](../LICENSE) for details.

**Note:** This project is forked from [hackbio-ca/hackathon](https://github.com/hackbio-ca/hackathon).
