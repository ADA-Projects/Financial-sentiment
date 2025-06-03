# 📈 Financial Sentiment Analysis API

A production-ready sentiment analysis API for financial text using fine-tuned FinBERT. This project provides state-of-the-art sentiment classification specifically designed for financial news, earnings reports, and market commentary.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

- **Multiple Models**: Fine-tuned FinBERT model optimized for financial sentiment
- **High Performance**: Efficient inference with caching and batch processing
- **Production Ready**: Docker containerization, health checks, and monitoring
- **Enhanced API**: Confidence scores, batch processing, and detailed responses
- **Easy Deployment**: One-command deployment with comprehensive testing
- **Backward Compatible**: Legacy endpoint support for existing integrations

## 📊 Model Performance

| Model | Accuracy | F1-Score | Training Data |
|-------|----------|----------|---------------|
| FinBERT (Fine-tuned) | 75%+ | 0.73+ | Financial PhraseBank |

*Performance metrics from validation set. Actual performance may vary by use case.*

## 🏗️ Architecture

```
├── src/
│   ├── api.py              # FastAPI application
│   ├── dataset.py          # Data loading utilities  
│   └── train_model.py      # Model training pipeline
├── outputs/
│   ├── finbert_fixed_model/      # Fine-tuned FinBERT model
│   ├── finbert_fixed_tokenizer/  # Model tokenizer
│   └── finbert_fixed_training/   # Training checkpoints
├── data/
│   └── data.csv           # Training dataset
└── test_api.py           # API testing suite
```

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd Financial-sentiment

# Make deployment script executable
chmod +x deployment.sh

# Deploy with one command
./deployment.sh deploy
```

The API will be available at `http://localhost:8000` with automatic health checks and testing.

### Option 2: Development Mode

```bash
# Start in development mode
./deployment.sh dev

# Or manually:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Docker Compose

```bash
# Full production setup with Redis and Nginx
docker-compose up -d

# Check status
docker-compose ps
```

## 📖 API Documentation

Once running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🔧 API Endpoints

### Single Text Analysis
```bash
POST /analyze
{
    "text": "Company profits increased by 25% this quarter",
    "model": "finbert"
}
```

**Response:**
```json
{
    "text": "Company profits increased by 25% this quarter",
    "sentiment": "positive",
    "confidence": 0.892,
    "probabilities": {
        "negative": 0.034,
        "neutral": 0.074,
        "positive": 0.892
    },
    "model_used": "finbert",
    "processing_time_ms": 45.2,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Batch Processing
```bash
POST /analyze/batch
{
    "texts": [
        "Revenue declined this quarter",
        "Strong earnings beat expectations",
        "Market conditions remain stable"
    ],
    "model": "finbert"
}
```

### Model Information
```bash
GET /models
```

### Legacy Support
```bash
POST /score  # Returns simple probability format
```

## 🧪 Testing

### Automated Testing
```bash
# Run comprehensive test suite
./deployment.sh test

# Or run manually
python test_api.py
```

### Manual Testing
```bash
# Test single prediction
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Company stock price soared after earnings", "model": "finbert"}'

# Test batch processing
curl -X POST "http://localhost:8000/analyze/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Profits up 20%", "Revenue declined"], "model": "finbert"}'
```

## 🎯 Model Training

### Training New Models

```bash
# Train FinBERT model
python src/train_model.py --model finbert --epochs 3 --batch_size 16
```

### Data Requirements

The training expects a CSV file with columns:
- `Sentence`: Financial text to analyze
- `Sentiment`: Label (negative, neutral, positive)

```csv
Sentence,Sentiment
"Company profits increased significantly",positive
"Revenue declined due to market conditions",negative
"Performance remained steady",neutral
```

## 🐳 Docker Configuration

### Production Dockerfile
```dockerfile
# Multi-stage build for optimized image size
FROM python:3.10-slim as production
# ... (see dockerfile for full configuration)
```

### Environment Variables
Copy `.env.template` to `.env` and customize:

```bash
# API Configuration
API_PORT=8000
DEFAULT_MODEL=finbert

# Performance
CACHE_SIZE_LIMIT=1000
MAX_BATCH_SIZE=100

# Security
CORS_ALLOW_ORIGINS=["*"]
RATE_LIMIT_ENABLED=false
```

## 📊 Monitoring and Observability

### Health Checks
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/cache/stats
```

### Logging
```bash
# View container logs
docker logs finbert-container

# Follow logs in real-time
docker logs -f finbert-container
```

### Performance Monitoring
- Request/response times logged
- Cache hit rates tracked
- Model inference metrics
- Error rate monitoring

## 🔒 Security Features

- **Input Validation**: Pydantic models with length limits
- **Rate Limiting**: Configurable request limits
- **CORS Configuration**: Customizable origins
- **Error Handling**: Graceful error responses
- **Health Monitoring**: Automated health checks

## 🚀 Production Deployment

### Cloud Deployment

**AWS ECS/Fargate:**
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t finbert-sentiment .
docker tag finbert-sentiment:latest <account>.dkr.ecr.us-east-1.amazonaws.com/finbert-sentiment:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/finbert-sentiment:latest
```

**Google Cloud Run:**
```bash
# Deploy to Cloud Run
gcloud run deploy finbert-api \
  --image gcr.io/PROJECT-ID/finbert-sentiment \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2
```

### Scaling Considerations

- **Memory**: 4GB+ recommended for optimal performance
- **CPU**: 2+ cores for concurrent requests
- **Storage**: Models require ~1GB disk space
- **Network**: Consider CDN for global deployment

## 📈 Performance Optimization

### Inference Speed
- Model caching reduces cold start time
- Batch processing for multiple texts
- Optimized tokenization pipeline
- Memory-efficient model loading

### Caching Strategy
- In-memory cache for frequent requests
- Redis support for distributed caching
- Configurable TTL and size limits
- Cache statistics and monitoring

## 🛠️ Development

### Local Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run with auto-reload
uvicorn src.api:app --reload

# Run tests
pytest tests/

# Code formatting
black src/
isort src/
```

### Adding New Models
1. Train your model using `src/train_model.py`
2. Save to `outputs/your_model/`
3. Update model loading in `src/api.py`
4. Add tests in `test_api.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library
- **FinBERT authors** for the pre-trained financial model
- **FastAPI team** for the excellent web framework
- **Financial PhraseBank** dataset contributors

## 📞 Support

- **Documentation**: Check `/docs` endpoint when running
- **Issues**: Open a GitHub issue
- **Performance**: See monitoring endpoints for diagnostics

---

