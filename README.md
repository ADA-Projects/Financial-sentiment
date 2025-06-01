# Financial Sentiment Analysis
Production-ready financial sentiment analysis using EconBERT embeddings with class-weighted training.
## Kaggle Setup

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account → Create New API Token
3. Download `kaggle.json`
4. Set up credentials locally:
```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```


## 🚀 Quick Start

### 1. Train the Model
```bash
python src/train_model.py
```

### 2. Start the API
```bash
python src/api.py
```

### 3. Test the API
```bash
python test_api.py
```

### 4. Docker Deployment
```bash
docker build -t financial-sentiment .
docker run -p 8000:8000 financial-sentiment
```

## 📊 Model Performance

- **Architecture**: EconBERT embeddings + Logistic Regression
- **Training Method**: Class-weighted (no upsampling) 
- **Test Macro F1**: ~0.71
- **Labels**: negative, neutral, positive

## 🔧 Key Features

- **Space-efficient**: Uses class weights instead of data upsampling
- **Production-ready**: FastAPI with automatic documentation
- **Docker support**: Containerized for easy deployment
- **Git LFS**: Handles large model files properly

## 📁 Project Structure

```
├── src/
│   ├── train_model.py      # Training pipeline
│   ├── api.py              # FastAPI service
│   └── dataset.py          # Dataset utilities
├── outputs/                # Trained model artifacts
├── data/                   # Training data
├── notebooks/              # Exploratory analysis
└── test_api.py            # API testing
```

## 🌐 API Endpoints

- `GET /health` - Health check
- `POST /predict` - Batch sentiment prediction
- `POST /predict/single` - Single sentence prediction
- `GET /model/info` - Model information
- `GET /docs` - Interactive API documentation

## 📈 Example Usage

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict/single",
    json={"sentence": "Company profits soared 25% this quarter"}
)
print(response.json())
# {"predicted_sentiment": "positive", "confidence": 0.85}

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"sentences": ["Market crash threatens portfolios", "Strong earnings beat"]}
)
```

## 🛠️ Development

### Requirements
```bash
pip install -r requirements.txt
```

### Configuration
Edit `config.json` to modify training parameters:
- `model_name`: Base model to use
- `batch_size`: Training batch size  
- `test_size`: Validation split ratio

### Training Process
1. Loads Financial Phrase Bank dataset
2. Extracts EconBERT embeddings (768-dim)
3. Trains class-weighted Logistic Regression
4. Evaluates with cross-validation
5. Saves model artifacts to `outputs/`

## 🚢 Deployment Options

- **Local**: `python src/api.py`
- **Docker**: `docker run -p 8000:8000 financial-sentiment`
- **Cloud**: Deploy to Heroku, Railway, or Render
- **Serverless**: Adapt for AWS Lambda or Google Cloud Functions

## 📚 Research Notes

This implementation addresses key challenges in financial sentiment analysis:

1. **Class Imbalance**: Uses class weights instead of upsampling
2. **Domain Adaptation**: Leverages EconBERT's financial understanding  
3. **Production Readiness**: Includes API, testing, and containerization
4. **Space Efficiency**: Avoids storing large model weights multiple times

## 🔬 Experimentation

The `notebooks/` folder contains exploratory analysis showing:
- Original data distribution and quality
- Comparison of different balancing strategies
- Performance analysis across model architectures

## 📝 License

MIT License - see LICENSE file for details.