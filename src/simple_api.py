#!/usr/bin/env python3
"""
Simple working API that bypasses label mapping issues
"""

import time
import logging
from datetime import datetime
from typing import List, Dict
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Sentiment Analysis API (Simple)",
    description="Working sentiment analysis API with robust label handling",
    version="2.1.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Global variables
models = {}
tokenizers = {}
prediction_cache = {}

class SingleHeadline(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    model: str = Field(default="finbert")
    
    @validator('text')
    def validate_text(cls, v):
        return v.strip()

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    processing_time_ms: float

def load_models():
    """Load models with robust error handling"""
    global models, tokenizers
    
    logger.info("Loading models...")
    
    model_configs = [
        ("finbert", "outputs/finbert_model", "outputs/finbert_tokenizer"),
        ("econbert", "outputs/econbert_model", "outputs/tokenizer")
    ]
    
    for model_name, model_path, tokenizer_path in model_configs:
        try:
            if Path(model_path).exists() and Path(tokenizer_path).exists():
                logger.info(f"Loading {model_name}...")
                
                tokenizers[model_name] = AutoTokenizer.from_pretrained(tokenizer_path)
                models[model_name] = AutoModelForSequenceClassification.from_pretrained(model_path)
                models[model_name].eval()
                
                logger.info(f"✅ {model_name} loaded successfully")
            else:
                logger.warning(f"❌ {model_name} files not found")
                
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
    
    logger.info(f"Loaded models: {list(models.keys())}")

def predict_sentiment_robust(text: str, model_name: str = "finbert") -> Dict:
    """Robust sentiment prediction that handles label mapping issues"""
    
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not available")
    
    start_time = time.time()
    
    try:
        model = models[model_name]
        tokenizer = tokenizers[model_name]
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Robust label mapping - use hardcoded labels to avoid config issues
        labels = ["negative", "neutral", "positive"]
        
        # Create probability dictionary
        prob_dict = {labels[i]: float(probs[i]) for i in range(len(probs))}
        
        # Get prediction
        predicted_class = int(probs.argmax())
        prediction = labels[predicted_class]
        confidence = float(probs[predicted_class])
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "text": text,
            "sentiment": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "model_used": model_name,
            "processing_time_ms": round(processing_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    load_models()
    logger.info("API startup complete")

@app.get("/")
async def root():
    return {
        "message": "Simple Financial Sentiment Analysis API",
        "version": "2.1.0",
        "status": "healthy",
        "available_models": ", ".join(list(models.keys()))
    }

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(item: SingleHeadline):
    """Analyze sentiment with robust handling"""
    result = predict_sentiment_robust(item.text, item.model)
    return SentimentResponse(**result)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/test")
async def test_predictions():
    """Test endpoint to verify predictions"""
    test_cases = [
        "Company profits increased significantly",
        "Revenue declined due to market conditions", 
        "Results met analyst expectations"
    ]
    
    results = []
    for text in test_cases:
        try:
            result = predict_sentiment_robust(text, "finbert")
            results.append({
                "text": text,
                "prediction": result["sentiment"],
                "confidence": result["confidence"]
            })
        except Exception as e:
            results.append({
                "text": text,
                "error": str(e)
            })
    
    return {"test_results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)