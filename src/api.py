#!/usr/bin/env python3
"""
Enhanced Financial Sentiment Analysis API
Features: Multiple models, confidence scores, batch processing, caching
"""

import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union
from pathlib import Path

import joblib
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Sentiment Analysis API",
    description="Production-ready sentiment analysis for financial text using FinBERT and EconBERT",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and cache
models = {}
tokenizers = {}
model_info = {}
prediction_cache = {}

class SingleHeadline(BaseModel):
    """Single headline for sentiment analysis"""
    text: str = Field(..., min_length=1, max_length=1000, description="Financial text to analyze")
    model: str = Field(default="finbert", description="Model to use: 'finbert' or 'econbert'")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class BatchHeadlines(BaseModel):
    """Batch of headlines for sentiment analysis"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of financial texts")
    model: str = Field(default="finbert", description="Model to use: 'finbert' or 'econbert'")

class SentimentResponse(BaseModel):
    """Enhanced sentiment analysis response"""
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    model_used: str
    processing_time_ms: float
    timestamp: str

class BatchSentimentResponse(BaseModel):
    """Batch sentiment analysis response"""
    results: List[SentimentResponse]
    total_processed: int
    average_confidence: float
    processing_time_ms: float
    model_used: str

class ModelInfo(BaseModel):
    """Model information response"""
    available_models: List[str]
    default_model: str
    model_details: Dict[str, Dict]

def load_models():
    """Load all available models and tokenizers"""
    global models, tokenizers, model_info
    
    logger.info("Loading models...")
    
    # Try to load model info
    info_path = Path("outputs/model_info.json")
    if info_path.exists():
        with open(info_path, 'r') as f:
            model_info = json.load(f)
    
    # Define available models
    available_models = {
        "finbert": {
            "model_path": "outputs/finbert_model",
            "tokenizer_path": "outputs/finbert_tokenizer",
            "description": "FinBERT model fine-tuned for financial sentiment"
        },
        "econbert": {
            "model_path": "outputs/econbert_model", 
            "tokenizer_path": "outputs/tokenizer",
            "description": "EconBERT model for economic text analysis"
        }
    }
    
    # Try to load pipeline models first (if they exist)
    pipeline_paths = {
        "finbert": "outputs/finbert_pipeline.joblib",
        "econbert": "outputs/class_weighted_pipeline.joblib"
    }
    
    for model_name, info in available_models.items():
        try:
            # First try to load pipeline (faster)
            pipeline_path = pipeline_paths.get(model_name)
            if pipeline_path and Path(pipeline_path).exists():
                logger.info(f"Loading {model_name} pipeline from {pipeline_path}")
                models[model_name] = joblib.load(pipeline_path)
                tokenizers[model_name] = None  # Pipeline includes tokenizer
            else:
                # Fallback to manual loading
                model_path = Path(info["model_path"])
                tokenizer_path = Path(info["tokenizer_path"])
                
                if model_path.exists() and tokenizer_path.exists():
                    logger.info(f"Loading {model_name} model and tokenizer")
                    tokenizers[model_name] = AutoTokenizer.from_pretrained(str(tokenizer_path))
                    models[model_name] = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                    models[model_name].eval()  # Set to evaluation mode
                else:
                    logger.warning(f"Could not find {model_name} model files")
                    
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
    
    logger.info(f"Successfully loaded models: {list(models.keys())}")

def predict_sentiment(text: str, model_name: str = "finbert") -> Dict:
    """Predict sentiment with confidence scores"""
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not available")
    
    start_time = time.time()
    
    try:
        model = models[model_name]
        
        # Check if it's a pipeline or raw model
        if hasattr(model, 'predict_proba'):
            # It's a sklearn pipeline
            probs = model.predict_proba([text])[0]
            prediction = model.predict([text])[0]
            
            # Map to sentiment labels
            labels = ["negative", "neutral", "positive"]
            prob_dict = {labels[i]: float(probs[i]) for i in range(len(probs))}
            
        else:
            # It's a transformers model
            tokenizer = tokenizers[model_name]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            labels = ["negative", "neutral", "positive"]
            prob_dict = {labels[i]: float(probs[i]) for i in range(len(probs))}
            prediction = labels[np.argmax(probs)]
        
        confidence = max(prob_dict.values())
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "text": text,
            "sentiment": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "model_used": model_name,
            "processing_time_ms": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def get_cache_key(text: str, model: str) -> str:
    """Generate cache key for predictions"""
    import hashlib
    return hashlib.md5(f"{text}_{model}".encode()).hexdigest()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_models()
    logger.info("API startup complete")

@app.get("/", response_model=Dict[str, str])
async def root():
    """API health check and info"""
    return {
        "message": "Financial Sentiment Analysis API",
        "version": "2.0.0",
        "status": "healthy",
        "available_models": list(models.keys()),
        "documentation": "/docs"
    }

@app.get("/models", response_model=ModelInfo)
async def get_models():
    """Get information about available models"""
    model_details = {}
    for model_name in models.keys():
        model_details[model_name] = {
            "description": f"{model_name.title()} model for financial sentiment analysis",
            "labels": ["negative", "neutral", "positive"],
            "loaded": True
        }
    
    return ModelInfo(
        available_models=list(models.keys()),
        default_model="finbert",
        model_details=model_details
    )

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(item: SingleHeadline):
    """Analyze sentiment of a single text"""
    # Check cache first
    cache_key = get_cache_key(item.text, item.model)
    if cache_key in prediction_cache:
        logger.info("Cache hit")
        return SentimentResponse(**prediction_cache[cache_key])
    
    # Make prediction
    result = predict_sentiment(item.text, item.model)
    
    # Cache result (with simple size limit)
    if len(prediction_cache) < 1000:  # Simple cache size limit
        prediction_cache[cache_key] = result
    
    return SentimentResponse(**result)

@app.post("/analyze/batch", response_model=BatchSentimentResponse)
async def analyze_batch(item: BatchHeadlines):
    """Analyze sentiment of multiple texts"""
    start_time = time.time()
    results = []
    
    for text in item.texts:
        try:
            result = predict_sentiment(text, item.model)
            results.append(SentimentResponse(**result))
        except Exception as e:
            logger.error(f"Failed to process text: {text[:50]}... Error: {e}")
            # Continue with other texts
            continue
    
    total_time = (time.time() - start_time) * 1000
    avg_confidence = np.mean([r.confidence for r in results]) if results else 0.0
    
    return BatchSentimentResponse(
        results=results,
        total_processed=len(results),
        average_confidence=round(avg_confidence, 3),
        processing_time_ms=round(total_time, 2),
        model_used=item.model
    )

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "cache_size": len(prediction_cache),
        "cache_limit": 1000,
        "hit_rate": "Not implemented"  # Would need hit/miss counters
    }

@app.delete("/cache/clear")
async def clear_cache():
    """Clear prediction cache"""
    global prediction_cache
    cache_size = len(prediction_cache)
    prediction_cache.clear()
    return {"message": f"Cache cleared. Removed {cache_size} entries."}

# Legacy endpoint for backward compatibility
@app.post("/score")
async def score_headline(item: SingleHeadline):
    """Legacy endpoint - redirects to /analyze"""
    result = await analyze_sentiment(item)
    # Return in old format for compatibility
    return result.probabilities

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "cache_size": len(prediction_cache),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
