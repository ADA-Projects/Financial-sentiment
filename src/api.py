#!/usr/bin/env python3
"""
Financial Sentiment Analysis API
"""

import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union
from pathlib import Path

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
    allow_origins=["*"],
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
    """Load models and tokenizers from model files (not pipelines)"""
    global models, tokenizers, model_info
    
    logger.info("Loading models from model files...")
    
    # Define available models with their paths
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
    
    # Load each model
    for model_name, info in available_models.items():
        try:
            model_path = Path(info["model_path"])
            tokenizer_path = Path(info["tokenizer_path"])
            
            if model_path.exists() and tokenizer_path.exists():
                logger.info(f"Loading {model_name} model and tokenizer")
                
                # Load tokenizer
                tokenizers[model_name] = AutoTokenizer.from_pretrained(str(tokenizer_path))
                
                # Load model
                models[model_name] = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                models[model_name].eval()  # Set to evaluation mode
                
                logger.info(f"✅ Successfully loaded {model_name}")
            else:
                logger.warning(f"❌ Could not find {model_name} model files")
                logger.warning(f"   Model path exists: {model_path.exists()}")
                logger.warning(f"   Tokenizer path exists: {tokenizer_path.exists()}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
    
    logger.info(f"Successfully loaded models: {list(models.keys())}")
    
    # Create model info
    model_info = {
        "models": {name: info for name, info in available_models.items() if name in models},
        "default_model": "finbert" if "finbert" in models else list(models.keys())[0] if models else None,
        "created_at": datetime.now().isoformat()
    }

def predict_sentiment(text: str, model_name: str = "finbert") -> Dict:
    """Predict sentiment using transformers models"""
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not available")
    
    start_time = time.time()
    
    try:
        model = models[model_name]
        tokenizer = tokenizers[model_name]
        
        # Tokenize input
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Map probabilities to labels
        labels = ["negative", "neutral", "positive"]
        prob_dict = {labels[i]: float(probs[i]) for i in range(len(probs))}
        
        # Get prediction
        predicted_class = np.argmax(probs)
        prediction = labels[predicted_class]
        confidence = float(probs[predicted_class])
        
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

@app.get("/")
async def root():
    """API health check and info"""
    return {
        "message": "Financial Sentiment Analysis API",
        "version": "2.0.0",
        "status": "healthy",
        "available_models": ", ".join(list(models.keys())),
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
            "loaded": True,
            "type": "transformers"
        }
    
    return ModelInfo(
        available_models=list(models.keys()),
        default_model=model_info.get("default_model", "finbert"),
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
    if len(prediction_cache) < 1000:
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
        "hit_rate": "Not implemented"
    }

@app.delete("/cache/clear")
async def clear_cache():
    """Clear prediction cache"""
    global prediction_cache
    cache_size = len(prediction_cache)
    prediction_cache.clear()
    return {"message": f"Cache cleared. Removed {cache_size} entries."}

@app.post("/score")
async def score_headline(item: SingleHeadline):
    """Legacy endpoint for backward compatibility"""
    result = await analyze_sentiment(item)
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

# Debug endpoint to check model loading
@app.get("/debug/models")
async def debug_models():
    """Debug endpoint to check model loading status"""
    debug_info = {
        "models_loaded": list(models.keys()),
        "tokenizers_loaded": list(tokenizers.keys()),
        "model_paths_checked": {}
    }
    
    # Check paths
    paths_to_check = {
        "finbert_model": "outputs/finbert_model",
        "finbert_tokenizer": "outputs/finbert_tokenizer", 
        "econbert_model": "outputs/econbert_model",
        "econbert_tokenizer": "outputs/tokenizer"
    }
    
    for name, path in paths_to_check.items():
        debug_info["model_paths_checked"][name] = {
            "path": path,
            "exists": Path(path).exists(),
            "files": list(Path(path).glob("*")) if Path(path).exists() else []
        }
    
    return debug_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
