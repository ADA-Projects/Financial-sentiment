#!/usr/bin/env python3
"""
EconBERT-Only API for Financial Sentiment Analysis
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np
import torch
import joblib
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EconBERTOnlyManager:
    """Manages EconBERT-only model loading and inference"""
    
    def __init__(self, model_dir: str = "outputs"):
        self.model_dir = Path(model_dir)
        self.tokenizer = None
        self.model_cls = None
        self.pipeline = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_models(self):
        """Load all model components"""
        logger.info("Loading EconBERT-only models...")
        
        # Load configuration
        config_path = self.model_dir / "model_info.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_info = json.load(f)
                self.config = model_info.get("config", {})
        else:
            # Default config
            self.config = {
                "label_mapping": {"negative": 0, "neutral": 1, "positive": 2},
                "max_length": 128
            }
        
        # Load tokenizer
        tokenizer_path = self.model_dir / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            use_fast=True,
            trust_remote_code=True
        )
        
        # Load EconBERT model
        model_path = self.model_dir / "econbert_model"
        self.model_cls = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model_cls.eval()
        self.model_cls.to(self.device)
        
        # Load pipeline (should be econbert_only_pipeline.joblib)
        pipeline_files = [
            "class_weighted_pipeline.joblib",  # New name
            "econbert_only_pipeline.joblib",   # Previous name
            "hybrid_pipeline.joblib"           # Original name
        ]
        
        for filename in pipeline_files:
            pipeline_path = self.model_dir / filename
            if pipeline_path.exists():
                self.pipeline = joblib.load(pipeline_path)
                logger.info(f"Loaded pipeline from {filename}")
                break
        
        if self.pipeline is None:
            raise FileNotFoundError("No pipeline found in outputs directory")
        
        logger.info("EconBERT-only models loaded successfully")
    
    def extract_embeddings(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        """Extract embeddings from sentences"""
        embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding="longest", 
                truncation=True,
                max_length=self.config.get("max_length", 128)
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model_cls(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)
    
    def predict(self, sentences: List[str]) -> List[Dict]:
        """Make predictions using ONLY EconBERT embeddings"""
        if not sentences:
            return []
        
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded")
        
        # Extract ONLY embeddings (no handcrafted features)
        embeddings = self.extract_embeddings(sentences)
        
        # Use embeddings directly (no concatenation)
        X = embeddings  # Shape: (n_sentences, 768)
        
        # Get predictions and probabilities
        predictions = self.pipeline.predict(X)
        probabilities = self.pipeline.predict_proba(X)
        
        # Convert to readable format
        id2label = {v: k for k, v in self.config["label_mapping"].items()}
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                "sentence": sentences[i],
                "predicted_sentiment": id2label[pred],
                "confidence": float(np.max(probs)),
                "probabilities": {
                    id2label[j]: float(probs[j]) 
                    for j in range(len(probs))
                }
            }
            results.append(result)
        
        return results

# Global model manager
model_manager = EconBERTOnlyManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    model_manager.load_models()
    yield
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Financial Sentiment Analysis API (EconBERT-Only)",
    description="API using pure EconBERT embeddings for financial sentiment analysis",
    version="2.0.0",
    lifespan=lifespan
)

# Request/Response models
class SentimentRequest(BaseModel):
    sentences: List[str] = Field(..., min_items=1, max_items=100)

class SingleSentimentRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=500)
    
class SentimentResponse(BaseModel):
    sentence: str
    predicted_sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    processing_time_ms: float

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.pipeline is not None,
        "version": "2.0.0",
        "model_type": "econbert_only"
    }

@app.post("/predict", response_model=BatchSentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """Predict sentiment for a list of sentences"""
    import time
    
    start_time = time.time()
    
    try:
        if model_manager.pipeline is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Make predictions
        results = model_manager.predict(request.sentences)
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchSentimentResponse(
            results=[SentimentResponse(**result) for result in results],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/single", response_model=SentimentResponse)
async def predict_single_sentiment(request: SingleSentimentRequest):
    """Predict sentiment for a single sentence"""
    try:
        if model_manager.pipeline is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        results = model_manager.predict([request.sentence])
        return SentimentResponse(**results[0]) if results else None
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "EconBERT-Only (No Handcrafted Features)",
        "labels": list(model_manager.config["label_mapping"].keys()),
        "max_sequence_length": model_manager.config.get("max_length", 128),
        "device": str(model_manager.device),
        "embedding_dim": 768,
        "features": "Pure EconBERT embeddings only"
    }

@app.get("/examples")
async def get_examples():
    """Get example sentences for testing"""
    return {
        "examples": [
            "Company X reported quarterly profit up 20%.",
            "Market crash threatens investor portfolios.",
            "Stock prices fell sharply after earnings miss.",
            "Fed raises interest rates causing market volatility.",
            "The board decided to maintain current dividend policy."
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
