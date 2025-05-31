#!/usr/bin/env python3
"""
FastAPI for Financial Sentiment Analysis
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np
import torch
import joblib
import json
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model loading and inference"""
    
    def __init__(self, model_dir: str = "outputs"):
        self.model_dir = Path(model_dir)
        self.tokenizer = None
        self.model_cls = None
        self.hybrid_pipeline = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_models(self):
        """Load all model components"""
        logger.info("Loading models...")
        
        # Load configuration
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
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
        
        # Load hybrid pipeline
        self.hybrid_pipeline = joblib.load(self.model_dir / "hybrid_pipeline.joblib")
        
        logger.info("Models loaded successfully")
    
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
    
    def compute_features(self, sentences: List[str]) -> np.ndarray:
        """Compute handcrafted features"""
        features = []
        
        for sentence in sentences:
            len_chars = len(sentence)
            len_words = len(sentence.split())
            pct_digits = len(re.findall(r'\d', sentence)) / (len(sentence) + 1)
            count_tickers = len(re.findall(r'\$[A-Z]{1,5}', sentence))
            has_profit = int('profit' in sentence.lower())
            has_loss = int('loss' in sentence.lower())
            exclamation_count = sentence.count('!')
            question_count = sentence.count('?')
            percent_signs = sentence.count('%')
            
            features.append([
                len_chars, len_words, pct_digits, count_tickers,
                has_profit, has_loss, exclamation_count,
                question_count, percent_signs
            ])
        
        return np.array(features)
    
    def predict(self, sentences: List[str]) -> List[Dict]:
        """Make predictions on sentences"""
        if not sentences:
            return []
        
        # Extract embeddings and features
        embeddings = self.extract_embeddings(sentences)
        features = self.compute_features(sentences)
        
        # Combine features
        X = np.concatenate([embeddings, features], axis=1)
        
        # Get predictions and probabilities
        predictions = self.hybrid_pipeline.predict(X)
        probabilities = self.hybrid_pipeline.predict_proba(X)
        
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
model_manager = ModelManager()

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
    title="Financial Sentiment Analysis API",
    description="API for analyzing sentiment in financial text using EconBERT",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class SentimentRequest(BaseModel):
    sentences: List[str] = Field(..., min_items=1, max_items=100, 
                                description="List of sentences to analyze (max 100)")

class SingleSentimentRequest(BaseModel):
    sentence: str = Field(..., min_length=1, max_length=500, description="Sentence to analyze")
    
class SentimentResponse(BaseModel):
    sentence: str
    predicted_sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.hybrid_pipeline is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=BatchSentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """Predict sentiment for a list of sentences"""
    import time
    
    start_time = time.time()
    
    try:
        if model_manager.hybrid_pipeline is None:
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
        if model_manager.hybrid_pipeline is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        results = model_manager.predict([request.sentence])
        return SentimentResponse(**results[0]) if results else None
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model_manager.config is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "model_type": "EconBERT + Handcrafted Features",
        "labels": list(model_manager.config["label_mapping"].keys()),
        "max_sequence_length": model_manager.config.get("max_length", 128),
        "device": str(model_manager.device)
    }

# Example usage endpoint
@app.get("/examples")
async def get_examples():
    """Get example sentences for testing"""
    return {
        "examples": [
            "Company X reported quarterly profit up 20%.",
            "Analyst warns of potential recession next year.",
            "The board decided to maintain current dividend policy.",
            "Stock prices fell sharply after earnings miss.",
            "Investment in new technology shows promising returns."
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
