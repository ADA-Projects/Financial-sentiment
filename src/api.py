#!/usr/bin/env python3
"""
Lightweight FastAPI for Financial Sentiment Analysis
Downloads model on-demand to save space
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
from huggingface_hub import snapshot_download
import logging
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightModelManager:
    """Manages model loading with minimal disk usage"""
    
    def __init__(self, model_dir: str = "outputs"):
        self.model_dir = Path(model_dir)
        self.tokenizer = None
        self.model_cls = None
        self.hybrid_pipeline = None
        self.model_info = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._temp_model_dir = None
        
    def load_pipeline_only(self):
        """Load only the trained pipeline (small file)"""
        logger.info("Loading hybrid pipeline...")
        
        # Load pipeline (this is small - few MB)
        pipeline_path = self.model_dir / "hybrid_pipeline.joblib"
        if pipeline_path.exists():
            self.hybrid_pipeline = joblib.load(pipeline_path)
            logger.info("Pipeline loaded")
        else:
            raise FileNotFoundError("No trained pipeline found. Run training first.")
        
        # Load model info
        info_path = self.model_dir / "model_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.model_info = json.load(f)
        else:
            # Default config if file doesn't exist
            self.model_info = {
                "model_name": "climatebert/econbert",
                "embedding_dim": 768,
                "feature_names": [
                    "len_chars", "len_words", "pct_digits", "count_tickers",
                    "has_profit", "has_loss", "exclamation_count", 
                    "question_count", "percent_signs"
                ]
            }
    
    def load_econbert_on_demand(self):
        """Load EconBERT only when needed (downloads fresh each time)"""
        if self.tokenizer is not None and self.model_cls is not None:
            return  # Already loaded
            
        logger.info("Loading EconBERT on-demand...")
        
        try:
            # Use temporary directory
            self._temp_model_dir = tempfile.mkdtemp()
            
            # Download to temp directory
            repo_local = snapshot_download(
                repo_id=self.model_info["model_name"],
                repo_type="model",
                cache_dir=self._temp_model_dir
            )
            
            # Load tokenizer
            tokenizer_path = Path(repo_local) / "EconBERT_Model" / "econbert_tokenizer"
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                use_fast=True,
                trust_remote_code=True
            )
            
            # Load model
            model_path = Path(repo_local) / "EconBERT_Model" / "econbert_weights"
            self.model_cls = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model_cls.eval()
            self.model_cls.to(self.device)
            
            logger.info("EconBERT loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load EconBERT: {e}")
            raise
    
    def cleanup_temp_files(self):
        """Clean up temporary model files"""
        if self._temp_model_dir and Path(self._temp_model_dir).exists():
            shutil.rmtree(self._temp_model_dir)
            logger.info("Cleaned up temporary model files")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def extract_embeddings(self, sentences: List[str], batch_size: int = 16) -> np.ndarray:
        """Extract embeddings with automatic cleanup"""
        # Load model if not already loaded
        self.load_econbert_on_demand()
        
        embeddings = []
        
        try:
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding="longest", 
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model_cls(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
                    embeddings.append(cls_embeddings)
                
                # Clear memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return np.vstack(embeddings)
            
        finally:
            # Always cleanup after extraction
            self.cleanup_temp_files()
            self.tokenizer = None
            self.model_cls = None
    
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
        """Make predictions"""
        if not sentences:
            return []
        
        if self.hybrid_pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline_only() first.")
        
        # Extract embeddings and features
        embeddings = self.extract_embeddings(sentences)
        features = self.compute_features(sentences)
        
        # Combine features
        X = np.concatenate([embeddings, features], axis=1)
        
        # Get predictions and probabilities
        predictions = self.hybrid_pipeline.predict(X)
        probabilities = self.hybrid_pipeline.predict_proba(X)
        
        # Convert to readable format
        label_mapping = self.model_info.get("config", {}).get("label_mapping", 
                                                              {"negative": 0, "neutral": 1, "positive": 2})
        id2label = {v: k for k, v in label_mapping.items()}
        
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
model_manager = LightweightModelManager()

# Create FastAPI app
app = FastAPI(
    title="Financial Sentiment Analysis API (Lightweight)",
    description="Space-efficient API for financial sentiment analysis",
    version="1.0.0"
)

# Load only the pipeline on startup
@app.on_event("startup")
async def startup_event():
    model_manager.load_pipeline_only()

# Request/Response models
class SentimentRequest(BaseModel):
    sentences: List[str] = Field(..., min_items=1, max_items=50)  # Reduced max for memory

class SentimentResponse(BaseModel):
    sentence: str
    predicted_sentiment: str
    confidence: float
    probabilities: Dict[str, float]

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_loaded": model_manager.hybrid_pipeline is not None,
        "version": "1.0.0",
        "mode": "lightweight"
    }

@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    """Predict sentiment for sentences"""
    import time
    start_time = time.time()
    
    try:
        results = model_manager.predict(request.sentences)
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "processing_time_ms": processing_time,
            "note": "Model downloaded fresh for this request"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/single")
async def predict_single_sentiment(sentence: str = Field(..., min_length=1, max_length=500)):
    """Predict sentiment for a single sentence"""
    try:
        results = model_manager.predict([sentence])
        return results[0] if results else None
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model_manager.model_info is None:
        raise HTTPException(status_code=503, detail="Model info not loaded")
    
    return {
        "model_type": "EconBERT + Handcrafted Features (Lightweight)",
        "feature_names": model_manager.model_info.get("feature_names", []),
        "embedding_dim": model_manager.model_info.get("embedding_dim", 768),
        "note": "Model weights downloaded on-demand to save space"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
