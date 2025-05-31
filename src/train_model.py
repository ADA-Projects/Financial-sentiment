#!/usr/bin/env python3
"""
EconBERT-Only training script for Financial Sentiment Analysis
No handcrafted features - just pure EconBERT embeddings
"""

import pandas as pd
import numpy as np
import torch
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EconBERTOnlyTrainer:
    def __init__(self, config_path="config.json"):
        """Initialize trainer with configuration"""
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cls = None
        self.tokenizer = None
        self.pipeline = None
        
    def load_config(self, config_path):
        """Load training configuration"""
        default_config = {
            "data_path": "data/data.csv",
            "output_dir": "outputs",
            "model_name": "climatebert/econbert",
            "max_length": 128,
            "batch_size": 32,
            "test_size": 0.2,
            "cv_folds": 5,
            "random_state": 42,
            "label_mapping": {"negative": 0, "neutral": 1, "positive": 2}
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_econbert(self):
        """Setup EconBERT model and tokenizer"""
        logger.info("Setting up EconBERT...")
        
        # Download EconBERT repository
        repo_local = snapshot_download(
            repo_id=self.config["model_name"], 
            repo_type="model"
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
            num_labels=3,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model_cls.eval()
        self.model_cls.to(self.device)
        
        logger.info("EconBERT setup complete")
    
    def load_and_prepare_data(self):
        """Load and prepare training data"""
        logger.info("Loading data...")
        
        df = pd.read_csv(self.config["data_path"])
        
        # Balance the dataset (upsample minority classes)
        label_counts = df['Sentiment'].value_counts()
        max_count = label_counts.max()
        
        balanced_dfs = []
        for label in df['Sentiment'].unique():
            label_df = df[df['Sentiment'] == label]
            upsampled = label_df.sample(
                max_count, 
                replace=True, 
                random_state=self.config["random_state"]
            )
            balanced_dfs.append(upsampled)
        
        balanced_df = pd.concat(balanced_dfs).sample(
            frac=1, 
            random_state=self.config["random_state"]
        ).reset_index(drop=True)
        
        logger.info(f"Original dataset: {len(df)} samples")
        logger.info(f"Balanced dataset: {len(balanced_df)} samples")
        
        return balanced_df
    
    def extract_embeddings(self, sentences, batch_size=None):
        """Extract CLS embeddings from EconBERT"""
        if batch_size is None:
            batch_size = self.config["batch_size"]
            
        embeddings = []
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Extracting embeddings"):
            batch = sentences[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.config["max_length"]
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model_cls(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # Last layer
                cls_embeddings = hidden_states[:, 0, :].cpu().numpy()  # CLS token
                embeddings.append(cls_embeddings)
            
            # Clear GPU memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.vstack(embeddings)
    
    def train_econbert_only(self, df):
        """Train using ONLY EconBERT embeddings"""
        logger.info("Training EconBERT-only model...")
        
        sentences = df['Sentence'].tolist()
        labels = df['Sentiment'].map(self.config["label_mapping"]).values
        
        # Extract ONLY embeddings (no handcrafted features)
        embeddings = self.extract_embeddings(sentences)
        
        # Use embeddings directly
        X = embeddings  # Shape: (n_samples, 768)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels,
            test_size=self.config["test_size"],
            stratify=labels,
            random_state=self.config["random_state"]
        )
        
        # Create pipeline with ONLY embeddings
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                solver='lbfgs',
                random_state=self.config["random_state"]
            )
        )
        
        # Cross-validation for robust evaluation
        cv_scores = []
        skf = StratifiedKFold(n_splits=self.config["cv_folds"], shuffle=True, 
                             random_state=self.config["random_state"])
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            fold_pipeline = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight='balanced',
                    max_iter=1000,
                    solver='lbfgs',
                    random_state=self.config["random_state"]
                )
            )
            
            fold_pipeline.fit(X_fold_train, y_fold_train)
            y_pred = fold_pipeline.predict(X_fold_val)
            score = f1_score(y_fold_val, y_pred, average='macro')
            cv_scores.append(score)
            logger.info(f"Fold {fold+1} macro F1: {score:.4f}")
        
        logger.info(f"CV macro F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train final model on all training data
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average='macro')
        logger.info(f"Test macro F1: {test_f1:.4f}")
        
        # Detailed classification report
        id2label = {v: k for k, v in self.config["label_mapping"].items()}
        target_names = [id2label[i] for i in sorted(id2label.keys())]
        
        print("\nEconBERT-Only Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return {
            'cv_scores': cv_scores,
            'test_f1': test_f1,
            'test_report': classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        }
    
    def save_model(self):
        """Save model components"""
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(exist_ok=True)
        
        # Save pipeline (small file)
        joblib.dump(self.pipeline, output_dir / "econbert_only_pipeline.joblib")
        
        # Save tokenizer
        tokenizer_dir = output_dir / "tokenizer"
        self.tokenizer.save_pretrained(str(tokenizer_dir))
        
        # Save model
        model_dir = output_dir / "econbert_model"
        self.model_cls.save_pretrained(str(model_dir))
        
        # Save configuration
        model_info = {
            "config": self.config,
            "model_name": self.config["model_name"],
            "model_type": "econbert_only",
            "embedding_dim": 768,
            "total_features": 768  # Only embeddings
        }
        
        with open(output_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"EconBERT-only model saved to {output_dir}")
    
    def run_training(self):
        """Main training workflow"""
        logger.info("Starting EconBERT-only training workflow...")
        
        # Setup model
        self.setup_econbert()
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        
        # Train model
        results = self.train_econbert_only(df)
        
        # Save model
        self.save_model()
        
        logger.info("Training complete!")
        return results

def main():
    """Main function"""
    trainer = EconBERTOnlyTrainer()
    results = trainer.run_training()
    print(f"\nFinal Results: {results}")

if __name__ == "__main__":
    main()