#!/usr/bin/env python3
"""
FinBERT training script - should be more stable than EconBERT
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
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinBERTTrainer:
    def __init__(self, config_path="config.json"):
        """Initialize FinBERT trainer"""
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
            "model_name": "ProsusAI/finbert",  # Use FinBERT instead of EconBERT
            "max_length": 128,
            "batch_size": 32,
            "test_size": 0.2,
            "cv_folds": 5,
            "random_state": 42,
            "balance_data": False,  # Use original distribution
            "label_mapping": {"negative": 0, "neutral": 1, "positive": 2}
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_finbert(self):
        """Setup FinBERT model and tokenizer"""
        logger.info("Setting up FinBERT...")
        
        # Load FinBERT directly from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            use_fast=True
        )
        
        # Load FinBERT model
        self.model_cls = AutoModelForSequenceClassification.from_pretrained(
            self.config["model_name"],
            num_labels=3,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model_cls.eval()
        self.model_cls.to(self.device)
        
        logger.info("FinBERT setup complete")
    
    def load_and_prepare_data(self):
        """Load data with minimal balancing"""
        logger.info("Loading data...")
        
        df = pd.read_csv(self.config["data_path"])
        
        # Show original distribution
        logger.info("Original class distribution:")
        class_counts = df['Sentiment'].value_counts()
        for sentiment, count in class_counts.items():
            percentage = 100 * count / len(df)
            logger.info(f"  {sentiment}: {count} ({percentage:.1f}%)")
        
        # Light balancing: only upsample negative to match positive (not neutral)
        negative_df = df[df['Sentiment'] == 'negative']
        positive_df = df[df['Sentiment'] == 'positive']
        neutral_df = df[df['Sentiment'] == 'neutral']
        
        # Target: make negative = positive count (don't touch neutral)
        target_count = len(positive_df)
        
        if len(negative_df) < target_count:
            negative_upsampled = negative_df.sample(
                target_count, 
                replace=True, 
                random_state=self.config["random_state"]
            )
            balanced_df = pd.concat([negative_upsampled, positive_df, neutral_df])
        else:
            balanced_df = df
        
        balanced_df = balanced_df.sample(frac=1, random_state=self.config["random_state"]).reset_index(drop=True)
        
        logger.info(f"Lightly balanced dataset: {len(balanced_df)} samples")
        balanced_counts = balanced_df['Sentiment'].value_counts()
        for sentiment, count in balanced_counts.items():
            percentage = 100 * count / len(balanced_df)
            logger.info(f"  {sentiment}: {count} ({percentage:.1f}%)")
        
        return balanced_df
    
    def extract_embeddings(self, sentences, batch_size=None):
        """Extract CLS embeddings from FinBERT"""
        if batch_size is None:
            batch_size = self.config["batch_size"]
            
        embeddings = []
        
        for i in tqdm(range(0, len(sentences), batch_size), desc="Extracting FinBERT embeddings"):
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
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return np.vstack(embeddings)
    
    def train_finbert(self, df):
        """Train FinBERT with class weights"""
        logger.info("Training FinBERT model...")
        
        sentences = df['Sentence'].tolist()
        labels = df['Sentiment'].map(self.config["label_mapping"]).values
        
        # Extract embeddings
        embeddings = self.extract_embeddings(sentences)
        X = embeddings  # Shape: (n_samples, 768)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels,
            test_size=self.config["test_size"],
            stratify=labels,
            random_state=self.config["random_state"]
        )
        
        # Compute class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
        
        logger.info("Class weights:")
        id2label = {v: k for k, v in self.config["label_mapping"].items()}
        for class_id, weight in class_weight_dict.items():
            logger.info(f"  {id2label[class_id]}: {weight:.3f}")
        
        # Create pipeline
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight=class_weight_dict,
                max_iter=1000,
                solver='lbfgs',
                random_state=self.config["random_state"]
            )
        )
        
        # Cross-validation
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
        
        # Train final model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average='macro')
        logger.info(f"Test macro F1: {test_f1:.4f}")
        
        # Classification report
        target_names = [id2label[i] for i in sorted(id2label.keys())]
        
        print("\nFinBERT Classification Report:")
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
        
        # Save pipeline
        joblib.dump(self.pipeline, output_dir / "finbert_pipeline.joblib")
        
        # Save tokenizer
        tokenizer_dir = output_dir / "finbert_tokenizer"
        self.tokenizer.save_pretrained(str(tokenizer_dir))
        
        # Save model
        model_dir = output_dir / "finbert_model"
        self.model_cls.save_pretrained(str(model_dir))
        
        # Save configuration
        model_info = {
            "config": self.config,
            "model_name": self.config["model_name"],
            "model_type": "finbert_embeddings",
            "embedding_dim": 768,
            "total_features": 768
        }
        
        with open(output_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"FinBERT model saved to {output_dir}")
    
    def run_training(self):
        """Main training workflow"""
        logger.info("Starting FinBERT training workflow...")
        
        # Setup model
        self.setup_finbert()
        
        # Load data
        df = self.load_and_prepare_data()
        
        # Train model
        results = self.train_finbert(df)
        
        # Save model
        self.save_model()
        
        logger.info("FinBERT training complete!")
        return results

def main():
    """Main function"""
    trainer = FinBERTTrainer()
    results = trainer.run_training()
    print(f"\nFinal Results: {results}")

if __name__ == "__main__":
    main()