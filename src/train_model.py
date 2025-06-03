#!/usr/bin/env python3
"""
FinBERT Fine-tuning Script for Financial Sentiment Analysis
Production-ready training pipeline with proper validation and error handling
"""

import pandas as pd
import torch
import logging
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinBERTTrainer:
    """Production-ready FinBERT training pipeline"""
    
    def __init__(self, config_path="training_config.json"):
        self.config = self.load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        
    def load_config(self, config_path):
        """Load training configuration with defaults"""
        default_config = {
            "data_path": "data/data.csv",
            "model_name": "bert-base-uncased",
            "output_dir": "outputs/finbert_fixed",
            "max_length": 256,
            "test_size": 0.2,
            "num_epochs": 2,
            "batch_size": 8,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "early_stopping_patience": 3,
            "min_accuracy_threshold": 0.7,
            "random_state": 42
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def check_data_quality(self):
        """Comprehensive data quality check"""
        logger.info("🔍 Checking data quality...")
        
        df = pd.read_csv(self.config["data_path"])
        logger.info(f"   Original size: {len(df)}")
        
        # Check required columns
        required_cols = ['Sentence', 'Sentiment']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check label distribution
        logger.info("   Label distribution:")
        label_counts = df['Sentiment'].value_counts()
        for label, count in label_counts.items():
            percentage = count / len(df) * 100
            logger.info(f"      {label}: {count} ({percentage:.1f}%)")
        
        # Data quality checks
        issues = []
        
        # Check class balance
        if label_counts.min() < 50:
            issues.append(f"Small class: {label_counts.idxmin()} has only {label_counts.min()} samples")
        
        # Check valid labels
        expected_labels = set(self.label_map.keys())
        actual_labels = set(df['Sentiment'].str.lower().str.strip().unique())
        invalid_labels = actual_labels - expected_labels
        if invalid_labels:
            issues.append(f"Invalid labels found: {invalid_labels}")
        
        # Check text quality
        df['text_length'] = df['Sentence'].astype(str).str.len()
        short_texts = len(df[df['text_length'] < 10])
        long_texts = len(df[df['text_length'] > 1000])
        
        if short_texts > len(df) * 0.1:
            issues.append(f"{short_texts} texts are very short (< 10 chars)")
        
        if long_texts > 0:
            logger.info(f"   Found {long_texts} long texts (> 1000 chars) - will be truncated")
        
        # Report issues
        if issues:
            logger.warning("   ⚠️ Data issues found:")
            for issue in issues:
                logger.warning(f"      - {issue}")
        else:
            logger.info("   ✅ Data quality looks good")
        
        return df
    
    def create_balanced_dataset(self, df):
        """Create clean, balanced dataset for training"""
        logger.info("📊 Creating balanced dataset...")
        
        # Clean data
        df = df.dropna()
        df['Sentence'] = df['Sentence'].astype(str).str.strip()
        df = df[df['Sentence'].str.len() >= 10]  # Remove very short texts
        
        # Standardize labels
        df['Sentiment'] = df['Sentiment'].str.lower().str.strip()
        df = df[df['Sentiment'].isin(self.label_map.keys())]
        df['labels'] = df['Sentiment'].map(self.label_map)
        
        logger.info(f"   Cleaned size: {len(df)}")
        
        # Log final distribution
        final_dist = df['Sentiment'].value_counts()
        for label, count in final_dist.items():
            percentage = count / len(df) * 100
            logger.info(f"      {label}: {count} ({percentage:.1f}%)")
        
        # Stratified split
        train_df, val_df = train_test_split(
            df, 
            test_size=self.config["test_size"], 
            stratify=df['labels'], 
            random_state=self.config["random_state"]
        )
        
        logger.info(f"   Train: {len(train_df)}, Validation: {len(val_df)}")
        
        return train_df, val_df
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"🤖 Loading model: {self.config['model_name']}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model_name"],
            num_labels=3,
            id2label={0: 'negative', 1: 'neutral', 2: 'positive'},
            label2id=self.label_map
        )
        
        logger.info("   ✅ Model and tokenizer loaded")
    
    def prepare_datasets(self, train_df, val_df):
        """Prepare HuggingFace datasets"""
        logger.info("📚 Preparing datasets...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.config["max_length"]
            )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': train_df['Sentence'].tolist(),
            'labels': train_df['labels'].tolist()
        })
        
        val_dataset = Dataset.from_dict({
            'text': val_df['Sentence'].tolist(),
            'labels': val_df['labels'].tolist()
        })
        
        # Tokenize
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        logger.info(f"   ✅ Datasets prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train_model(self, train_dataset, val_dataset):
        """Train the model with optimal settings"""
        logger.info("🏋️ Starting model training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.config['output_dir']}_training",
            num_train_epochs=self.config["num_epochs"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"] * 2,
            learning_rate=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            warmup_steps=self.config["warmup_steps"],
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            report_to=None  # Disable wandb/tensorboard
        )
        
        # Data collator and callbacks
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config["early_stopping_patience"]
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[early_stopping],
        )
        
        # Train
        trainer.train()
        
        return trainer
    
    def evaluate_model(self, trainer, val_dataset):
        """Comprehensive model evaluation"""
        logger.info("📊 Evaluating model performance...")
        
        # Basic evaluation
        eval_results = trainer.evaluate()
        logger.info(f"   Validation loss: {eval_results['eval_loss']:.4f}")
        
        # Detailed predictions
        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   F1 Macro: {f1_macro:.4f}")
        logger.info(f"   F1 Weighted: {f1_weighted:.4f}")
        
        # Classification report
        target_names = ['negative', 'neutral', 'positive']
        report = classification_report(y_true, y_pred, target_names=target_names)
        logger.info(f"\n📋 Classification Report:\n{report}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'validation_loss': eval_results['eval_loss'],
            'classification_report': report
        }
    
    def test_clear_examples(self):
        """Test model on obvious examples"""
        logger.info("🧪 Testing on clear examples...")
        
        test_cases = [
            ("Outstanding quarterly results with record profits and strong growth", "positive"),
            ("Devastating losses and bankruptcy filing announced today", "negative"),
            ("Company reported results in line with market expectations", "neutral"),
            ("Stock price soared 50% after amazing earnings beat", "positive"),
            ("Major layoffs and plant closures due to poor performance", "negative"),
            ("Quarterly revenue remained stable compared to last year", "neutral")
        ]
        
        self.model.eval()
        correct = 0
        
        for text, expected in test_cases:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            predicted_class = probs.argmax()
            labels = ['negative', 'neutral', 'positive']
            prediction = labels[predicted_class]
            confidence = probs[predicted_class]
            
            match = prediction == expected
            if match:
                correct += 1
            
            status = "✅" if match else "❌"
            logger.info(f"   {status} '{text[:60]}...' → {prediction} ({confidence:.3f})")
        
        clear_accuracy = correct / len(test_cases)
        logger.info(f"\n   Clear examples accuracy: {correct}/{len(test_cases)} ({clear_accuracy*100:.1f}%)")
        
        return clear_accuracy
    
    def save_model(self, metrics):
        """Save model with metadata"""
        output_model_dir = f"{self.config['output_dir']}_model"
        output_tokenizer_dir = f"{self.config['output_dir']}_tokenizer"
        
        # Save model and tokenizer
        self.model.save_pretrained(output_model_dir)
        self.tokenizer.save_pretrained(output_tokenizer_dir)
        
        # Save training metadata
        metadata = {
            "model_name": self.config["model_name"],
            "training_config": self.config,
            "performance": metrics,
            "label_mapping": self.label_map,
            "model_type": "finbert_financial_sentiment",
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        with open(f"{self.config['output_dir']}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"   ✅ Model saved to {output_model_dir}")
        logger.info(f"   ✅ Tokenizer saved to {output_tokenizer_dir}")
        logger.info(f"   ✅ Metadata saved to {self.config['output_dir']}_metadata.json")
    
    def run_training(self):
        """Complete training pipeline"""
        logger.info("🚀 Starting FinBERT training pipeline...")
        
        try:
            # Check data
            df = self.check_data_quality()
            
            # Prepare datasets
            train_df, val_df = self.create_balanced_dataset(df)
            
            # Setup model
            self.setup_model_and_tokenizer()
            
            # Prepare datasets
            train_dataset, val_dataset = self.prepare_datasets(train_df, val_df)
            
            # Train
            trainer = self.train_model(train_dataset, val_dataset)
            
            # Evaluate
            metrics = self.evaluate_model(trainer, val_dataset)
            
            # Test clear examples
            clear_accuracy = self.test_clear_examples()
            metrics['clear_examples_accuracy'] = clear_accuracy
            
            # Check if model is good enough to save
            if metrics['accuracy'] >= self.config["min_accuracy_threshold"]:
                self.save_model(metrics)
                logger.info("🎉 Training completed successfully!")
                return metrics
            else:
                logger.warning(f"❌ Model accuracy {metrics['accuracy']:.3f} below threshold {self.config['min_accuracy_threshold']}")
                logger.warning("   Model not saved. Consider adjusting hyperparameters or data.")
                return None
                
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise

def main():
    """Main training function"""
    print("🔄 FinBERT Financial Sentiment Training")
    print("=" * 50)
    
    # Check if we should train
    print("This will fine-tune BERT for financial sentiment analysis.")
    print("Training typically takes 10-20 minutes depending on data size.")
    
    response = input("\nProceed with training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("🖥️  Using CPU (training will be slower)")
    
    # Create trainer and run
    trainer = FinBERTTrainer()
    
    try:
        results = trainer.run_training()
        
        if results:
            print("\n🎉 Training completed successfully!")
            print(f"   Final accuracy: {results['accuracy']:.3f}")
            print(f"   F1 macro: {results['f1_macro']:.3f}")
            print("\n📋 Next steps:")
            print("   1. Test the API with the new model")
            print("   2. Run comprehensive tests: python test_api.py")
            print("   3. Deploy with: ./deployment.sh")
        else:
            print("\n❌ Training failed - model performance below threshold")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")

if __name__ == "__main__":
    main()