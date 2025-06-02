#!/usr/bin/env python3
"""
Quick and reliable model retraining script
Uses proven techniques to ensure proper training
"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
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
from pathlib import Path

def check_data_quality():
    """Check and clean the training data"""
    print("🔍 Checking data quality...")
    
    df = pd.read_csv("data/data.csv")
    print(f"   Original size: {len(df)}")
    
    # Check label distribution
    print("   Label distribution:")
    label_counts = df['Sentiment'].value_counts()
    print(label_counts)
    
    # Check for issues
    issues = []
    if label_counts.min() < 100:
        issues.append(f"Small class: {label_counts.idxmin()} has only {label_counts.min()} samples")
    
    if len(df['Sentiment'].unique()) != 3:
        issues.append(f"Expected 3 classes, found: {list(df['Sentiment'].unique())}")
    
    # Check text quality
    df['text_length'] = df['Sentence'].astype(str).str.len()
    short_texts = len(df[df['text_length'] < 10])
    if short_texts > 0:
        issues.append(f"{short_texts} texts are very short (< 10 chars)")
    
    if issues:
        print("   ⚠️ Data issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("   ✅ Data looks good")
    
    return df

def create_balanced_dataset(df, test_size=0.2):
    """Create a balanced, clean dataset"""
    print("\n📊 Creating balanced dataset...")
    
    # Clean data
    df = df.dropna()
    df['Sentence'] = df['Sentence'].astype(str).str.strip()
    df = df[df['Sentence'].str.len() >= 10]  # Remove very short texts
    
    # Ensure we have standard labels
    label_map = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    # Map any variations
    df['Sentiment'] = df['Sentiment'].str.lower().str.strip()
    df = df[df['Sentiment'].isin(label_map.keys())]
    df['labels'] = df['Sentiment'].map(label_map)
    
    print(f"   Cleaned size: {len(df)}")
    print(f"   Final distribution: {df['Sentiment'].value_counts().to_dict()}")
    
    # Split stratified
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['labels'], 
        random_state=42
    )
    
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}")
    
    return train_df, val_df, label_map

def train_finbert_from_scratch():
    """Train FinBERT with minimal but effective approach"""
    print("\n🚀 Training FinBERT from scratch...")
    
    # Check data
    df = check_data_quality()
    train_df, val_df, label_map = create_balanced_dataset(df)
    
    # Use base BERT model to avoid pre-existing issues
    model_name = "bert-base-uncased"  # Start fresh
    
    print(f"   Using base model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: 'negative', 1: 'neutral', 2: 'positive'},
        label2id={'negative': 0, 'neutral': 1, 'positive': 2}
    )
    
    # Prepare datasets
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=256  # Shorter for faster training
        )
    
    train_dataset = Dataset.from_dict({
        'text': train_df['Sentence'].tolist(),
        'labels': train_df['labels'].tolist()
    })
    
    val_dataset = Dataset.from_dict({
        'text': val_df['Sentence'].tolist(),
        'labels': val_df['labels'].tolist()
    })
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Conservative training arguments
    training_args = TrainingArguments(
        output_dir="outputs/finbert_fixed_training",
        num_train_epochs=2,  # Fewer epochs to avoid overfitting
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,  # Lower learning rate
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Add early stopping to prevent overfitting
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )
    
    # Train
    print("   🏋️ Starting training...")
    trainer.train()
    
    # Evaluate
    print("\n📊 Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"   Validation loss: {eval_results['eval_loss']:.3f}")
    
    # Test predictions
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"   Validation accuracy: {accuracy:.3f}")
    
    if accuracy > 0.7:  # Only save if decent performance
        # Save model and tokenizer
        output_dir = "outputs/finbert_fixed"
        model.save_pretrained(f"{output_dir}_model")
        tokenizer.save_pretrained(f"{output_dir}_tokenizer")
        
        print(f"   ✅ Model saved to {output_dir}_model")
        
        # Test on clear examples
        test_clear_examples(model, tokenizer)
        
        return True
    else:
        print(f"   ❌ Accuracy too low ({accuracy:.3f}), not saving")
        return False

def test_clear_examples(model, tokenizer):
    """Test on very clear sentiment examples"""
    print("\n🧪 Testing on clear examples...")
    
    test_cases = [
        ("This is excellent news with outstanding profits and growth", "positive"),
        ("Terrible losses and bankruptcy filing", "negative"),
        ("Results were as expected with no changes", "neutral"),
        ("Stock price surged 50% on amazing earnings beat", "positive"),
        ("Company failed and lost everything", "negative")
    ]
    
    model.eval()
    correct = 0
    
    for text, expected in test_cases:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        predicted_class = probs.argmax()
        labels = ['negative', 'neutral', 'positive']
        prediction = labels[predicted_class]
        confidence = probs[predicted_class]
        
        match = prediction == expected
        if match:
            correct += 1
        
        status = "✅" if match else "❌"
        print(f"   {status} '{text[:50]}...' → {prediction} ({confidence:.3f})")
    
    print(f"\n   Clear examples accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")

def main():
    print("🔄 Quick Model Retraining")
    print("=" * 50)
    
    # Check if we should retrain
    print("This will retrain FinBERT from scratch using BERT-base.")
    print("It should take about 10-15 minutes.")
    
    response = input("\nProceed with retraining? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Clear any GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train
    success = train_finbert_from_scratch()
    
    if success:
        print("\n🎉 Retraining successful!")
        print("\n📋 Next steps:")
        print("   1. Update your API to use: outputs/finbert_fixed_model")
        print("   2. Test the new model")
        print("   3. Compare with old model performance")
    else:
        print("\n❌ Retraining failed. Check the data quality.")

if __name__ == "__main__":
    main()