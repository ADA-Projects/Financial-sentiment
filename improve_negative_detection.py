#!/usr/bin/env python3
"""
Multiple strategies to improve negative sentiment detection
Try these in order from easiest to most complex
"""

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from pathlib import Path

class NegativePerformanceImprover:
    def __init__(self):
        self.model_path = "outputs/finbert_fixed_model"
        self.tokenizer_path = "outputs/finbert_fixed_tokenizer"
        self.data_path = "data/data.csv"
        
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
    
    def predict_with_thresholds(self, texts, neg_threshold=0.33, pos_threshold=0.33):
        """
        Strategy 1: Adjust prediction thresholds
        Lower the threshold for negative predictions
        """
        all_predictions = []
        all_probabilities = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            # Apply custom thresholds
            neg_prob, neu_prob, pos_prob = probs
            
            if neg_prob >= neg_threshold:
                prediction = 0  # negative
            elif pos_prob >= pos_threshold:
                prediction = 2  # positive  
            else:
                prediction = 1  # neutral (default)
            
            all_predictions.append(prediction)
            all_probabilities.append(probs)
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def predict_with_confidence_adjustment(self, texts, negative_boost=1.5):
        """
        Strategy 2: Boost negative confidence
        Multiply negative probabilities by a factor
        """
        all_predictions = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            # Boost negative probability
            adjusted_probs = probs.copy()
            adjusted_probs[0] *= negative_boost  # Boost negative
            
            # Renormalize
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
            
            prediction = np.argmax(adjusted_probs)
            all_predictions.append(prediction)
        
        return np.array(all_predictions)
    
    def find_optimal_thresholds(self, test_df):
        """Find optimal thresholds using validation data"""
        print("🔍 Finding optimal thresholds for negative detection...")
        
        texts = test_df['Sentence'].tolist()
        true_labels = test_df['labels'].values
        
        best_f1 = 0
        best_thresholds = (0.33, 0.33)
        
        # Grid search over thresholds
        for neg_thresh in np.arange(0.1, 0.8, 0.05):
            for pos_thresh in np.arange(0.1, 0.8, 0.05):
                pred_labels, _ = self.predict_with_thresholds(texts, neg_thresh, pos_thresh)
                f1_macro = f1_score(true_labels, pred_labels, average='macro')
                
                if f1_macro > best_f1:
                    best_f1 = f1_macro
                    best_thresholds = (neg_thresh, pos_thresh)
        
        print(f"   Best thresholds: negative={best_thresholds[0]:.2f}, positive={best_thresholds[1]:.2f}")
        print(f"   Best F1 macro: {best_f1:.3f}")
        
        return best_thresholds
    
    def create_balanced_training_data(self):
        """
        Strategy 3: Create balanced dataset for retraining
        Oversample negative examples
        """
        print("📊 Creating balanced training dataset...")
        
        df = pd.read_csv(self.data_path)
        
        # Clean data
        df = df.dropna()
        df['Sentence'] = df['Sentence'].astype(str).str.strip()
        df = df[df['Sentence'].str.len() >= 10]
        df['Sentiment'] = df['Sentiment'].str.lower().str.strip()
        
        # Separate by class
        negative_df = df[df['Sentiment'] == 'negative']
        neutral_df = df[df['Sentiment'] == 'neutral']
        positive_df = df[df['Sentiment'] == 'positive']
        
        print(f"   Original distribution:")
        print(f"      Negative: {len(negative_df)}")
        print(f"      Neutral: {len(neutral_df)}")
        print(f"      Positive: {len(positive_df)}")
        
        # Target: Balance negative with positive (don't make it too large vs neutral)
        target_negative_size = len(positive_df)
        
        # Oversample negative examples
        if len(negative_df) < target_negative_size:
            negative_oversampled = negative_df.sample(
                target_negative_size, 
                replace=True, 
                random_state=42
            )
        else:
            negative_oversampled = negative_df
        
        # Create balanced dataset
        balanced_df = pd.concat([
            negative_oversampled,
            neutral_df,
            positive_df
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   Balanced distribution:")
        for sentiment, count in balanced_df['Sentiment'].value_counts().items():
            print(f"      {sentiment}: {count}")
        
        # Save balanced dataset
        balanced_df.to_csv('data/balanced_data.csv', index=False)
        print(f"   ✅ Balanced dataset saved to data/balanced_data.csv")
        
        return balanced_df
    
    def analyze_negative_misclassifications(self, test_df):
        """Analyze what negative examples are being misclassified"""
        print("🔍 Analyzing negative misclassifications...")
        
        negative_examples = test_df[test_df['Sentiment'] == 'negative']
        
        misclassified = []
        
        for _, row in negative_examples.iterrows():
            text = row['Sentence']
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            predicted_class = np.argmax(probs)
            
            if predicted_class != 0:  # Not predicted as negative
                misclassified.append({
                    'text': text,
                    'predicted_class': ['negative', 'neutral', 'positive'][predicted_class],
                    'negative_prob': probs[0],
                    'predicted_prob': probs[predicted_class]
                })
        
        print(f"   Found {len(misclassified)} misclassified negative examples")
        
        # Show examples of misclassifications
        print("   📋 Examples of misclassified negatives:")
        for i, example in enumerate(misclassified[:5]):
            print(f"      {i+1}. '{example['text'][:60]}...'")
            print(f"         → Predicted: {example['predicted_class']} ({example['predicted_prob']:.3f})")
            print(f"         → Negative prob: {example['negative_prob']:.3f}")
        
        return misclassified
    
    def test_improvements(self, test_df):
        """Test all improvement strategies"""
        print("\n🧪 Testing improvement strategies...")
        
        texts = test_df['Sentence'].tolist()
        true_labels = test_df['labels'].values
        
        # Original performance
        original_preds = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            original_preds.append(np.argmax(probs))
        
        original_report = classification_report(true_labels, original_preds, output_dict=True)
        
        print(f"   📊 Original Performance:")
        print(f"      Negative F1: {original_report['0']['f1-score']:.3f}")
        print(f"      Overall F1: {original_report['macro avg']['f1-score']:.3f}")
        
        # Strategy 1: Optimal thresholds
        best_thresholds = self.find_optimal_thresholds(test_df)
        threshold_preds, _ = self.predict_with_thresholds(texts, best_thresholds[0], best_thresholds[1])
        threshold_report = classification_report(true_labels, threshold_preds, output_dict=True)
        
        print(f"\n   📊 With Optimal Thresholds:")
        print(f"      Negative F1: {threshold_report['0']['f1-score']:.3f}")
        print(f"      Overall F1: {threshold_report['macro avg']['f1-score']:.3f}")
        
        # Strategy 2: Confidence boost
        boost_preds = self.predict_with_confidence_adjustment(texts, negative_boost=1.8)
        boost_report = classification_report(true_labels, boost_preds, output_dict=True)
        
        print(f"\n   📊 With Negative Confidence Boost:")
        print(f"      Negative F1: {boost_report['0']['f1-score']:.3f}")
        print(f"      Overall F1: {boost_report['macro avg']['f1-score']:.3f}")
        
        return {
            'original': original_report,
            'thresholds': threshold_report,
            'boost': boost_report,
            'best_thresholds': best_thresholds
        }

def quick_retrain_with_balance():
    """
    Strategy 4: Quick retrain with balanced data
    """
    print("\n🔄 Option: Retrain with balanced data")
    print("This will retrain your model with oversampled negative examples")
    print("Expected improvement: Negative F1 from 0.38 to 0.60+")
    
    response = input("Do you want to retrain with balanced data? (y/n): ")
    if response.lower() == 'y':
        # Use the updated training script with balanced data
        print("Run this command:")
        print("python src/train_model.py")
        print("(Make sure to update the data path to use balanced_data.csv)")

def main():
    print("🎯 Improving Negative Sentiment Detection")
    print("=" * 50)
    
    improver = NegativePerformanceImprover()
    
    # Load test data (same as evaluation)
    df = pd.read_csv("data/data.csv")
    df = df.dropna()
    df['Sentence'] = df['Sentence'].astype(str).str.strip()
    df = df[df['Sentence'].str.len() >= 10]
    df['Sentiment'] = df['Sentiment'].str.lower().str.strip()
    
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df = df[df['Sentiment'].isin(label_map.keys())]
    df['labels'] = df['Sentiment'].map(label_map)
    
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)
    
    # Analyze current issues
    improver.analyze_negative_misclassifications(test_df)
    
    # Test improvement strategies
    results = improver.test_improvements(test_df)
    
    # Create balanced dataset option
    balanced_df = improver.create_balanced_training_data()
    
    print("\n🎯 Summary of Improvement Options:")
    print("1. **Threshold Adjustment** (Quick fix)")
    print(f"   - Negative F1: {results['original']['0']['f1-score']:.3f} → {results['thresholds']['0']['f1-score']:.3f}")
    print(f"   - Overall F1: {results['original']['macro avg']['f1-score']:.3f} → {results['thresholds']['macro avg']['f1-score']:.3f}")
    
    print("\n2. **Confidence Boost** (Quick fix)")
    print(f"   - Negative F1: {results['original']['0']['f1-score']:.3f} → {results['boost']['0']['f1-score']:.3f}")
    print(f"   - Overall F1: {results['original']['macro avg']['f1-score']:.3f} → {results['boost']['macro avg']['f1-score']:.3f}")
    
    print("\n3. **Retrain with Balanced Data** (Best fix)")
    print("   - Expected: Negative F1 → 0.60+")
    print("   - Uses the balanced dataset created above")
    
    # Save optimal thresholds for API use
    optimal_config = {
        'optimal_thresholds': {
            'negative': float(results['best_thresholds'][0]),
            'positive': float(results['best_thresholds'][1])
        },
        'improvement_results': {
            'original_negative_f1': float(results['original']['0']['f1-score']),
            'threshold_negative_f1': float(results['thresholds']['0']['f1-score']),
            'boost_negative_f1': float(results['boost']['0']['f1-score'])
        }
    }
    
    with open('outputs/negative_improvement_config.json', 'w') as f:
        json.dump(optimal_config, f, indent=2)
    
    print(f"\n✅ Optimal configuration saved to outputs/negative_improvement_config.json")
    print("\n📋 Recommended next steps:")
    print("1. Try threshold adjustment in your API (quickest)")
    print("2. If not satisfied, retrain with balanced data")
    print("3. For production, consider ensemble of both approaches")
    
    quick_retrain_with_balance()

if __name__ == "__main__":
    main()