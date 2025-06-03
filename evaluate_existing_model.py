#!/usr/bin/env python3
"""
Evaluate your existing trained FinBERT model to get real performance metrics
This will test your current model and generate metrics for your README
"""

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        self.model_path = "outputs/finbert_fixed_model"
        self.tokenizer_path = "outputs/finbert_fixed_tokenizer"
        self.data_path = "data/data.csv"
        
        self.model = None
        self.tokenizer = None
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
    def load_model(self):
        """Load your trained model and tokenizer"""
        print("🤖 Loading your trained model...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            
            print(f"   ✅ Model loaded from {self.model_path}")
            print(f"   ✅ Tokenizer loaded from {self.tokenizer_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            print("\n🔍 Available paths:")
            print(f"   Model path exists: {Path(self.model_path).exists()}")
            print(f"   Tokenizer path exists: {Path(self.tokenizer_path).exists()}")
            
            if Path(self.model_path).exists():
                model_files = list(Path(self.model_path).glob("*"))
                print(f"   Model files: {[f.name for f in model_files]}")
            
            return False
    
    def load_and_prepare_test_data(self):
        """Load and prepare test data"""
        print("\n📊 Loading test data...")
        
        df = pd.read_csv(self.data_path)
        print(f"   Total samples: {len(df)}")
        
        # Clean data same way as training
        df = df.dropna()
        df['Sentence'] = df['Sentence'].astype(str).str.strip()
        df = df[df['Sentence'].str.len() >= 10]
        
        # Standardize labels
        df['Sentiment'] = df['Sentiment'].str.lower().str.strip()
        df = df[df['Sentiment'].isin(self.label_map.keys())]
        df['labels'] = df['Sentiment'].map(self.label_map)
        
        print(f"   Cleaned samples: {len(df)}")
        print("   Label distribution:")
        for label, count in df['Sentiment'].value_counts().items():
            print(f"      {label}: {count} ({count/len(df)*100:.1f}%)")
        
        # Create test split (same as training would have used)
        _, test_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df['labels'], 
            random_state=42
        )
        
        print(f"   Test set size: {len(test_df)}")
        
        return test_df
    
    def predict_batch(self, texts, batch_size=16):
        """Make predictions on a batch of texts"""
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                predictions = torch.argmax(probs, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def evaluate_model(self, test_df):
        """Comprehensive model evaluation"""
        print("\n🧪 Evaluating model performance...")
        
        texts = test_df['Sentence'].tolist()
        true_labels = test_df['labels'].values
        
        # Make predictions
        pred_labels, probabilities = self.predict_batch(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        f1_macro = f1_score(true_labels, pred_labels, average='macro')
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
        
        print(f"   📊 Overall Performance:")
        print(f"      Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"      F1 Macro: {f1_macro:.4f}")
        print(f"      F1 Weighted: {f1_weighted:.4f}")
        
        # Detailed classification report
        target_names = ['negative', 'neutral', 'positive']
        report = classification_report(
            true_labels, pred_labels, 
            target_names=target_names,
            output_dict=True
        )
        
        print(f"\n   📋 Detailed Classification Report:")
        print(classification_report(true_labels, pred_labels, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': pred_labels.tolist(),
            'probabilities': probabilities.tolist(),
            'true_labels': true_labels.tolist()
        }
    
    def test_clear_examples(self):
        """Test on very obvious examples"""
        print("\n🎯 Testing on clear examples...")
        
        clear_examples = [
            ("Outstanding quarterly earnings with record profits and exceptional growth", "positive"),
            ("Devastating financial losses and bankruptcy filing announced", "negative"),
            ("Company reported quarterly results in line with expectations", "neutral"),
            ("Stock price soared 50% after amazing earnings beat consensus", "positive"),
            ("Major layoffs and plant closures due to terrible performance", "negative"),
            ("Revenue remained stable compared to the previous quarter", "neutral"),
            ("Incredible breakthrough in profitability and market expansion", "positive"),
            ("Massive debt crisis and potential company collapse looming", "negative")
        ]
        
        correct = 0
        results = []
        
        for text, expected in clear_examples:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            predicted_class = probs.argmax()
            prediction = self.id2label[predicted_class]
            confidence = probs[predicted_class]
            
            match = prediction == expected
            if match:
                correct += 1
            
            status = "✅" if match else "❌"
            print(f"   {status} '{text[:60]}...'")
            print(f"      → {prediction} ({confidence:.3f}) | Expected: {expected}")
            
            results.append({
                'text': text,
                'predicted': prediction,
                'expected': expected,
                'confidence': confidence,
                'correct': match
            })
        
        clear_accuracy = correct / len(clear_examples)
        print(f"\n   🎯 Clear Examples Accuracy: {correct}/{len(clear_examples)} ({clear_accuracy*100:.1f}%)")
        
        return clear_accuracy, results
    
    def create_confusion_matrix_plot(self, cm):
        """Create confusion matrix visualization"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive']
        )
        plt.title('Confusion Matrix - FinBERT Financial Sentiment')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   📊 Confusion matrix saved to outputs/confusion_matrix.png")
    
    def save_results(self, metrics, clear_results):
        """Save evaluation results"""
        print("\n💾 Saving evaluation results...")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        # Create comprehensive results
        results = {
            'model_info': {
                'model_path': self.model_path,
                'tokenizer_path': self.tokenizer_path,
                'model_type': 'finbert_financial_sentiment'
            },
            'performance': {
                'accuracy': float(metrics['accuracy']),
                'f1_macro': float(metrics['f1_macro']),
                'f1_weighted': float(metrics['f1_weighted'])
            },
            'detailed_metrics': convert_numpy_types(metrics['classification_report']),
            'clear_examples': {
                'accuracy': float(clear_results[0]),
                'results': convert_numpy_types(clear_results[1])
            },
            'evaluation_date': pd.Timestamp.now().isoformat()
        }
        
        # Save to JSON
        with open('outputs/model_evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create README metrics snippet
        readme_snippet = f"""
## 📊 Model Performance

Our fine-tuned FinBERT model achieves the following performance on financial sentiment analysis:

| Metric | Score |
|--------|-------|
| Accuracy | {metrics['accuracy']:.1%} |
| F1 Score (Macro) | {metrics['f1_macro']:.3f} |
| F1 Score (Weighted) | {metrics['f1_weighted']:.3f} |

### Detailed Performance by Class

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | {metrics['classification_report']['negative']['precision']:.3f} | {metrics['classification_report']['negative']['recall']:.3f} | {metrics['classification_report']['negative']['f1-score']:.3f} |
| Neutral | {metrics['classification_report']['neutral']['precision']:.3f} | {metrics['classification_report']['neutral']['recall']:.3f} | {metrics['classification_report']['neutral']['f1-score']:.3f} |
| Positive | {metrics['classification_report']['positive']['precision']:.3f} | {metrics['classification_report']['positive']['recall']:.3f} | {metrics['classification_report']['positive']['f1-score']:.3f} |

**Clear Examples Accuracy**: {clear_results[0]:.1%} (performance on obviously positive/negative financial statements)

*Evaluation conducted on {len(metrics['true_labels'])} test samples using stratified split.*
"""
        
        with open('outputs/readme_metrics.md', 'w') as f:
            f.write(readme_snippet)
        
        print(f"   ✅ Results saved to outputs/model_evaluation.json")
        print(f"   ✅ README snippet saved to outputs/readme_metrics.md")
        
        return results
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🚀 FinBERT Model Evaluation")
        print("=" * 50)
        
        # Load model
        if not self.load_model():
            return None
        
        # Load test data
        test_df = self.load_and_prepare_test_data()
        
        # Evaluate model
        metrics = self.evaluate_model(test_df)
        
        # Test clear examples
        clear_accuracy, clear_results = self.test_clear_examples()
        
        # Create visualizations
        self.create_confusion_matrix_plot(np.array(metrics['confusion_matrix']))
        
        # Save results
        results = self.save_results(metrics, (clear_accuracy, clear_results))
        
        print("\n🎉 Evaluation completed!")
        print(f"   📊 Overall Accuracy: {metrics['accuracy']:.1%}")
        print(f"   📊 F1 Score: {metrics['f1_macro']:.3f}")
        print(f"   🎯 Clear Examples: {clear_accuracy:.1%}")
        print("\n📋 Next steps:")
        print("   1. Copy content from outputs/readme_metrics.md to your README")
        print("   2. Include outputs/confusion_matrix.png in your documentation")
        print("   3. Use these metrics for your publication")
        
        return results

def main():
    evaluator = ModelEvaluator()
    results = evaluator.run_evaluation()
    
    if results:
        print("\n✅ Real metrics generated successfully!")
    else:
        print("\n❌ Evaluation failed - check your model paths")

if __name__ == "__main__":
    main()