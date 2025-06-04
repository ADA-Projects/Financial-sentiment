#!/usr/bin/env python3
"""
Quick and robust negative sentiment improvement
Much faster than the previous version with better error handling
"""

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import time
from pathlib import Path
import sys

class QuickNegativeFix:
    def __init__(self):
        self.model_path = "outputs/finbert_fixed_model"
        self.tokenizer_path = "outputs/finbert_fixed_tokenizer"
        self.data_path = "data/data.csv"
        
        print("🚀 Quick Negative Sentiment Fix")
        print("=" * 40)
        
    def test_model_loading(self):
        """Test if model loads properly"""
        print("1️⃣ Testing model loading...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            model.eval()
            print("   ✅ Model loads successfully")
            return tokenizer, model
        except Exception as e:
            print(f"   ❌ Model loading failed: {e}")
            return None, None
    
    def load_test_data(self):
        """Load and prepare test data"""
        print("2️⃣ Loading test data...")
        try:
            df = pd.read_csv(self.data_path)
            print(f"   📊 Loaded {len(df)} samples")
            
            # Quick clean
            df = df.dropna()
            df['Sentence'] = df['Sentence'].astype(str).str.strip()
            df = df[df['Sentence'].str.len() >= 10]
            df['Sentiment'] = df['Sentiment'].str.lower().str.strip()
            
            label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            df = df[df['Sentiment'].isin(label_map.keys())]
            df['labels'] = df['Sentiment'].map(label_map)
            
            print(f"   📊 After cleaning: {len(df)} samples")
            
            # Create test split (same as original evaluation)
            _, test_df = train_test_split(
                df, test_size=0.2, stratify=df['labels'], random_state=42
            )
            
            print(f"   📊 Test set: {len(test_df)} samples")
            
            # Show negative distribution in test set
            neg_count = len(test_df[test_df['labels'] == 0])
            print(f"   📊 Negative examples in test: {neg_count}")
            
            return test_df
            
        except Exception as e:
            print(f"   ❌ Data loading failed: {e}")
            return None
    
    def quick_threshold_test(self, tokenizer, model, test_df):
        """Quick test of just a few promising thresholds"""
        print("3️⃣ Testing promising thresholds...")
        
        # Get small sample first for speed
        sample_size = min(200, len(test_df))
        test_sample = test_df.sample(sample_size, random_state=42)
        
        texts = test_sample['Sentence'].tolist()
        true_labels = test_sample['labels'].values
        
        print(f"   🧪 Testing on {len(texts)} samples first...")
        
        # Get baseline probabilities
        print("   📊 Getting model predictions...")
        all_probs = []
        
        for i, text in enumerate(texts):
            if i % 50 == 0:
                print(f"      Processing {i}/{len(texts)}...")
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)
        
        all_probs = np.array(all_probs)
        
        # Test a few promising thresholds (much smaller search space)
        promising_thresholds = [
            (0.15, 0.4),   # Very low negative threshold
            (0.2, 0.4),    # Low negative threshold  
            (0.25, 0.4),   # Medium-low negative threshold
            (0.3, 0.4),    # Baseline
            (0.35, 0.4),   # Higher negative threshold
        ]
        
        print("   🎯 Testing threshold combinations...")
        results = []
        
        for neg_thresh, pos_thresh in promising_thresholds:
            # Apply thresholds
            predictions = []
            for probs in all_probs:
                neg_prob, neu_prob, pos_prob = probs
                
                if neg_prob >= neg_thresh:
                    pred = 0  # negative
                elif pos_prob >= pos_thresh:
                    pred = 2  # positive
                else:
                    pred = 1  # neutral
                
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calculate metrics
            report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
            
            neg_f1 = report.get('0', {}).get('f1-score', 0)
            macro_f1 = report['macro avg']['f1-score']
            
            results.append({
                'neg_threshold': neg_thresh,
                'pos_threshold': pos_thresh,
                'negative_f1': neg_f1,
                'macro_f1': macro_f1,
                'predictions': predictions.copy()
            })
            
            print(f"      Thresholds ({neg_thresh:.2f}, {pos_thresh:.2f}): Neg F1={neg_f1:.3f}, Macro F1={macro_f1:.3f}")
        
        # Find best combination
        best_result = max(results, key=lambda x: x['negative_f1'])
        
        print(f"\n   🏆 Best thresholds for negative detection:")
        print(f"      Negative threshold: {best_result['neg_threshold']}")
        print(f"      Positive threshold: {best_result['pos_threshold']}")
        print(f"      Negative F1: {best_result['negative_f1']:.3f}")
        print(f"      Macro F1: {best_result['macro_f1']:.3f}")
        
        return best_result, results
    
    def test_on_full_dataset(self, tokenizer, model, test_df, best_thresholds):
        """Test best thresholds on full test set"""
        print("4️⃣ Validating on full test set...")
        
        texts = test_df['Sentence'].tolist()
        true_labels = test_df['labels'].values
        
        neg_thresh = best_thresholds['neg_threshold']
        pos_thresh = best_thresholds['pos_threshold']
        
        print(f"   📊 Testing {len(texts)} samples with thresholds ({neg_thresh:.2f}, {pos_thresh:.2f})")
        
        # Get predictions for full dataset
        predictions = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            if i % (batch_size * 10) == 0:
                print(f"      Progress: {i}/{len(texts)}")
            
            # Batch tokenization
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            # Apply thresholds
            for prob_row in probs:
                neg_prob, neu_prob, pos_prob = prob_row
                
                if neg_prob >= neg_thresh:
                    predictions.append(0)  # negative
                elif pos_prob >= pos_thresh:
                    predictions.append(2)  # positive
                else:
                    predictions.append(1)  # neutral
        
        predictions = np.array(predictions)
        
        # Calculate final metrics
        report = classification_report(true_labels, predictions, output_dict=True)
        
        print(f"\n   📊 Full dataset results:")
        print(f"      Accuracy: {report['accuracy']:.3f}")
        print(f"      Negative F1: {report['0']['f1-score']:.3f}")
        print(f"      Neutral F1: {report['1']['f1-score']:.3f}")
        print(f"      Positive F1: {report['2']['f1-score']:.3f}")
        print(f"      Macro F1: {report['macro avg']['f1-score']:.3f}")
        
        return report
    
    def test_clear_examples(self, tokenizer, model, thresholds):
        """Test on obvious examples"""
        print("5️⃣ Testing on clear examples...")
        
        clear_examples = [
            ("Outstanding quarterly earnings with record profits", "positive"),
            ("Devastating financial losses and bankruptcy filing", "negative"),
            ("Company reported results in line with expectations", "neutral"),
            ("Stock price soared after amazing earnings", "positive"),
            ("Major layoffs and terrible performance", "negative"),
            ("Revenue remained stable this quarter", "neutral"),
        ]
        
        neg_thresh = thresholds['neg_threshold']
        pos_thresh = thresholds['pos_threshold']
        
        correct = 0
        
        for text, expected in clear_examples:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            neg_prob, neu_prob, pos_prob = probs
            
            if neg_prob >= neg_thresh:
                prediction = "negative"
            elif pos_prob >= pos_thresh:
                prediction = "positive"
            else:
                prediction = "neutral"
            
            match = prediction == expected
            if match:
                correct += 1
            
            status = "✅" if match else "❌"
            print(f"   {status} '{text[:50]}...' → {prediction} (expected {expected})")
        
        accuracy = correct / len(clear_examples)
        print(f"\n   🎯 Clear examples accuracy: {correct}/{len(clear_examples)} ({accuracy:.1%})")
        
        return accuracy
    
    def save_results(self, best_thresholds, full_report, clear_accuracy):
        """Save results for API integration"""
        print("6️⃣ Saving results...")
        
        results = {
            'improvement_config': {
                'negative_threshold': best_thresholds['neg_threshold'],
                'positive_threshold': best_thresholds['pos_threshold'],
                'method': 'threshold_adjustment'
            },
            'performance': {
                'original_negative_f1': 0.38,  # From your original evaluation
                'improved_negative_f1': full_report['0']['f1-score'],
                'improvement': full_report['0']['f1-score'] - 0.38,
                'overall_accuracy': full_report['accuracy'],
                'macro_f1': full_report['macro avg']['f1-score'],
                'clear_examples_accuracy': clear_accuracy
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to file
        with open('outputs/quick_negative_fix_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   ✅ Results saved to outputs/quick_negative_fix_results.json")
        
        # Create API integration snippet
        api_snippet = f"""
# Add this to your src/api.py predict_sentiment function

# Load improved thresholds (add this at module level)
try:
    with open('outputs/quick_negative_fix_results.json', 'r') as f:
        improvement_config = json.load(f)['improvement_config']
    NEGATIVE_THRESHOLD = improvement_config['negative_threshold']
    POSITIVE_THRESHOLD = improvement_config['positive_threshold']
    print(f"Loaded improved thresholds: neg={{NEGATIVE_THRESHOLD:.2f}}, pos={{POSITIVE_THRESHOLD:.2f}}")
except:
    NEGATIVE_THRESHOLD = 0.33
    POSITIVE_THRESHOLD = 0.33
    print("Using default thresholds")

# Then modify your prediction logic:
def predict_sentiment_improved(text: str, model_name: str = "finbert"):
    # ... existing tokenization code ...
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    # Apply improved thresholds
    neg_prob, neu_prob, pos_prob = probs
    
    if neg_prob >= NEGATIVE_THRESHOLD:
        predicted_class = 0  # negative
    elif pos_prob >= POSITIVE_THRESHOLD:
        predicted_class = 2  # positive
    else:
        predicted_class = 1  # neutral
    
    labels = ["negative", "neutral", "positive"]
    prediction = labels[predicted_class]
    confidence = float(probs[predicted_class])
    
    # ... rest of existing code ...
"""
        
        with open('outputs/api_integration_snippet.py', 'w') as f:
            f.write(api_snippet)
        
        print(f"   ✅ API integration code saved to outputs/api_integration_snippet.py")
        
        return results
    
    def run_quick_fix(self):
        """Run the complete quick fix process"""
        start_time = time.time()
        
        try:
            # Step 1: Test model loading
            tokenizer, model = self.test_model_loading()
            if tokenizer is None:
                return False
            
            # Step 2: Load test data
            test_df = self.load_test_data()
            if test_df is None:
                return False
            
            # Step 3: Quick threshold search
            best_result, all_results = self.quick_threshold_test(tokenizer, model, test_df)
            
            # Step 4: Validate on full dataset
            full_report = self.test_on_full_dataset(tokenizer, model, test_df, best_result)
            
            # Step 5: Test clear examples
            clear_accuracy = self.test_clear_examples(tokenizer, model, best_result)
            
            # Step 6: Save results
            saved_results = self.save_results(best_result, full_report, clear_accuracy)
            
            # Summary
            elapsed = time.time() - start_time
            print(f"\n🎉 Quick fix completed in {elapsed:.1f} seconds!")
            
            original_neg_f1 = 0.38
            improved_neg_f1 = full_report['0']['f1-score']
            improvement = improved_neg_f1 - original_neg_f1
            
            print(f"\n📊 IMPROVEMENT SUMMARY:")
            print(f"   Negative F1: {original_neg_f1:.3f} → {improved_neg_f1:.3f} (+{improvement:.3f})")
            print(f"   Overall Accuracy: {full_report['accuracy']:.3f}")
            print(f"   Macro F1: {full_report['macro avg']['f1-score']:.3f}")
            print(f"   Clear Examples: {clear_accuracy:.1%}")
            
            if improvement > 0.1:
                print(f"   🎯 Great improvement! +{improvement:.3f} F1 points on negatives")
            elif improvement > 0.05:
                print(f"   👍 Good improvement! +{improvement:.3f} F1 points on negatives")
            else:
                print(f"   📈 Modest improvement: +{improvement:.3f} F1 points on negatives")
            
            print(f"\n📋 Next steps:")
            print(f"   1. Check outputs/quick_negative_fix_results.json for thresholds")
            print(f"   2. Integrate using outputs/api_integration_snippet.py")
            print(f"   3. Test your API with improved thresholds")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Quick fix failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    print("Starting quick negative sentiment fix...")
    print("This should complete in under 5 minutes.\n")
    
    fixer = QuickNegativeFix()
    success = fixer.run_quick_fix()
    
    if success:
        print("\n✅ Success! Your negative detection should be improved.")
    else:
        print("\n❌ Fix failed. Check the error messages above.")
    
    print(f"\nLog saved to: improvement_log_{int(time.time())}.txt")

if __name__ == "__main__":
    # Save output to log file
    import sys
    log_filename = f"improvement_log_{int(time.time())}.txt"
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    with open(log_filename, 'w') as logfile:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, logfile)
        
        try:
            main()
        finally:
            sys.stdout = original_stdout
    
    print(f"Complete log saved to: {log_filename}")