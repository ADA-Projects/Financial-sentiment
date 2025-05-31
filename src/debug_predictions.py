#!/usr/bin/env python3
"""
Debug wrong predictions
"""

import requests
import json

def debug_wrong_predictions():
    """Test the problematic sentences and analyze why they're wrong"""
    
    base_url = "http://localhost:8000"
    
    # Problematic sentences
    test_cases = [
        {
            "sentence": "Market crash threatens investor portfolios",
            "expected": "negative",
            "reason": "Crash and threatens are clearly negative"
        },
        {
            "sentence": "Fed raises interest rates causing market volatility", 
            "expected": "negative",
            "reason": "Rate hikes usually cause market stress"
        },
        {
            "sentence": "Company reports massive losses this quarter",
            "expected": "negative", 
            "reason": "Massive losses should be clearly negative"
        },
        {
            "sentence": "Stock price plunges after earnings miss",
            "expected": "negative",
            "reason": "Plunges and miss are negative indicators"
        },
        {
            "sentence": "Analysts upgrade stock to strong buy rating",
            "expected": "positive",
            "reason": "Upgrade and strong buy are positive"
        }
    ]
    
    print("🔍 Debugging Wrong Predictions...")
    print("=" * 60)
    
    correct = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{case['sentence']}'")
        print(f"   Expected: {case['expected']}")
        print(f"   Why: {case['reason']}")
        
        try:
            response = requests.post(
                f"{base_url}/predict/single",
                json={"sentence": case['sentence']}
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result['predicted_sentiment']
                confidence = result['confidence']
                
                if predicted == case['expected']:
                    print(f"   ✅ CORRECT: {predicted} (confidence: {confidence:.3f})")
                    correct += 1
                else:
                    print(f"   ❌ WRONG: {predicted} (confidence: {confidence:.3f})")
                    print(f"   📊 Probabilities:")
                    for sentiment, prob in result['probabilities'].items():
                        emoji = "🎯" if sentiment == case['expected'] else "  "
                        print(f"      {emoji} {sentiment}: {prob:.3f}")
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Overall Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    
    if correct < total:
        print("\n💡 Suggestions to improve:")
        print("1. Try EconBERT-only (remove handcrafted features)")
        print("2. Add more sophisticated sentiment lexicons")
        print("3. Re-balance training data with more negative examples")
        print("4. Fine-tune EconBERT specifically on your domain")

if __name__ == "__main__":
    debug_wrong_predictions()