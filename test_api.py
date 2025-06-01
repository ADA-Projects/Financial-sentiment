#!/usr/bin/env python3
"""
Simple API test script
"""

import requests
import json
import time

def test_api():
    base_url = "http://localhost:8000"
    
    print("🔍 Testing Financial Sentiment API...")
    
    # Test 1: Health check
    print("\n1. Health Check:")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data['status']}")
            print(f"   ✅ Model loaded: {data['model_loaded']}")
            print(f"   ✅ Version: {data['version']}")
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")
            return
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        print("   💡 Make sure the API is running: python src/api.py")
        return
    
    # Test 2: Single prediction
    print("\n2. Single Prediction:")
    try:
        payload = {"sentence": "Apple reported strong quarterly earnings beating expectations"}
        response = requests.post(
            f"{base_url}/predict/single",
            json=payload
        )
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Sentence: {result['sentence'][:50]}...")
            print(f"   ✅ Predicted: {result['predicted_sentiment']}")
            print(f"   ✅ Confidence: {result['confidence']:.3f}")
            print(f"   ✅ Probabilities:")
            for sentiment, prob in result['probabilities'].items():
                print(f"      - {sentiment}: {prob:.3f}")
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Batch prediction
    print("\n3. Batch Prediction:")
    try:
        test_sentences = [
            "Company profits soared 25% this quarter",
            "Market crash threatens investor portfolios", 
            "Economic indicators remain stable",
            "Tesla stock price jumps on strong delivery numbers",
            "Fed raises interest rates causing market volatility"
        ]
        
        payload = {"sentences": test_sentences}
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/predict",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            total_time = time.time() - start_time
            
            print(f"   ✅ Processed {len(data['results'])} sentences")
            print(f"   ✅ API processing time: {data['processing_time_ms']:.0f}ms")
            print(f"   ✅ Total request time: {total_time*1000:.0f}ms")
            print(f"   ✅ Results:")
            
            for i, result in enumerate(data['results']):
                sentiment = result['predicted_sentiment']
                confidence = result['confidence']
                sentence = result['sentence'][:40] + "..." if len(result['sentence']) > 40 else result['sentence']
                
                print(f"      {i+1}. '{sentence}' → {sentiment} ({confidence:.3f})")
                
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Model info
    print("\n4. Model Information:")
    try:
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model type: {data['model_type']}")
            print(f"   ✅ Labels: {data['labels']}")
            print(f"   ✅ Max length: {data['max_sequence_length']}")
            print(f"   ✅ Device: {data['device']}")
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Examples endpoint
    print("\n5. Example Sentences:")
    try:
        response = requests.get(f"{base_url}/examples")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Available examples:")
            for i, example in enumerate(data['examples'][:3]):  # Show first 3
                print(f"      {i+1}. {example}")
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🎉 API testing complete!")
    print("\n💡 You can also test interactively at: http://localhost:8000/docs")

if __name__ == "__main__":
    test_api()