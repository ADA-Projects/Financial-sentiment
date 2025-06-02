#!/usr/bin/env python3
"""
Comprehensive test suite for the Financial Sentiment Analysis API
Tests all endpoints and features including batch processing and caching
"""

import requests
import json
import time
from typing import Dict, List
import asyncio
import aiohttp

class FinBertAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_check(self) -> bool:
        """Test basic health check"""
        print("🏥 Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                print(f"   Models loaded: {data['models_loaded']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test root endpoint"""
        print("\n🏠 Testing root endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Root endpoint: {data['message']}")
                print(f"   Available models: {data['available_models']}")
                return True
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            return False
    
    def test_models_endpoint(self) -> bool:
        """Test models information endpoint"""
        print("\n🤖 Testing models endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/models")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Models endpoint successful")
                print(f"   Available models: {data['available_models']}")
                print(f"   Default model: {data['default_model']}")
                return True
            else:
                print(f"❌ Models endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Models endpoint error: {e}")
            return False
    
    def test_single_prediction(self, model: str = "finbert") -> bool:
        """Test single text prediction"""
        print(f"\n🔍 Testing single prediction with {model}...")
        
        test_cases = [
            {
                "text": "Company profits increased by 25% this quarter",
                "expected": "positive"
            },
            {
                "text": "Revenue declined significantly due to market conditions", 
                "expected": "negative"
            },
            {
                "text": "The company maintained steady performance",
                "expected": "neutral"
            }
        ]
        
        success_count = 0
        
        for i, case in enumerate(test_cases):
            try:
                payload = {
                    "text": case["text"],
                    "model": model
                }
                
                response = self.session.post(f"{self.base_url}/analyze", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    predicted = data["sentiment"]
                    confidence = data["confidence"]
                    processing_time = data["processing_time_ms"]
                    
                    print(f"   Test {i+1}: '{case['text'][:50]}...'")
                    print(f"   Predicted: {predicted} (confidence: {confidence:.3f})")
                    print(f"   Processing time: {processing_time}ms")
                    
                    # Check if prediction matches expectation (optional)
                    if predicted == case["expected"]:
                        print(f"   ✅ Matches expected: {case['expected']}")
                    else:
                        print(f"   ⚠️  Expected: {case['expected']}, got: {predicted}")
                    
                    success_count += 1
                else:
                    print(f"   ❌ Request failed: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        success = success_count == len(test_cases)
        print(f"✅ Single prediction test: {success_count}/{len(test_cases)} successful")
        return success
    
    def test_batch_prediction(self, model: str = "finbert") -> bool:
        """Test batch prediction"""
        print(f"\n📊 Testing batch prediction with {model}...")
        
        test_texts = [
            "Stock prices soared after earnings beat expectations",
            "Company announced major layoffs and cost cutting measures",
            "Quarterly results met analyst forecasts",
            "New product launch shows promising early adoption",
            "Regulatory concerns weigh on future growth prospects"
        ]
        
        try:
            payload = {
                "texts": test_texts,
                "model": model
            }
            
            start_time = time.time()
            response = self.session.post(f"{self.base_url}/analyze/batch", json=payload)
            request_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Batch prediction successful")
                print(f"   Total processed: {data['total_processed']}")
                print(f"   Average confidence: {data['average_confidence']:.3f}")
                print(f"   API processing time: {data['processing_time_ms']}ms")
                print(f"   Total request time: {request_time:.1f}ms")
                
                # Show individual results
                for i, result in enumerate(data['results'][:3]):  # Show first 3
                    print(f"   Result {i+1}: {result['sentiment']} ({result['confidence']:.3f})")
                
                return True
            else:
                print(f"❌ Batch prediction failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return False
    
    def test_caching(self) -> bool:
        """Test caching functionality"""
        print("\n💾 Testing caching...")
        
        test_text = "Company profits increased substantially this quarter"
        payload = {
            "text": test_text,
            "model": "finbert"
        }
        
        try:
            # First request (cache miss)
            start_time = time.time()
            response1 = self.session.post(f"{self.base_url}/analyze", json=payload)
            time1 = (time.time() - start_time) * 1000
            
            # Second request (should be cache hit)
            start_time = time.time()
            response2 = self.session.post(f"{self.base_url}/analyze", json=payload)
            time2 = (time.time() - start_time) * 1000
            
            if response1.status_code == 200 and response2.status_code == 200:
                data1 = response1.json()
                data2 = response2.json()
                
                # Results should be identical
                same_results = (
                    data1["sentiment"] == data2["sentiment"] and
                    data1["confidence"] == data2["confidence"]
                )
                
                print(f"✅ Caching test successful")
                print(f"   First request: {time1:.1f}ms")
                print(f"   Second request: {time2:.1f}ms")
                print(f"   Results identical: {same_results}")
                print(f"   Speed improvement: {(time1 - time2):.1f}ms")
                
                return same_results
            else:
                print(f"❌ Caching test failed: {response1.status_code}, {response2.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Caching test error: {e}")
            return False
    
    def test_cache_stats(self) -> bool:
        """Test cache statistics endpoint"""
        print("\n📈 Testing cache statistics...")
        try:
            response = self.session.get(f"{self.base_url}/cache/stats")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Cache stats: {data}")
                return True
            else:
                print(f"❌ Cache stats failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cache stats error: {e}")
            return False
    
    def test_legacy_endpoint(self) -> bool:
        """Test legacy /score endpoint"""
        print("\n🔄 Testing legacy endpoint...")
        try:
            payload = {
                "text": "Company performance exceeded expectations",
                "model": "finbert"
            }
            
            response = self.session.post(f"{self.base_url}/score", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # Should return old format (just probabilities)
                has_probs = all(label in data for label in ["negative", "neutral", "positive"])
                print(f"✅ Legacy endpoint works: {has_probs}")
                print(f"   Response: {data}")
                return has_probs
            else:
                print(f"❌ Legacy endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Legacy endpoint error: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling"""
        print("\n⚠️  Testing error handling...")
        
        test_cases = [
            {
                "name": "Empty text",
                "payload": {"text": "", "model": "finbert"},
                "expected_status": 422
            },
            {
                "name": "Invalid model",
                "payload": {"text": "Valid text", "model": "invalid_model"},
                "expected_status": 400
            },
            {
                "name": "Too long text",
                "payload": {"text": "x" * 2000, "model": "finbert"},
                "expected_status": 422
            }
        ]
        
        success_count = 0
        
        for case in test_cases:
            try:
                response = self.session.post(f"{self.base_url}/analyze", json=case["payload"])
                
                if response.status_code == case["expected_status"]:
                    print(f"   ✅ {case['name']}: Correctly returned {response.status_code}")
                    success_count += 1
                else:
                    print(f"   ❌ {case['name']}: Expected {case['expected_status']}, got {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ {case['name']}: Error {e}")
        
        return success_count == len(test_cases)
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        print("🚀 Starting comprehensive API tests...\n")
        
        results = {}
        
        # Basic connectivity tests
        results["health_check"] = self.test_health_check()
        results["root_endpoint"] = self.test_root_endpoint()
        results["models_endpoint"] = self.test_models_endpoint()
        
        # Core functionality tests
        results["single_prediction"] = self.test_single_prediction()
        results["batch_prediction"] = self.test_batch_prediction()
        
        # Performance and caching tests
        results["caching"] = self.test_caching()
        results["cache_stats"] = self.test_cache_stats()
        
        # Compatibility and error handling
        results["legacy_endpoint"] = self.test_legacy_endpoint()
        results["error_handling"] = self.test_error_handling()
        
        # Test multiple models if available
        if "econbert" in self.get_available_models():
            print("\n🔬 Testing EconBERT model...")
            results["econbert_prediction"] = self.test_single_prediction("econbert")
        
        # Summary
        self.print_summary(results)
        return results
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = self.session.get(f"{self.base_url}/models")
            if response.status_code == 200:
                return response.json()["available_models"]
        except:
            pass
        return []
    
    def print_summary(self, results: Dict[str, bool]):
        """Print test summary"""
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
        
        print("-"*60)
        print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! API is ready for production.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        print("="*60)

async def test_concurrent_requests():
    """Test concurrent request handling"""
    print("\n🔄 Testing concurrent requests...")
    
    async def make_request(session, text, request_id):
        payload = {
            "text": f"{text} (request {request_id})",
            "model": "finbert"
        }
        
        try:
            async with session.post("http://localhost:8000/analyze", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "id": request_id,
                        "sentiment": data["sentiment"],
                        "processing_time": data["processing_time_ms"],
                        "success": True
                    }
                else:
                    return {"id": request_id, "success": False, "status": response.status}
        except Exception as e:
            return {"id": request_id, "success": False, "error": str(e)}
    
    base_text = "Company earnings exceeded expectations"
    num_requests = 10
    
    try:
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            # Create concurrent requests
            tasks = [
                make_request(session, base_text, i) 
                for i in range(num_requests)
            ]
            
            results = await asyncio.gather(*tasks)
            total_time = (time.time() - start_time) * 1000
            
            successful = [r for r in results if r.get("success", False)]
            
            print(f"✅ Concurrent requests test completed")
            print(f"   Total requests: {num_requests}")
            print(f"   Successful: {len(successful)}")
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Average per request: {total_time/num_requests:.1f}ms")
            
            if successful:
                avg_processing = sum(r["processing_time"] for r in successful) / len(successful)
                print(f"   Average processing time: {avg_processing:.1f}ms")
            
            return len(successful) == num_requests
            
    except Exception as e:
        print(f"❌ Concurrent test error: {e}")
        return False

def main():
    """Main test runner"""
    print("🧪 Financial Sentiment Analysis API - Test Suite")
    print("=" * 60)
    
    # Initialize tester
    tester = FinBertAPITester()
    
    # Run synchronous tests
    results = tester.run_all_tests()
    
    # Run async concurrent test
    print("\n🔄 Running concurrent request test...")
    try:
        concurrent_result = asyncio.run(test_concurrent_requests())
        results["concurrent_requests"] = concurrent_result
    except Exception as e:
        print(f"❌ Could not run concurrent test: {e}")
        results["concurrent_requests"] = False
    
    # Final summary
    print(f"\n🎯 FINAL RESULTS: {sum(results.values())}/{len(results)} tests passed")
    
    if all(results.values()):
        print("🚀 API is production-ready!")
        return 0
    else:
        print("⚠️  Some issues found. Please review.")
        return 1

if __name__ == "__main__":
    exit(main())