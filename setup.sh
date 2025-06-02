#!/bin/bash
# setup.sh - Initial project setup and configuration

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Setting up Financial Sentiment Analysis Project${NC}"
echo "=================================================="

# Create necessary directories
echo -e "${BLUE}📁 Creating project structure...${NC}"
mkdir -p logs
mkdir -p cache
mkdir -p tests

# Copy environment template if .env doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${BLUE}📝 Creating .env file from template...${NC}"
    cp .env.template .env
    echo -e "${YELLOW}⚠️  Please review and customize .env file${NC}"
fi

# Make scripts executable
echo -e "${BLUE}🔧 Making scripts executable...${NC}"
chmod +x deployment.sh

# Update API to use your actual models
echo -e "${BLUE}🔄 Updating API configuration...${NC}"

# Replace your current API with the enhanced version
if [ -f "src/api_backup.py" ]; then
    echo -e "${YELLOW}⚠️  Backing up current API to src/api_old.py${NC}"
    cp src/api.py src/api_old.py
fi

# Check which models are available
echo -e "${BLUE}🔍 Checking available models...${NC}"

if [ -f "outputs/finbert_pipeline.joblib" ]; then
    echo -e "${GREEN}✅ FinBERT pipeline found${NC}"
    FINBERT_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  FinBERT pipeline not found${NC}"
    FINBERT_AVAILABLE=false
fi

if [ -f "outputs/class_weighted_pipeline.joblib" ]; then
    echo -e "${GREEN}✅ EconBERT pipeline found${NC}"
    ECONBERT_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  EconBERT pipeline not found${NC}"
    ECONBERT_AVAILABLE=false
fi

# Check for model directories
if [ -d "outputs/finbert_model" ] && [ -d "outputs/finbert_tokenizer" ]; then
    echo -e "${GREEN}✅ FinBERT model files found${NC}"
    FINBERT_MODEL_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  FinBERT model files not found${NC}"
    FINBERT_MODEL_AVAILABLE=false
fi

if [ -d "outputs/econbert_model" ] && [ -d "outputs/tokenizer" ]; then
    echo -e "${GREEN}✅ EconBERT model files found${NC}"
    ECONBERT_MODEL_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  EconBERT model files not found${NC}"
    ECONBERT_MODEL_AVAILABLE=false
fi

# Create model info JSON
echo -e "${BLUE}📊 Creating model info...${NC}"
cat > outputs/model_info.json << EOF
{
    "models": {
        "finbert": {
            "pipeline_available": $FINBERT_AVAILABLE,
            "model_files_available": $FINBERT_MODEL_AVAILABLE,
            "description": "FinBERT model fine-tuned for financial sentiment analysis",
            "labels": ["negative", "neutral", "positive"]
        },
        "econbert": {
            "pipeline_available": $ECONBERT_AVAILABLE,
            "model_files_available": $ECONBERT_MODEL_AVAILABLE,
            "description": "EconBERT model for economic text analysis",
            "labels": ["negative", "neutral", "positive"]
        }
    },
    "default_model": "finbert",
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# Install/update requirements if in virtual environment
if [ -n "$VIRTUAL_ENV" ] || [ -d ".venv" ]; then
    echo -e "${BLUE}📦 Installing/updating dependencies...${NC}"
    
    if [ -d ".venv" ] && [ -z "$VIRTUAL_ENV" ]; then
        echo -e "${YELLOW}⚠️  Activating virtual environment...${NC}"
        source .venv/bin/activate
    fi
    
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies updated${NC}"
else
    echo -e "${YELLOW}⚠️  No virtual environment detected. Create one with:${NC}"
    echo "   python -m venv .venv"
    echo "   source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows"
    echo "   pip install -r requirements.txt"
fi

# Create a simple test data file
echo -e "${BLUE}🧪 Creating test data...${NC}"
cat > data/test_samples.json << 'EOF'
{
    "positive_samples": [
        "Company profits soared 25% exceeding all expectations",
        "Strong earnings growth drives stock price higher",
        "Revenue increased substantially this quarter",
        "Excellent financial performance across all segments",
        "Record-breaking quarterly results announced"
    ],
    "negative_samples": [
        "Company reports significant losses this quarter",
        "Revenue declined sharply due to market conditions",
        "Stock price plummeted after disappointing earnings",
        "Major layoffs announced as company struggles",
        "Bankruptcy filing expected within months"
    ],
    "neutral_samples": [
        "Company maintains steady performance this quarter",
        "Results met analyst expectations",
        "Business operations continue as planned",
        "No significant changes in financial outlook",
        "Standard quarterly report filed with SEC"
    ]
}
EOF

# Create a quick test script
echo -e "${BLUE}🧪 Creating quick test script...${NC}"
cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""Quick API test script"""

import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    # Test samples
    samples = [
        "Company profits increased by 25%",
        "Revenue declined this quarter", 
        "Performance remained steady"
    ]
    
    print("🧪 Quick API Test")
    print("=" * 30)
    
    # Test health
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ API is healthy")
        else:
            print("❌ API health check failed")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("💡 Make sure API is running: ./deployment.sh dev")
        return
    
    # Test predictions
    for i, text in enumerate(samples, 1):
        try:
            response = requests.post(
                f"{base_url}/analyze",
                json={"text": text, "model": "finbert"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Test {i}: {data['sentiment']} ({data['confidence']:.3f})")
            else:
                print(f"❌ Test {i} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Test {i} error: {e}")
    
    print("\n🎉 Quick test completed!")

if __name__ == "__main__":
    test_api()
EOF

chmod +x quick_test.py

# Create gitignore additions if needed
echo -e "${BLUE}📝 Updating .gitignore...${NC}"
cat >> .gitignore << 'EOF'

# Environment files
.env
.env.local
.env.production

# Logs
logs/
*.log

# Cache
cache/
.cache/

# Test outputs
test-results/
coverage/

# IDE
.vscode/settings.json
.idea/workspace.xml

EOF

# Final setup summary
echo ""
echo "=================================="
echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo "=================================="
echo ""
echo "📋 What's been set up:"
echo "   ✅ Project directories created"
echo "   ✅ Environment template copied to .env"
echo "   ✅ Scripts made executable"
echo "   ✅ Model info generated"
echo "   ✅ Test data created"
echo "   ✅ Quick test script ready"
echo ""
echo "🚀 Next steps:"
echo "   1. Review and customize .env file"
echo "   2. Start API: ./deployment.sh dev"
echo "   3. Test API: python quick_test.py"
echo "   4. View docs: http://localhost:8000/docs"
echo ""
echo "📊 Available models:"
if [ "$FINBERT_AVAILABLE" = true ] || [ "$FINBERT_MODEL_AVAILABLE" = true ]; then
    echo "   ✅ FinBERT ready"
else
    echo "   ⚠️  FinBERT needs training"
fi

if [ "$ECONBERT_AVAILABLE" = true ] || [ "$ECONBERT_MODEL_AVAILABLE" = true ]; then
    echo "   ✅ EconBERT ready"
else
    echo "   ⚠️  EconBERT needs training"
fi
echo ""
echo "💡 Tips:"
echo "   - Use 'python quick_test.py' for fast testing"
echo "   - Use './deployment.sh test' for comprehensive tests"
echo "   - Use './deployment.sh deploy' for Docker deployment"
echo "   - Check logs with './deployment.sh logs'"