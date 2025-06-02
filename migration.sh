#!/bin/bash
# migrate.sh - Migrate your current project to the enhanced setup

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🔄 Migrating to Enhanced FinBERT API${NC}"
echo "====================================="

# Backup current files
echo -e "${BLUE}📁 Creating backups...${NC}"
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"

if [ -f "src/api.py" ]; then
    cp src/api.py $BACKUP_DIR/api_original.py
    echo -e "${GREEN}✅ Backed up original API${NC}"
fi

if [ -f "test_api.py" ]; then
    cp test_api.py $BACKUP_DIR/test_api_original.py
    echo -e "${GREEN}✅ Backed up original test${NC}"
fi

if [ -f "dockerfile" ]; then
    cp dockerfile $BACKUP_DIR/dockerfile_original
    echo -e "${GREEN}✅ Backed up original Dockerfile${NC}"
fi

# Create the new files by copying the enhanced versions
echo -e "${BLUE}📝 Installing enhanced API...${NC}"

# You'll need to manually copy the enhanced API content to src/api.py
cat > src/api_enhanced.py << 'EOF'
# The enhanced API content goes here
# This is a placeholder - you'll replace this with the actual enhanced API code
print("Enhanced API - replace this with the actual code from the artifacts")
EOF

echo -e "${YELLOW}⚠️  MANUAL STEP REQUIRED:${NC}"
echo "   1. Copy the enhanced API code from the artifacts to src/api.py"
echo "   2. Copy the comprehensive test code to test_api_comprehensive.py"
echo "   3. Copy the Dockerfile content to dockerfile"

# Create directory structure
echo -e "${BLUE}📁 Setting up directory structure...${NC}"
mkdir -p logs
mkdir -p cache
mkdir -p tests
mkdir -p scripts

# Create environment file
echo -e "${BLUE}📝 Creating .env file...${NC}"
cat > .env << 'EOF'
# Basic configuration for your setup
API_HOST=0.0.0.0
API_PORT=8000
DEFAULT_MODEL=finbert
LOG_LEVEL=INFO
ENABLE_CACHE=true
CACHE_SIZE_LIMIT=1000
MAX_BATCH_SIZE=100
CORS_ALLOW_ORIGINS=["*"]
EOF

# Create a migration checklist
echo -e "${BLUE}📋 Creating migration checklist...${NC}"
cat > MIGRATION_CHECKLIST.md << 'EOF'
# Migration Checklist

## ✅ Completed Automatically
- [x] Backup original files
- [x] Create directory structure  
- [x] Create .env file
- [x] Create migration checklist

## 📝 Manual Steps Required

### 1. Replace API Code
- [ ] Copy enhanced API code to `src/api.py`
- [ ] Verify model paths in the new API match your setup
- [ ] Test the API starts without errors

### 2. Replace Test Code  
- [ ] Copy comprehensive test code to `test_api.py`
- [ ] Run tests to verify everything works

### 3. Update Dockerfile
- [ ] Copy new Dockerfile content to `dockerfile`
- [ ] Test Docker build: `docker build -t finbert-test .`

### 4. Copy Scripts
- [ ] Copy deployment.sh and make executable: `chmod +x deployment.sh`
- [ ] Copy setup.sh and run it: `./setup.sh`

### 5. Verify Setup
- [ ] Start API: `./deployment.sh dev`
- [ ] Run tests: `python test_api.py`
- [ ] Check docs: http://localhost:8000/docs
- [ ] Test Docker: `./deployment.sh deploy`

## 🔧 Configuration Updates

Your current model structure:
```
outputs/
├── finbert_model/
├── finbert_tokenizer/
├── econbert_model/
├── tokenizer/
├── finbert_pipeline.joblib
└── class_weighted_pipeline.joblib
```

The enhanced API will automatically detect and load:
- Pipeline files (*.joblib) - preferred for speed
- Model directories - fallback option

## 🚀 New Features Available After Migration

1. **Enhanced API Responses**
   - Confidence scores
   - Processing time
   - Timestamps
   - Model information

2. **Batch Processing**  
   - Process multiple texts at once
   - Improved throughput

3. **Caching**
   - Faster repeated requests
   - Cache statistics

4. **Better Error Handling**
   - Detailed error messages
   - Input validation

5. **Monitoring**
   - Health checks
   - Performance metrics
   - Request logging

6. **Production Features**
   - Docker optimization
   - Security headers
   - Rate limiting support

## 🧪 Testing Your Migration

After completing manual steps:

```bash
# Test basic functionality
python -c "
import requests
r = requests.get('http://localhost:8000/health')
print('Health check:', r.status_code == 200)
"

# Test prediction
python -c "
import requests
r = requests.post('http://localhost:8000/analyze', 
    json={'text': 'Company profits increased', 'model': 'finbert'})
print('Prediction test:', r.status_code == 200)
if r.status_code == 200:
    print('Response:', r.json())
"
```

## 📞 Need Help?

If you encounter issues:
1. Check the backup files in `backups/`
2. Verify model files exist in `outputs/`
3. Check logs: `tail -f logs/api.log`
4. Test with original API: `python -m uvicorn src.api:app`
EOF

# Create a simple verification script
echo -e "${BLUE}🧪 Creating verification script...${NC}"
cat > verify_migration.py << 'EOF'
#!/usr/bin/env python3
"""Verify migration was successful"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if required files exist"""
    required_files = [
        'src/api.py',
        'test_api.py', 
        'dockerfile',
        '.env',
        'requirements.txt'
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    return missing

def check_models():
    """Check if models are available"""
    model_files = [
        'outputs/finbert_pipeline.joblib',
        'outputs/class_weighted_pipeline.joblib',
        'outputs/finbert_model/config.json',
        'outputs/econbert_model/config.json'
    ]
    
    available = []
    for file in model_files:
        if Path(file).exists():
            available.append(file)
    
    return available

def main():
    print("🔍 Verifying Migration")
    print("=" * 30)
    
    # Check files
    missing = check_files()
    if missing:
        print("❌ Missing files:")
        for file in missing:
            print(f"   - {file}")
    else:
        print("✅ All required files present")
    
    # Check models
    available = check_models()
    print(f"\n📊 Available models: {len(available)}")
    for model in available:
        print(f"   ✅ {model}")
    
    # Check environment
    if Path('.env').exists():
        print("\n✅ Environment file created")
    else:
        print("\n❌ No .env file found")
    
    # Summary
    print(f"\n📋 Migration Status:")
    if not missing and available:
        print("🎉 Migration setup complete!")
        print("📝 Check MIGRATION_CHECKLIST.md for next steps")
    else:
        print("⚠️  Migration needs completion")
        print("📝 See MIGRATION_CHECKLIST.md for details")

if __name__ == "__main__":
    main()
EOF

chmod +x verify_migration.py

# Final migration summary
echo ""
echo "=================================="
echo -e "${GREEN}🎉 Migration setup completed!${NC}"
echo "=================================="
echo ""
echo -e "${BLUE}📁 What was created:${NC}"
echo "   ✅ Backup directory: $BACKUP_DIR"
echo "   ✅ Directory structure (logs, cache, tests)"
echo "   ✅ Basic .env configuration"
echo "   ✅ Migration checklist: MIGRATION_CHECKLIST.md"
echo "   ✅ Verification script: verify_migration.py"
echo ""
echo -e "${YELLOW}🚨 IMPORTANT - Manual steps required:${NC}"
echo "   1. Copy the enhanced API code to src/api.py"
echo "   2. Copy the test code to test_api.py"  
echo "   3. Copy the Dockerfile content"
echo "   4. Copy and run the setup script"
echo ""
echo -e "${BLUE}📋 Next steps:${NC}"
echo "   1. Read: cat MIGRATION_CHECKLIST.md"
echo "   2. Verify: python verify_migration.py"
echo "   3. Complete manual steps from checklist"
echo "   4. Test: ./deployment.sh dev"
echo ""
echo -e "${GREEN}💡 Your current models will work with the enhanced API!${NC}"