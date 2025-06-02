#!/bin/bash
# deployment.sh - Production deployment script for FinBERT API

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_NAME="finbert-api"
DOCKER_IMAGE="finbert-sentiment:latest"
CONTAINER_NAME="finbert-container"
API_PORT=8000
HEALTH_CHECK_URL="http://localhost:${API_PORT}/health"

echo -e "${BLUE}🚀 FinBERT API Deployment Script${NC}"
echo "=================================="

# Function to print colored output
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command_exists docker; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command_exists python3; then
        log_error "Python 3 is not installed."
        exit 1
    fi
    
    # Check if models exist
    if [ ! -d "outputs" ]; then
        log_error "No outputs directory found. Please train your models first."
        exit 1
    fi
    
    if [ ! -f "outputs/finbert_model/config.json" ] && [ ! -f "outputs/finbert_pipeline.joblib" ]; then
        log_error "No FinBERT model found. Please train the model first."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    if docker build -t $DOCKER_IMAGE .; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Stop existing container
stop_existing() {
    log_info "Stopping existing container..."
    
    if docker ps -a | grep -q $CONTAINER_NAME; then
        docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
        docker rm $CONTAINER_NAME >/dev/null 2>&1 || true
        log_success "Existing container stopped and removed"
    else
        log_info "No existing container found"
    fi
}

# Start new container
start_container() {
    log_info "Starting new container..."
    
    docker run -d \
        --name $CONTAINER_NAME \
        -p $API_PORT:8000 \
        -v $(pwd)/outputs:/app/outputs:ro \
        -v $(pwd)/logs:/app/logs \
        --restart unless-stopped \
        $DOCKER_IMAGE
    
    if [ $? -eq 0 ]; then
        log_success "Container started successfully"
    else
        log_error "Failed to start container"
        exit 1
    fi
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Wait for container to start
    sleep 10
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f $HEALTH_CHECK_URL >/dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts - waiting for API to be ready..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    docker logs $CONTAINER_NAME
    exit 1
}

# Run tests
run_tests() {
    log_info "Running API tests..."
    
    if python3 test_api.py; then
        log_success "All tests passed"
    else
        log_warning "Some tests failed - check output above"
    fi
}

# Show status
show_status() {
    echo ""
    echo "=================================="
    log_success "Deployment completed successfully!"
    echo "=================================="
    echo ""
    echo "📊 Service Information:"
    echo "   API URL: http://localhost:$API_PORT"
    echo "   Documentation: http://localhost:$API_PORT/docs"
    echo "   Health Check: $HEALTH_CHECK_URL"
    echo ""
    echo "🐳 Docker Information:"
    echo "   Image: $DOCKER_IMAGE"
    echo "   Container: $CONTAINER_NAME"
    echo ""
    echo "📝 Useful Commands:"
    echo "   View logs: docker logs $CONTAINER_NAME"
    echo "   Stop API: docker stop $CONTAINER_NAME"
    echo "   Restart API: docker restart $CONTAINER_NAME"
    echo "   Remove container: docker rm $CONTAINER_NAME"
    echo ""
}

# Main deployment function
deploy() {
    check_prerequisites
    build_image
    stop_existing
    start_container
    health_check
    run_tests
    show_status
}

# Development mode
dev_mode() {
    log_info "Starting development mode..."
    
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv .venv
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install dependencies
    log_info "Installing dependencies..."
    pip install -r requirements.txt
    
    # Start API in development mode
    log_info "Starting API in development mode..."
    log_warning "Press Ctrl+C to stop"
    
    python3 -m uvicorn src.api:app --host 0.0.0.0 --port $API_PORT --reload
}

# Show usage
usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "   deploy    Deploy API using Docker (default)"
    echo "   dev       Start API in development mode"
    echo "   test      Run tests against running API"
    echo "   stop      Stop the running container"
    echo "   logs      Show container logs"
    echo "   status    Show deployment status"
    echo "   clean     Clean up Docker resources"
    echo ""
}

# Parse command line arguments
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    dev)
        dev_mode
        ;;
    test)
        log_info "Running tests..."
        python3 test_api.py
        ;;
    stop)
        log_info "Stopping container..."
        docker stop $CONTAINER_NAME
        log_success "Container stopped"
        ;;
    logs)
        docker logs -f $CONTAINER_NAME
        ;;
    status)
        if docker ps | grep -q $CONTAINER_NAME; then
            log_success "Container is running"
            curl -s $HEALTH_CHECK_URL | python3 -m json.tool
        else
            log_warning "Container is not running"
        fi
        ;;
    clean)
        log_info "Cleaning up Docker resources..."
        docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
        docker rm $CONTAINER_NAME >/dev/null 2>&1 || true
        docker rmi $DOCKER_IMAGE >/dev/null 2>&1 || true
        log_success "Cleanup completed"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac