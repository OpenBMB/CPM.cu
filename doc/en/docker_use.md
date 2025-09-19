# CPM.cu Docker User Guide

This document is for **users** who want to use CPM.cu Docker images for LLM inference.

## Quick Start

### Prerequisites

- Docker installed
- NVIDIA GPU drivers (recommended version >= 525.60.13)
- Supported GPUs: A100/A800, RTX 30/40/50 series, H100/H800
- Python versions: 3.8–3.12 are supported (host Python not required when using Docker)

### Get the Image

```bash
# Method 1: Use pre-built image (recommended)
docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:v1.0.0
docker tag modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:v1.0.0 cpmcu:cuda12.6-release

# Optional: CUDA 12.8 (recommended for RTX 50 series)
docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.8:v1.0.0
docker tag modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.8:v1.0.0 cpmcu:cuda12.8-release

# Method 2: Build locally
git clone https://github.com/OpenBMB/CPM.cu.git
cd CPM.cu/docker/cuda12.6
./build.sh
```

Local build reference [Docker Build Guide](docker_build.md)

## Basic Usage

### Interactive Mode

```bash
# Start interactive container
docker run --gpus all -it cpmcu:cuda12.6-release

# Run inside container
python examples/minicpm4/test_generate.py --help
```

### Text Generation Examples

```bash
# Use default prompt
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py

# Use custom prompt
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py --prompt-text "Explain what artificial intelligence is"

# Use prompt file
docker run --gpus all \
  -v /path/to/your/prompt.txt:/workspace/prompt.txt \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py --prompt-file /workspace/prompt.txt
```

### API Server

```bash
# Start OpenAI-compatible API server
docker run --gpus all -p 8000:8000 cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py

# Run in background
docker run --gpus all -p 8000:8000 -d --name cpmcu-server \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py
```

### Test API Server

```bash
# Test API in another terminal
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minicpm4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Advanced Usage

### Offline Usage (Download on Host then Mount, Recommended)

When the container cannot directly access Hugging Face or download speeds are slow, it's recommended to download the model locally on the host first, then mount it into the container.

```bash
# Download model using hf
huggingface-cli download openbmb/MiniCPM4-8B --local-dir /path/to/model

# Also download draft model & FRSpec (for speculative decoding, recommended to prepare offline)
huggingface-cli download openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu --local-dir /path/to/draft

# Mount and use local path at runtime
docker run --rm --gpus all \
  -v /path/to/model:/workspace/model \
  -v /path/to/draft:/workspace/draft \
  cpmcu:cuda12.6-release \
  bash -lc 'cd examples && python3 minicpm4/test_generate.py \
    --model-path /workspace/model \
    --draft-model-path /workspace/draft \
    --frspec-path /workspace/draft \
    --prompt-text "Hello" --num-generate 128 --use-stream false'
```

For custom configuration, you can view all available parameters via cpmcu.cli -h:
```bash
docker run --rm --gpus all cpmcu:cuda12.6-release python -m cpmcu.cli -h
```

### Model Configuration

```bash
# Specify model path
docker run --gpus all \
  -v /path/to/model:/workspace/model \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py --model-path /workspace/model

# Adjust generation parameters
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --max-tokens 512 \
  --temperature 0.7 \
  --top-p 0.9
```

### Memory Optimization

```bash
# Limit GPU memory usage
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --memory-limit 0.6  # Use 60% of GPU memory

# Enable memory mapping
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --use-mmap
```

### Performance Tuning

```bash
# Enable speculative sampling
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --draft-model-path /path/to/draft/model

# Adjust batch size
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py \
  --max-batch-size 32
```

## Common Use Cases

### 1. Code Generation

```bash
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --prompt-text "Write a Python function to calculate Fibonacci sequence" \
  --max-tokens 256
```

### 2. Document Summarization

```bash
# Prepare long document
echo "Your long document content..." > document.txt

docker run --gpus all \
  -v $(pwd)/document.txt:/workspace/document.txt \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --prompt-text "Please summarize the main content of the following document: $(cat /workspace/document.txt)" \
  --max-tokens 200
```

### 3. Chat System

```bash
# Start chat API server
docker run --gpus all -p 8000:8000 -d --name chat-server \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py

# Use Python client
python -c "
import openai
client = openai.OpenAI(base_url='http://localhost:8000/v1', api_key='dummy')
response = client.chat.completions.create(
    model='minicpm4',
    messages=[{'role': 'user', 'content': 'Hello, please introduce yourself'}]
)
print(response.choices[0].message.content)
"
```

## Data Persistence

### Model Cache

```bash
# Create model cache directory
mkdir -p ~/.cache/huggingface

# Mount cache directory
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py
```

### Output Saving

```bash
# Save generation results
docker run --gpus all \
  -v $(pwd)/output:/workspace/output \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py \
  --output-file /workspace/output/result.txt
```

## Troubleshooting

### GPU Related Issues

**Check GPU availability**:
```bash
docker run --gpus all cpmcu:cuda12.6-release nvidia-smi
```

**GPU memory insufficient**:
```bash
# Reduce memory usage
docker run --gpus all cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py --memory-limit 0.5
```

**CUDA version incompatible**:
```bash
# Check CUDA version
docker run --gpus all cpmcu:cuda12.6-release nvcc --version
```

### Model Loading Issues

**Model files not found**:
```bash
# Check model path
docker run --gpus all \
  -v /path/to/model:/workspace/model \
  cpmcu:cuda12.6-release \
  ls -la /workspace/model
```

**Permission issues**:
```bash
# Fix permissions
sudo chown -R $(id -u):$(id -g) /path/to/model
```

### Network Issues

**API server inaccessible**:
```bash
# Check port mapping
docker ps | grep cpmcu

# Check firewall
sudo ufw status
```

**Model download failed**:
```bash
# Use proxy
docker run --gpus all \
  -e HTTP_PROXY=http://proxy:port \
  -e HTTPS_PROXY=http://proxy:port \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py
```

## Best Practices

### Production Deployment

1. **Use fixed tags**: `cpmcu:cuda12.6-release` instead of `latest`
2. **Set resource limits**: Use `--memory` and `--cpus` parameters
3. **Health checks**: Configure container health checks
4. **Log management**: Use appropriate log drivers

```bash
# Production deployment example
docker run -d \
  --name cpmcu-prod \
  --gpus all \
  --memory=32g \
  --cpus=8 \
  --restart=unless-stopped \
  -p 8000:8000 \
  -v /data/models:/workspace/models \
  -v /logs:/workspace/logs \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py \
  --model-path /workspace/models/minicpm4 \
  --log-file /workspace/logs/server.log
```

### Development and Debugging

```bash
# Development mode (mount code directory)
docker run --gpus all -it \
  -v $(pwd):/workspace/dev \
  cpmcu:cuda12.6-release \
  bash

# Debug mode (keep container)
docker run --gpus all -it --rm=false \
  cpmcu:cuda12.6-release \
  python examples/minicpm4/test_generate.py --debug
```

## Updates and Maintenance

### Update Image

```bash
# Pull latest version
docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:latest
docker tag modelbest-registry.cn-beijing.cr.aliyuncs.com/model-align/cpmcu_cu12.6:latest cpmcu:cuda12.6-release

# Restart service
docker stop cpmcu-server
docker rm cpmcu-server
docker run --gpus all -p 8000:8000 -d --name cpmcu-server \
  cpmcu/cpmcu:cuda12.6-release \
  python examples/minicpm4/start_server.py
```
