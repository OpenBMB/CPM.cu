#!/bin/bash
# Usage: ./build.sh [CUDA_ARCH] [BASE_IMAGE]
# CUDA_ARCH: optional, override default arch list, e.g. "80,86,89"
# BASE_IMAGE: optional, specify base image
# Proxy support: pass http_proxy/https_proxy/no_proxy via environment variables

set -e

CUDA_ARCH=${1:-""}
BASE_IMAGE=${2:-""}
IMAGE_TAG="cpmcu:cuda12.8-release"

DEFAULT_CUDA_ARCH="80,86,87,89,90,120"
# Default base image (prefer nvcr.io variant)
DEFAULT_BASE_IMAGE="docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel"

echo "=== CPM.cu Docker Build Config ==="
echo "Build type: Release (production optimized)"
echo "Image tag: $IMAGE_TAG"

# Show base image info
if [ -n "$BASE_IMAGE" ]; then
    echo "Base image: $BASE_IMAGE (custom)"
else
    echo "Base image: $DEFAULT_BASE_IMAGE (default)"
    # Check whether the nvcr.io image exists locally
    if docker image inspect docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel >/dev/null 2>&1; then
        echo "✅ Detected local nvcr.io image, will use local copy"
    else
        echo "⚠️  No local nvcr.io image, will pull from registry"
    fi
fi

# Show CUDA arch info
if [ -n "$CUDA_ARCH" ]; then
    echo "CUDA arch: $CUDA_ARCH (custom)"
else
    echo "CUDA arch: $DEFAULT_CUDA_ARCH (default for mainstream GPUs)"
fi
echo "================================"

# Proxy config (read from environment by priority)
HTTP_PROXY_VAL=${http_proxy:-${HTTP_PROXY:-""}}
HTTPS_PROXY_VAL=${https_proxy:-${HTTPS_PROXY:-""}}
NO_PROXY_VAL=${no_proxy:-${NO_PROXY:-""}}

if [ -n "$HTTP_PROXY_VAL" ] || [ -n "$HTTPS_PROXY_VAL" ]; then
  echo "Proxy: detected from environment"
  [ -n "$HTTP_PROXY_VAL" ] && echo "  http_proxy=$HTTP_PROXY_VAL"
  [ -n "$HTTPS_PROXY_VAL" ] && echo "  https_proxy=$HTTPS_PROXY_VAL"
  [ -n "$NO_PROXY_VAL" ] && echo "  no_proxy=$NO_PROXY_VAL"
else
  echo "Proxy: not detected. If needed, export in this shell:"
  echo "  export http_proxy=http://USER:PASS@HOST:PORT"
  echo "  export https_proxy=http://USER:PASS@HOST:PORT"
  echo "  export no_proxy=localhost,127.0.0.1,::1"
fi

# Build args
BUILD_ARGS=""
if [ -n "$CUDA_ARCH" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg CUDA_ARCH=$CUDA_ARCH"
fi
if [ -n "$BASE_IMAGE" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg BASE_IMAGE=$BASE_IMAGE"
fi
# Pass proxy args to build
if [ -n "$HTTP_PROXY_VAL" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg http_proxy=$HTTP_PROXY_VAL"
fi
if [ -n "$HTTPS_PROXY_VAL" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg https_proxy=$HTTPS_PROXY_VAL"
fi
if [ -n "$NO_PROXY_VAL" ]; then
    BUILD_ARGS="$BUILD_ARGS --build-arg no_proxy=$NO_PROXY_VAL"
fi

# Extra docker build flags
DOCKER_FLAGS=""
# If proxy targets 127.0.0.1/localhost/host.docker.internal add host mapping
if [ -n "$HTTP_PROXY_VAL$HTTPS_PROXY_VAL" ]; then
  if echo "$HTTP_PROXY_VAL $HTTPS_PROXY_VAL" | grep -qiE '127\.0\.0\.1|localhost|host\.docker\.internal'; then
    DOCKER_FLAGS="$DOCKER_FLAGS --add-host=host.docker.internal:host-gateway"
    echo "Added: --add-host=host.docker.internal:host-gateway"
  fi
fi
# Optional: use host network (set USE_HOST_NETWORK=1)
if [ "${USE_HOST_NETWORK:-0}" = "1" ]; then
  DOCKER_FLAGS="$DOCKER_FLAGS --network=host"
  echo "Enabled: --network=host"
fi

# Build
echo "Start building Docker image (no cache)..."
echo "docker build --no-cache $DOCKER_FLAGS $BUILD_ARGS -t $IMAGE_TAG ."
docker build --no-cache $DOCKER_FLAGS $BUILD_ARGS -t $IMAGE_TAG .

echo "Build completed!"
echo ""
echo "How to run:"
echo "  # Run a container"
echo "  docker run --gpus all -it $IMAGE_TAG"
echo ""
echo "  # Run API server"
echo "  docker run --gpus all -p 8000:8000 -it $IMAGE_TAG python examples/minicpm4/start_server.py"
echo ""
echo "  # Run examples"
echo "  docker run --gpus all -it $IMAGE_TAG python examples/minicpm4/test_generate.py --help"

# Build notes
echo ""
echo "=== Image features ==="
echo "✅ Out-of-the-box: supports mainstream GPUs"
echo "✅ Data types: fp16, bf16"
echo "✅ Release mode: optimized for performance"
echo "✅ Memory-friendly build: 4 compile jobs"
echo ""
echo "=== Supported GPU arch ==="
echo "80: A100, A800 (data center)"
echo "86: RTX 3090, RTX 3080, RTX 3070 (consumer)"
echo "87: Jetson Orin (edge)"
echo "89: RTX 4090, RTX 4080 (consumer)"
echo "90: H100, H800 (data center)"
echo "120: RTX 5090, RTX 5070Ti (Blackwell)"
echo ""
echo "=== Examples ==="
echo "# Basic"
echo "./build.sh                                    # default build"
echo "./build.sh \"80,90\"                          # custom CUDA arch"
echo ""
echo "# Specify base image"
echo "./build.sh \"\" docker.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel  # use Docker Hub image"
echo "./build.sh \"86,89,120\" nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04  # full custom"
echo ""
echo "=== CUDA arch examples ==="
echo "\"80,90,120\"     # data center only"
echo "\"86,89\"     # consumer only"
echo "\"87\"        # Jetson only"
echo "\"120\"       # Blackwell only"
echo "=========================="
