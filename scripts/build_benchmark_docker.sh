#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-llm-benchmark:runner}"

echo "Building Docker image: ${IMAGE_NAME}"
docker build -f Dockerfile.benchmark -t "${IMAGE_NAME}" .
echo "Build complete: ${IMAGE_NAME}"
