#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-llm-benchmark:runner}"
BENCHMARK="aiwf_medium_context"
JUDGE_MODEL="gemini-2.5-flash"
CASES="all"
RUNS_PER_CASE=1

usage() {
  cat <<'EOF'
Run benchmark inside Docker (EC2-friendly).

Usage:
  bash scripts/run_benchmark_docker.sh [options]

Options:
  --benchmark <name>         Benchmark name (default: aiwf_medium_context)
  --judge-model <name>       Judge model (default: gemini-2.5-flash)
  --cases <csv|all>          Case IDs to run (default: all)
  --runs-per-case <n>        Number of runs per selected case (default: 1)
  --image <name>             Docker image tag (default: llm-benchmark:runner)
  -h, --help                 Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
    --judge-model)
      JUDGE_MODEL="$2"
      shift 2
      ;;
    --cases)
      CASES="$2"
      shift 2
      ;;
    --runs-per-case)
      RUNS_PER_CASE="$2"
      shift 2
      ;;
    --image)
      IMAGE_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if ! [[ "${RUNS_PER_CASE}" =~ ^[0-9]+$ ]] || [[ "${RUNS_PER_CASE}" -lt 1 ]]; then
  echo "ERROR: --runs-per-case must be a positive integer."
  exit 1
fi

if [[ ! -f ".env" ]]; then
  echo "ERROR: .env file not found in repository root."
  echo "Create .env with API keys before running."
  exit 1
fi

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Docker image ${IMAGE_NAME} not found. Building..."
  bash scripts/build_benchmark_docker.sh
fi

echo "Running benchmark in Docker..."
echo "  image:         ${IMAGE_NAME}"
echo "  benchmark:     ${BENCHMARK}"
echo "  judge model:   ${JUDGE_MODEL}"
echo "  cases:         ${CASES}"
echo "  runs per case: ${RUNS_PER_CASE}"

docker run --rm -it \
  -v "$PWD":/workspace \
  --env-file "$PWD/.env" \
  -e HOME=/tmp \
  -e UV_PYTHON=3.12 \
  "${IMAGE_NAME}" \
  bash -lc "uv sync --python 3.12 && bash scripts/run_benchmark_across_models.sh --benchmark \"${BENCHMARK}\" --judge-model \"${JUDGE_MODEL}\" --cases \"${CASES}\" --runs-per-case \"${RUNS_PER_CASE}\""
