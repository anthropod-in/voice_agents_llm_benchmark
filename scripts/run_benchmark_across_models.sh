#!/usr/bin/env bash
set -euo pipefail

# Run benchmark across multiple model/service cases, then judge each run.
#
# Available cases include 6 cases:
#   - openai:gpt-4.1
#   - openai:gpt-4.1-mini
#   - azure:gpt-4.1
#   - azure:gpt-4.1-mini
#   - google:gemini-2.5-flash-lite
#   - google:gemini-2.5-flash
#
# Default --cases all excludes azure-gpt-4.1 from comparison due to
# quota/latency instability. It can still be run explicitly.
#
# Dynamic case selection:
#   --cases openai-gpt-4.1,google-gemini-2.5-flash
#   --cases all
#
# Usage:
#   bash scripts/run_benchmark_across_models.sh
#   bash scripts/run_benchmark_across_models.sh --runs-per-case 2
#   bash scripts/run_benchmark_across_models.sh --cases openai-gpt-4.1,azure-gpt-4.1
#   bash scripts/run_benchmark_across_models.sh --list-cases

BENCHMARK="aiwf_medium_context"
JUDGE_MODEL="gemini-2.5-flash"
RUNS_PER_CASE=1
CASES_ARG="all"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_benchmark_across_models.sh [options]

Options:
  --benchmark <name>         Benchmark name (default: aiwf_medium_context)
  --judge-model <name>       Judge model (default: gemini-2.5-flash)
  --runs-per-case <n>        Number of runs per selected case (default: 1)
  --cases <csv|all>          Comma-separated case IDs, or "all"
  --list-cases               Print available case IDs and exit
  -h, --help                 Show this help message
EOF
}

# case_id|provider_label|model|service
ALL_CASES=(
  "openai-gpt-4.1|openai|gpt-4.1|openai"
  "openai-gpt-4.1-mini|openai|gpt-4.1-mini|openai"
  "azure-gpt-4.1|azure|gpt-4.1|pipecat.services.azure.llm.AzureLLMService"
  "azure-gpt-4.1-mini|azure|gpt-4.1-mini|pipecat.services.azure.llm.AzureLLMService"
  "google-gemini-2.5-flash-lite|google|gemini-2.5-flash-lite|google"
  "google-gemini-2.5-flash|google|gemini-2.5-flash|google"
)

# Comparison defaults used when --cases all
DEFAULT_CASES=(
  "openai-gpt-4.1|openai|gpt-4.1|openai"
  "openai-gpt-4.1-mini|openai|gpt-4.1-mini|openai"
  "azure-gpt-4.1-mini|azure|gpt-4.1-mini|pipecat.services.azure.llm.AzureLLMService"
  "google-gemini-2.5-flash-lite|google|gemini-2.5-flash-lite|google"
  "google-gemini-2.5-flash|google|gemini-2.5-flash|google"
)

list_cases() {
  echo "Available case IDs:"
  for row in "${ALL_CASES[@]}"; do
    IFS='|' read -r case_id provider model service <<<"${row}"
    echo "  - ${case_id} (provider=${provider}, model=${model}, service=${service})"
  done
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
    --runs-per-case)
      RUNS_PER_CASE="$2"
      shift 2
      ;;
    --cases)
      CASES_ARG="$2"
      shift 2
      ;;
    --list-cases)
      list_cases
      exit 0
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

declare -a SELECTED_CASES=()
if [[ "${CASES_ARG}" == "all" ]]; then
  SELECTED_CASES=("${DEFAULT_CASES[@]}")
else
  IFS=',' read -r -a requested_ids <<<"${CASES_ARG}"
  for req in "${requested_ids[@]}"; do
    found=0
    for row in "${ALL_CASES[@]}"; do
      IFS='|' read -r case_id provider model service <<<"${row}"
      if [[ "${req}" == "${case_id}" ]]; then
        SELECTED_CASES+=("${row}")
        found=1
        break
      fi
    done
    if [[ "${found}" -eq 0 ]]; then
      echo "ERROR: Unknown case id '${req}'"
      list_cases
      exit 1
    fi
  done
fi

declare -a SELECTED_MODEL_KEYS=()
for row in "${SELECTED_CASES[@]}"; do
  IFS='|' read -r _case_id provider model _service <<<"${row}"
  SELECTED_MODEL_KEYS+=("${provider}:${model}")
done
SELECTED_MODELS_CSV="$(IFS=,; echo "${SELECTED_MODEL_KEYS[*]}")"

run_and_judge_once() {
  local model="$1"
  local service="$2"
  local provider="$3"

  local out
  out="$(uv run multi-turn-eval run "${BENCHMARK}" --model "${model}" --service "${service}" 2>&1)"
  echo "${out}"

  local run_dir
  run_dir="$(printf '%s\n' "${out}" | awk -F'Output directory: ' '/Output directory:/ {print $2}' | tail -n 1)"
  if [[ -z "${run_dir}" ]]; then
    echo "ERROR: Could not parse run directory from output."
    exit 1
  fi

  echo "Judging run: ${run_dir}"
  uv run multi-turn-eval judge "${run_dir}" --judge-model "${JUDGE_MODEL}"
}

echo
echo "================================================================================"
echo "Benchmark:      ${BENCHMARK}"
echo "Judge model:    ${JUDGE_MODEL}"
echo "Runs per case:  ${RUNS_PER_CASE}"
echo "Selected cases: ${#SELECTED_CASES[@]}"
echo "================================================================================"
echo

for row in "${SELECTED_CASES[@]}"; do
  IFS='|' read -r case_id provider model service <<<"${row}"
  echo
  echo "================================================================================"
  echo "CASE: ${case_id}"
  echo "provider=${provider}"
  echo "model=${model}"
  echo "service=${service}"
  echo "================================================================================"

  for ((i=1; i<=RUNS_PER_CASE; i++)); do
    echo
    echo "[${case_id}] run ${i}/${RUNS_PER_CASE}"
    run_and_judge_once "${model}" "${service}" "${provider}"
  done

  echo
  echo "Quick aggregate for ${provider}:${model} (last ${RUNS_PER_CASE} run(s))"
  uv run python aggregate_results.py --benchmark "${BENCHMARK}" --model "${provider}:${model}" --runs "${RUNS_PER_CASE}"
done

echo
echo "================================================================================"
echo "Aggregate across selected models (last ${RUNS_PER_CASE} run(s) per model)"
echo "================================================================================"
uv run python aggregate_results.py --benchmark "${BENCHMARK}" --models "${SELECTED_MODELS_CSV}" --runs "${RUNS_PER_CASE}"

echo
echo "Done."
