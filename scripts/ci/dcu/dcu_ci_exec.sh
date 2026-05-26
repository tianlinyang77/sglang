#!/bin/bash
set -euo pipefail

CONTAINER_NAME="${DCU_CI_CONTAINER_NAME:-ci_sglang}"
WORKDIR="/sglang-checkout/test"
declare -A ENV_MAP=(
  [SGLANG_IS_IN_CI]=1
  [SGLANG_IS_IN_CI_DCU]=1
  [PYTHONDONTWRITEBYTECODE]=1
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workdir)
      WORKDIR="$2"
      shift 2
      ;;
    -e)
      IFS="=" read -r key val <<< "$2"
      ENV_MAP["$key"]="$val"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

CHECKOUT_ROOT="${WORKDIR%/test}"
if [[ "${CHECKOUT_ROOT}" == "${WORKDIR}" ]]; then
  CHECKOUT_ROOT="/sglang-checkout"
fi

if [[ -z "${ENV_MAP[PYTHONPATH]+set}" ]]; then
  ENV_MAP[PYTHONPATH]="${CHECKOUT_ROOT}/python"
fi

ENV_ARGS=()
for key in "${!ENV_MAP[@]}"; do
  ENV_ARGS+=("-e" "$key=${ENV_MAP[$key]}")
done

if docker exec \
  -w "$WORKDIR" \
  "${ENV_ARGS[@]}" \
  "$CONTAINER_NAME" "$@"; then
  exit 0
else
  FIRST_EXIT_CODE=$?
fi

echo "First attempt failed with exit code $FIRST_EXIT_CODE"

if [[ "$FIRST_EXIT_CODE" -eq 1 || "$FIRST_EXIT_CODE" -eq 137 || "$FIRST_EXIT_CODE" -eq 255 ]]; then
  echo "Exit code $FIRST_EXIT_CODE indicates test failure, not retrying"
  exit "$FIRST_EXIT_CODE"
fi

echo "Retrying with HF_HUB_OFFLINE=1 to use cached models"

docker exec \
  -w "$WORKDIR" \
  "${ENV_ARGS[@]}" \
  -e HF_HUB_OFFLINE=1 \
  "$CONTAINER_NAME" "$@"
