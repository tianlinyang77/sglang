#!/bin/bash
set -euo pipefail

CONTAINER_NAME="${DCU_CI_CONTAINER_NAME:-ci_sglang}"
SKIP_SGLANG_BUILD="${DCU_CI_SKIP_SGLANG_BUILD:-}"
SKIP_TEST_TIME_DEPS="${DCU_CI_SKIP_TEST_TIME_DEPS:-}"
EXTRAS="${DCU_CI_PYTHON_EXTRAS:-dev_hip}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --container-name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --python-extras)
      EXTRAS="$2"
      shift 2
      ;;
    --skip-sglang-build)
      SKIP_SGLANG_BUILD="1"
      shift
      ;;
    --skip-test-time-deps)
      SKIP_TEST_TIME_DEPS="1"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--container-name NAME] [--python-extras EXTRAS] [--skip-sglang-build] [--skip-test-time-deps]"
      exit 0
      ;;
    *)
      echo "Unknown option $1" >&2
      exit 1
      ;;
  esac
done

install_with_retry() {
  local max_attempts=3
  local cmd="$*"

  for attempt in $(seq 1 "$max_attempts"); do
    echo "Attempt $attempt/$max_attempts: $cmd"
    if eval "$cmd"; then
      return 0
    fi
    if [[ "$attempt" -lt "$max_attempts" ]]; then
      sleep 5
    fi
  done

  echo "Failed after $max_attempts attempts: $cmd" >&2
  return 1
}

docker exec "$CONTAINER_NAME" python3 -m pip install --upgrade pip

if [[ -z "$SKIP_SGLANG_BUILD" ]]; then
  docker exec "$CONTAINER_NAME" pip uninstall sgl-kernel -y || true
  docker exec "$CONTAINER_NAME" pip uninstall sglang-kernel -y || true
  docker exec "$CONTAINER_NAME" pip uninstall sglang -y || true
  docker exec "$CONTAINER_NAME" find /opt/venv -name "*.pyc" -delete || true
  docker exec "$CONTAINER_NAME" find /opt/venv -name "__pycache__" -type d -exec rm -rf {} + || true
  docker exec "$CONTAINER_NAME" find /sglang-checkout -name "*.pyc" -delete || true
  docker exec "$CONTAINER_NAME" find /sglang-checkout -name "__pycache__" -type d -exec rm -rf {} + || true

  if docker exec "$CONTAINER_NAME" test -f /sglang-checkout/sgl-kernel/pyproject_rocm.toml; then
    docker exec -w /sglang-checkout/sgl-kernel "$CONTAINER_NAME" bash -c "rm -f pyproject.toml && mv pyproject_rocm.toml pyproject.toml && python3 setup_rocm.py install"
  fi

  if docker exec "$CONTAINER_NAME" test -f /sglang-checkout/python/pyproject_other.toml; then
    docker exec "$CONTAINER_NAME" bash -c "rm -rf /sglang-checkout/python/pyproject.toml && mv /sglang-checkout/python/pyproject_other.toml /sglang-checkout/python/pyproject.toml"
  fi

  install_with_retry docker exec "$CONTAINER_NAME" pip install -e "/sglang-checkout/python[${EXTRAS}]"
fi

if [[ -z "$SKIP_TEST_TIME_DEPS" ]]; then
  install_with_retry docker exec "$CONTAINER_NAME" pip install pytest requests psutil
fi
