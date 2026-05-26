#!/bin/bash
set -euo pipefail

IMAGE="${DCU_CI_IMAGE:-}"
CONTAINER_NAME="${DCU_CI_CONTAINER_NAME:-ci_sglang}"
CACHE_HOST="${DCU_CI_CACHE_HOST:-/home/runner/sgl-data}"
CUSTOM_DEVICE_FLAGS="${DCU_CI_DEVICE_FLAGS:-}"
SKIP_PULL="${DCU_CI_SKIP_PULL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --container-name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --cache-host)
      CACHE_HOST="$2"
      shift 2
      ;;
    --device-flags)
      CUSTOM_DEVICE_FLAGS="$2"
      shift 2
      ;;
    --skip-pull)
      SKIP_PULL="1"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 --image IMAGE [--container-name NAME] [--cache-host PATH] [--device-flags FLAGS] [--skip-pull]"
      exit 0
      ;;
    *)
      echo "Unknown option $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${IMAGE}" ]]; then
  echo "DCU_CI_IMAGE or --image is required" >&2
  exit 1
fi

DEVICE_ARGS=()
if [[ -n "${CUSTOM_DEVICE_FLAGS}" ]]; then
  read -r -a DEVICE_ARGS <<< "${CUSTOM_DEVICE_FLAGS}"
else
  if [[ -e /dev/kfd ]]; then
    DEVICE_ARGS+=(--device=/dev/kfd)
  fi
  if [[ -e /dev/dri ]]; then
    DEVICE_ARGS+=(--device=/dev/dri)
  fi
fi

CACHE_VOLUME=()
if [[ -d "${CACHE_HOST}" ]]; then
  CACHE_VOLUME=(-v "${CACHE_HOST}:/sgl-data")
fi

HOST_RUNTIME_VOLUMES=()
if [[ -d /opt/hyhal ]]; then
  HOST_RUNTIME_VOLUMES+=(-v /opt/hyhal:/opt/hyhal:ro)
fi
if [[ -d /public/opendas/DL_DATA/llm-models ]]; then
  HOST_RUNTIME_VOLUMES+=(-v /public/opendas/DL_DATA/llm-models:/public/opendas/DL_DATA/llm-models:ro)
fi

if [[ -z "${SKIP_PULL}" ]]; then
  docker pull "${IMAGE}"
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}"
fi

docker run -dt --user root \
  "${DEVICE_ARGS[@]}" \
  --ulimit nofile=65536:65536 \
  -v "${GITHUB_WORKSPACE:-$PWD}:/sglang-checkout" \
  "${CACHE_VOLUME[@]}" \
  "${HOST_RUNTIME_VOLUMES[@]}" \
  --group-add video \
  --shm-size 32g \
  --cap-add=SYS_PTRACE \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME=/sgl-data/hf-cache \
  -e HF_HUB_ETAG_TIMEOUT=300 \
  -e HF_HUB_DOWNLOAD_TIMEOUT=300 \
  -e SGLANG_IS_IN_CI=1 \
  -e SGLANG_IS_IN_CI_DCU=1 \
  --security-opt seccomp=unconfined \
  -w /sglang-checkout \
  --name "${CONTAINER_NAME}" \
  "${IMAGE}"

docker exec "${CONTAINER_NAME}" git config --global --add safe.directory /sglang-checkout
