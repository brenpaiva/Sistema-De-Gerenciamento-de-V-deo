#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-config/demo.yaml}"
GPU_MODE="${GPU_MODE:-auto}"
HW_DECODE="${HW_DECODE:-auto}"
IMAGE="${IMAGE:-gesec-viewer:latest}"

if [ -z "${DISPLAY:-}" ]; then
  export DISPLAY=:0
fi

if command -v xhost >/dev/null 2>&1; then
  xhost +local:docker >/dev/null
fi

gpu_args=()
if [ "${USE_NVIDIA:-0}" = "1" ]; then
  gpu_args=(--gpus all)
fi

docker build -t "$IMAGE" .

docker run --rm -it \
  "${gpu_args[@]}" \
  --network host \
  -e DISPLAY="$DISPLAY" \
  -e QT_QPA_PLATFORM=xcb \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)":/app \
  "$IMAGE" \
  python -m gesec_viewer --config "$CONFIG" --gpu "$GPU_MODE" --hw-decode "$HW_DECODE"
