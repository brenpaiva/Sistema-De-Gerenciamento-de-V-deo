#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for this optional RTSP demo."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "FFmpeg is required to publish the demo files as RTSP streams."
  exit 1
fi

if [ ! -f "media/demo_cam1.mp4" ] || [ ! -f "media/demo_cam2.mp4" ]; then
  echo "Demo videos were not found. Run: python scripts/generate_demo_videos.py"
  exit 1
fi

docker rm -f gesec-mediamtx >/dev/null 2>&1 || true
docker run -d --name gesec-mediamtx -p 8554:8554 bluenviron/mediamtx:latest >/dev/null

echo "Publishing RTSP streams:"
echo "  rtsp://localhost:8554/cam1"
echo "  rtsp://localhost:8554/cam2"

ffmpeg -re -stream_loop -1 -i media/demo_cam1.mp4 -an -c:v libx264 -preset veryfast -tune zerolatency -f rtsp rtsp://localhost:8554/cam1 &
pid1=$!
ffmpeg -re -stream_loop -1 -i media/demo_cam2.mp4 -an -c:v libx264 -preset veryfast -tune zerolatency -f rtsp rtsp://localhost:8554/cam2 &
pid2=$!

trap 'kill "$pid1" "$pid2"; docker rm -f gesec-mediamtx >/dev/null 2>&1 || true' EXIT
wait
