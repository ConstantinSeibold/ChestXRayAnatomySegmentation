#!/usr/bin/env bash
# Build the CXAS image. The weights are passed as a named build context; all code comes from this repo.
#   WEIGHTS=~/.cxas/weights docker/build.sh [cpu|cu128] [tag]
set -euo pipefail
FLAVOR=${1:-cpu}
HERE=$(cd "$(dirname "$0")/.." && pwd)
VERSION=$(sed -n 's/^ *version="\(.*\)",/\1/p' "$HERE/setup.py")
TAG=${2:-cseibold/cxas:${VERSION}-${FLAVOR}}
WEIGHTS=${WEIGHTS:-$HOME/.cxas/weights}

# Stage the weights as hard links where possible, symlinks resolved, so the context holds real files only.
STAGE=$(mktemp -d "${HERE}/.weights_stage.XXXX")
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/weights"
for m in UNet_ResNet50_default.pth M2F_anatomy_v2 landmarks_v1; do
  src=$(readlink -f "$WEIGHTS/$m")
  cp -al "$src" "$STAGE/weights/$m" 2>/dev/null || cp -r "$src" "$STAGE/weights/$m"
done
echo "cxas $(git -C "$HERE" rev-parse --short HEAD)$(git -C "$HERE" diff --quiet || echo +dirty)" > "$STAGE/weights/SOURCES"
du -shL "$STAGE"/*

docker build --platform linux/amd64 \
  --build-arg TORCH_FLAVOR="$FLAVOR" \
  --build-context weights="$STAGE/weights" \
  -f "$HERE/docker/Dockerfile" -t "$TAG" "$HERE"
