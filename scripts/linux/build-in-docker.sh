#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?usage: build-in-docker.sh <version> <arch> [out-dir]}"
ARCH="${2:?usage: build-in-docker.sh <version> <arch> [out-dir]}"
OUT_DIR="${3:-${OUT_DIR:-build/linux}}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "$ARCH" in
    x86_64 | amd64)
        ARCH="x86_64"
        PLATFORM="${PLATFORM:-linux/amd64}"
        ;;
    aarch64 | arm64)
        ARCH="aarch64"
        PLATFORM="${PLATFORM:-linux/arm64}"
        ;;
    *)
        echo "unsupported architecture: $ARCH" >&2
        exit 1
        ;;
esac

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

echo "Building Linux packages for $ARCH on Ubuntu 22.04 ($PLATFORM)"
docker run --rm \
    --platform "$PLATFORM" \
    -v "$ROOT":/work \
    -v "$OUT_DIR":/out \
    -w /work \
    -e VERSION="$VERSION" \
    -e ARCH="$ARCH" \
    ubuntu:22.04 \
    bash -lc '
        set -euo pipefail
        apt-get update -qq
        if ! DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
            build-essential \
            ca-certificates \
            cmake \
            curl \
            dpkg-dev \
            file \
            libgtk-4-dev \
            pkg-config \
            rpm >/tmp/apt.log 2>&1; then
            cat /tmp/apt.log
            exit 1
        fi
        if ! curl -sSf https://sh.rustup.rs | \
            sh -s -- -y --profile minimal --default-toolchain 1.98.0 >/tmp/rustup.log 2>&1; then
            cat /tmp/rustup.log
            exit 1
        fi
        export PATH="/root/.cargo/bin:$PATH"
        export CARGO_TARGET_DIR=/tmp/target
        scripts/linux/package.sh "$VERSION" "$ARCH" /out
    '
