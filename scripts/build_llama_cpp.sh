#!/usr/bin/env bash
# Build llama.cpp's llama-server natively on the Jetson (CUDA, sm_87).
#
# Alternative to the NVIDIA container (systemd/llama-server.service) when
# docker misbehaves — no runtime dependency on docker, and the same GGUF
# Ollama pulled. Needs: apt install cmake cuda-toolkit-12-6 (~2.5 GB).
#
#   sudo scripts/build_llama_cpp.sh            # → /opt/llama.cpp/bin/llama-server
#   sudo cp systemd/llama-server-native.service /etc/systemd/system/llama-server.service
#
# Build takes ~20-30 min on the 6 A78AE cores; -j4 keeps the radio usable.
set -euo pipefail

PREFIX=${PREFIX:-/opt/llama.cpp}
SRC=${SRC:-/opt/llama.cpp/src}
REF=${LLAMA_CPP_REF:-master}
JOBS=${JOBS:-4}
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=${CUDACXX:-/usr/local/cuda/bin/nvcc}

command -v cmake >/dev/null || { echo "cmake missing: apt install cmake"; exit 1; }
[ -x "$CUDACXX" ] || { echo "nvcc missing: apt install cuda-toolkit-12-6"; exit 1; }

mkdir -p "$PREFIX"
if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 --branch "$REF" https://github.com/ggml-org/llama.cpp "$SRC"
else
    git -C "$SRC" fetch --depth 1 origin "$REF" && git -C "$SRC" checkout -q FETCH_HEAD
fi

cmake -S "$SRC" -B "$SRC/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$SRC/build" --config Release -j "$JOBS" --target llama-server llama-bench

mkdir -p "$PREFIX/bin" "$PREFIX/lib"
cp "$SRC/build/bin/llama-server" "$SRC/build/bin/llama-bench" "$PREFIX/bin/"
# Shared ggml/llama libs live next to the binaries (rpath $ORIGIN/../lib).
find "$SRC/build/bin" -name "*.so*" -exec cp -a {} "$PREFIX/lib/" \; 2>/dev/null || true
find "$SRC/build" -name "libggml*.so*" -o -name "libllama*.so*" | xargs -r -I{} cp -a {} "$PREFIX/lib/"
echo "installed: $PREFIX/bin/llama-server ($(git -C "$SRC" rev-parse --short HEAD))"
LD_LIBRARY_PATH="$PREFIX/lib" "$PREFIX/bin/llama-server" --version
