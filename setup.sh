#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Daytona Gemma 4 E4B + Telegram bot — one-time setup
# Downloads: llama.cpp (prebuilt binary), the Gemma 4 E4B GGUF model, Python deps.
# Everything lands inside this folder. Re-running is safe (downloads resume/skip).
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

# ── Which model? ─────────────────────────────────────────────────────────────
# DEFAULT: UNCENSORED — HauhauCS Gemma 4 E4B Aggressive, Q4_K_P, ~5.8 GB.
# (K_P is the model card's quality pick: quality preserved ~1-2 quant levels
#  better than Q4_K_M at only ~5-15% larger file size.)
MODEL_URL="${MODEL_URL:-https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/resolve/main/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf}"
#
# Alternatives — run setup with MODEL_URL set to one of these instead:
#   Q4_K_M uncensored (5.34 GB): https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/resolve/main/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
#   Official Google, censored, QAT Q4_0 (5.15 GB): https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILE="$(basename "${MODEL_URL%%\?*}")"
mkdir -p models llama

echo "==> [1/3] llama.cpp binary"
if [ ! -x llama/llama-server ]; then
  # NOTE: /releases/latest points to a marker release with NO binaries in it,
  # so search the recent build releases (bXXXX) for the plain CPU asset instead.
  ASSET_URL="$(curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5" \
    | grep -o '"browser_download_url": *"[^"]*"' \
    | grep -i 'bin-ubuntu-x64\.tar\.gz' \
    | head -1 | cut -d'"' -f4 || true)"
  # Pinned fallback if the API gave nothing (rate limit, layout change, ...).
  [ -z "$ASSET_URL" ] && ASSET_URL="https://github.com/ggml-org/llama.cpp/releases/download/b11856/llama-b11856-bin-ubuntu-x64.tar.gz"
  echo "    downloading: $ASSET_URL"
  mkdir -p llama/extract
  if curl -fL -o llama/pkg "$ASSET_URL"; then
    if [ "$(head -c2 llama/pkg)" = "PK" ]; then unzip -oq llama/pkg -d llama/extract; else tar -xzf llama/pkg -C llama/extract; fi
    # The release nests binaries one folder deep — flatten it.
    BIN_PATH="$(find llama/extract -name llama-server -type f | head -1)"
    [ -n "$BIN_PATH" ] && cp -r "$(dirname "$BIN_PATH")"/. llama/
    chmod +x llama/llama-server 2>/dev/null || true
  fi
  rm -rf llama/extract llama/pkg
fi
if [ ! -x llama/llama-server ]; then
  echo "    prebuilt download failed — building from source (slower, needs git+cmake+gcc)"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp llama-src
  cmake -B llama-src/build -S llama-src -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build llama-src/build -j "$(nproc)" --target llama-server
  cp llama-src/build/bin/llama-server llama/
  cp llama-src/build/bin/*.so llama/ 2>/dev/null || true
fi
echo "    OK: $(./llama/llama-server --version 2>/dev/null | head -1 || echo 'binary ready')"

echo "==> [2/3] model: $MODEL_FILE"
if [ ! -s "models/$MODEL_FILE" ]; then
  # -C - resumes an interrupted download; the file is ~5.8 GB for the default Q4_K_P.
  curl -L -C - -o "models/$MODEL_FILE" "$MODEL_URL"
fi
ls -lh "models/$MODEL_FILE"
echo "MODEL_PATH=models/$MODEL_FILE" > config.env

echo "==> [3/3] Python deps (python-telegram-bot)"
if python3 -m venv .venv 2>/dev/null; then
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt
else
  echo "    venv unavailable — installing into the sandbox Python directly"
  pip3 install -q --break-system-packages -r requirements.txt || pip3 install -q -r requirements.txt
fi

echo ""
echo "Setup done. Next:"
echo "  export BOT_TOKEN=\"<token from BotFather>\""
echo "  bash run.sh"
