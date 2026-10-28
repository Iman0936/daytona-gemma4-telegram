#!/usr/bin/env bash
# Starts llama-server (Gemma 4 E4B) in the background, then the Telegram bot.
# Stop everything with Ctrl+C, then:  pkill -f llama-server
set -e
cd "$(dirname "$0")"

: "${BOT_TOKEN:?Set BOT_TOKEN first:  export BOT_TOKEN=\"<token from BotFather>\"}"
source config.env   # MODEL_PATH=... written by setup.sh

# Gemma 4 on Daytona Tier 1 (4 vCPU / 8 GiB RAM) — LONG-MEMORY configuration:
#   -c 32768          : 32K context, so long conversations fit in the model's
#                       memory (native max is 128K — does not fit this RAM)
#   --cache-type q8_0 : KV cache in 8 bits instead of 16 — halves the cache
#                       (this is what makes 32K fit next to the ~5.8 GB
#                       weights at all) at negligible quality cost
#   --flash-attn on   : required for a quantized V cache in llama.cpp
#   -t 4              : one thread per vCPU
# FALLBACK if the server fails to start (see server.log): -c 16384 and remove
# both --cache-type flags and --flash-attn (full f16 cache, half the memory).
./llama/llama-server -m "$MODEL_PATH" --jinja -c 32768 -t 4 \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --host 127.0.0.1 --port 8080 --alias gemma-4-e4b > server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "Waiting for the model to load..."
for i in $(seq 1 90); do
  if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then echo "Model is up."; break; fi
  sleep 2
  if [ "$i" = "90" ]; then echo "Server did not come up — see server.log"; tail -20 server.log; exit 1; fi
done

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
echo "Starting Telegram bot (polling). Ctrl+C stops the bot AND the model server."
"$PY" bot.py
