#!/usr/bin/env bash
# One-command start: asks for the bot token + owner ID the first time
# (stored ONLY in this sandbox, in secrets.env), then starts everything.
# Later runs just reuse secrets.env — no need to enter anything again.
set -e
cd "$(dirname "$0")"

if [ ! -f secrets.env ]; then
  printf "Paste BOT_TOKEN (input hidden): "
  read -r -s BOT_TOKEN
  echo
  printf "Paste your numeric Telegram ID: "
  read -r ALLOWED_IDS
  printf "BOT_TOKEN=%s\nALLOWED_IDS=%s\nMAX_TOKENS=8192\n" "$BOT_TOKEN" "$ALLOWED_IDS" > secrets.env
  chmod 600 secrets.env
fi

set -a
source secrets.env
set +a
bash run.sh
