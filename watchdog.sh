#!/usr/bin/env bash
# Watchdog: keeps the Gemma 4 E4B stack alive on Daytona.
# The sandbox can stay "Started" while the bot or the model server dies
# (crash, OOM on the 8 GiB tier, a recycled session). This script checks
# every 60 seconds and restarts whatever died, and writes each death to
# watchdog.log so the cause is visible next time.
# First time: run `bash start.sh` once (it asks for the token), then:
#   nohup bash watchdog.sh > watchdog.out 2>&1 &
cd "$(dirname "$0")"
exec 9>watchdog.lock
flock -n 9 || { echo "watchdog already running"; exit 0; }

[ -f secrets.env ] || { echo "run bash start.sh once first (it asks for the token)"; exit 1; }
set -a; source secrets.env; set +a
source config.env   # MODEL_PATH=... written by setup.sh

start_server() {
  : > server.log
  ./llama/llama-server -m "$MODEL_PATH" --jinja -c 32768 -t 4 \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
    --host 127.0.0.1 --port 8080 --alias gemma-4-e4b >> server.log 2>&1 &
}
start_bot() {
  PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
  nohup "$PY" bot.py >> bot.log 2>&1 &
}
healthy() { curl -s http://127.0.0.1:8080/health | grep -q '"ok"'; }
wait_up() { for i in $(seq 1 90); do healthy && return 0; sleep 2; done; return 1; }

# initial bring-up
pgrep -f llama-server >/dev/null || start_server
wait_up
pgrep -f "python.*bot\.py" >/dev/null || start_bot
echo "[$(date '+%F %T')] watchdog up" >> watchdog.log

fails=0
while true; do
  sleep 60
  if ! pgrep -f llama-server >/dev/null; then
    echo "[$(date '+%F %T')] server process gone — restarting stack" >> watchdog.log
    pkill -9 -f "python.*bot\.py"; sleep 1
    start_server; wait_up; start_bot; fails=0
  elif ! healthy; then
    fails=$((fails + 1))
    if [ "$fails" -ge 5 ]; then
      echo "[$(date '+%F %T')] server unhealthy 5 checks in a row — restarting stack" >> watchdog.log
      pkill -f llama-server; pkill -9 -f "python.*bot\.py"; sleep 2
      start_server; wait_up; start_bot; fails=0
    fi
  else
    fails=0
    if ! pgrep -f "python.*bot\.py" >/dev/null; then
      echo "[$(date '+%F %T')] bot process gone — restarting bot" >> watchdog.log
      start_bot
    fi
  fi
done
