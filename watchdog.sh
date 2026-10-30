#!/usr/bin/env bash
# Watchdog v2: keeps the Gemma 4 E4B stack alive on Daytona.
#
# v1 only checked that the bot PROCESS exists. On 2026-10-30 the bot was
# found alive but silently dead for ~4 days: Telegram had closed its
# pooled connections (sockets stuck in CLOSE-WAIT) and the long-poll
# never recovered, while this watchdog saw a living process and a healthy
# server and did nothing. v2 supervises the bot's heartbeat.txt instead
# (bot.py refreshes it every 60 s after a successful Telegram probe):
# a stale heartbeat means the bot is wedged -> SIGKILL + restart.
# SIGKILL, not SIGTERM: PTB traps SIGTERM for a graceful shutdown that a
# wedged event loop can never complete (proven 2026-10-30).
#
# Normally you never start this by hand: `bash start.sh` launches it
# detached (and it brings up the server + bot). Manual start, if needed:
#   setsid nohup bash watchdog.sh > watchdog.out 2>&1 &
cd "$(dirname "$0")"
exec 9>watchdog.lock
flock -n 9 || { echo "watchdog already running"; exit 0; }

[ -f secrets.env ] || { echo "run bash start.sh once first (it asks for the token)"; exit 1; }
set -a; source secrets.env; set +a
source config.env   # MODEL_PATH=... written by setup.sh

start_server() {
  : > server.log
  ./llama/llama-server -m "$MODEL_PATH" --jinja -c 65536 -t 4 \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
    --host 127.0.0.1 --port 8080 --alias gemma-4-e4b >> server.log 2>&1 &
}
start_bot() {
  PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
  rm -f heartbeat.txt
  nohup "$PY" bot.py >> bot.log 2>&1 &
}
healthy() { curl -s http://127.0.0.1:8080/health | grep -q '"ok"'; }
wait_up() { for i in $(seq 1 90); do healthy && return 0; sleep 2; done; return 1; }
hb_fresh() {
  [ -f heartbeat.txt ] || return 1
  local now mt
  now=$(date +%s); mt=$(stat -c %Y heartbeat.txt 2>/dev/null || echo 0)
  [ $((now - mt)) -le 180 ]
}

# initial bring-up
pgrep -f llama-server >/dev/null || start_server
wait_up
pgrep -f "python.*bot\.py" >/dev/null || start_bot
echo "[$(date '+%F %T')] watchdog v2 up" >> watchdog.log

fails=0
hb_bad=0
while true; do
  sleep 60
  if ! pgrep -f llama-server >/dev/null; then
    echo "[$(date '+%F %T')] server process gone — restarting stack" >> watchdog.log
    pkill -9 -f "python.*bot\.py"; sleep 1
    start_server; wait_up; start_bot; fails=0; hb_bad=0
  elif ! healthy; then
    fails=$((fails + 1))
    if [ "$fails" -ge 5 ]; then
      echo "[$(date '+%F %T')] server unhealthy 5 checks in a row — restarting stack" >> watchdog.log
      pkill -f llama-server; pkill -9 -f "python.*bot\.py"; sleep 2
      start_server; wait_up; start_bot; fails=0; hb_bad=0
    fi
  else
    fails=0
    if ! pgrep -f "python.*bot\.py" >/dev/null; then
      echo "[$(date '+%F %T')] bot process gone — restarting bot" >> watchdog.log
      start_bot; hb_bad=0
    elif ! hb_fresh; then
      hb_bad=$((hb_bad + 1))
      if [ "$hb_bad" -ge 4 ]; then
        echo "[$(date '+%F %T')] bot heartbeat stale — wedged bot, SIGKILL + restart" >> watchdog.log
        pkill -9 -f "python.*bot\.py"; sleep 1
        start_bot; hb_bad=0
      fi
    else
      hb_bad=0
    fi
  fi
done
