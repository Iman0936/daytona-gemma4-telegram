#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# THE one command for this package:
#
#     bash start.sh
#
# On a fresh sandbox it does everything, in front of you:
#   1. setup (llama.cpp binary + model + Python deps) — only what is missing
#   2. asks for the bot token + your numeric Telegram ID — FIRST time only,
#      validated live against Telegram, then stored ONLY in this sandbox
#      (secrets.env, chmod 600, gitignored — never in the repo)
#   3. brings the whole stack up detached (model server + bot + watchdog)
#      and waits until the server is really healthy
# Later runs just make sure everything is up — nothing is asked again.
# Everything keeps running after you close the terminal; the watchdog
# restarts whatever dies (including a silently wedged bot).
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"

# ── 1. setup, only what is missing ───────────────────────────────────────────
if [ ! -x llama/llama-server ] || [ ! -f config.env ] || ! ls models/*.gguf >/dev/null 2>&1 || [ ! -x .venv/bin/python ]; then
  echo "==> First run: setting up (llama.cpp + model + Python deps)..."
  bash setup.sh || { echo "❌ setup failed — see the output above"; exit 1; }
fi

# ── 2. secrets, first time only ──────────────────────────────────────────────
if [ ! -f secrets.env ]; then
  printf "Paste BOT_TOKEN (input hidden): "
  read -r -s BOT_TOKEN
  echo
  if ! curl -s --max-time 15 "https://api.telegram.org/bot${BOT_TOKEN}/getMe" | grep -q '"ok":true'; then
    echo "❌ Telegram rejected this token — nothing was saved. Run bash start.sh again."
    exit 1
  fi
  printf "Paste your numeric Telegram ID: "
  read -r ALLOWED_IDS
  ALLOWED_IDS="$(echo "$ALLOWED_IDS" | tr -d ' ')"
  if ! echo "$ALLOWED_IDS" | grep -qE '^[0-9]+(,[0-9]+)*$'; then
    echo "❌ That ID does not look numeric — nothing was saved. Run bash start.sh again."
    exit 1
  fi
  printf "BOT_TOKEN=%s\nALLOWED_IDS=%s\n" "$BOT_TOKEN" "$ALLOWED_IDS" > secrets.env
  chmod 600 secrets.env
  echo "✅ Token OK — saved in secrets.env (this sandbox only, never in git)."
fi

set -a
source secrets.env
set +a

healthy() { curl -s --max-time 5 http://127.0.0.1:8080/health | grep -q '"ok"'; }
bot_up() { pgrep -f "python.*bot\.py" >/dev/null; }
watchdog_up() { pgrep -f "watchdog\.sh" >/dev/null; }

# ── 3. already up? ───────────────────────────────────────────────────────────
if healthy && bot_up && watchdog_up; then
  echo "✅ Everything is already up (server + bot + watchdog)."
  exit 0
fi

# ── 4. watchdog first — it brings up / heals the stack ──────────────────────
if ! watchdog_up; then
  echo "==> Starting watchdog (it launches the server and the bot)..."
  setsid nohup bash watchdog.sh > watchdog.out 2>&1 &
fi

# ── 5. wait until the server is really healthy ──────────────────────────────
echo "==> Waiting for the model to load (can take a few minutes)..."
for i in $(seq 1 100); do
  healthy && break
  if ! pgrep -f llama-server >/dev/null && ! watchdog_up; then
    echo "❌ Server process and watchdog both died — last server.log lines:"
    tail -15 server.log 2>/dev/null
    exit 1
  fi
  sleep 3
done
healthy || { echo "❌ Server still not healthy after ~5 minutes — see server.log"; tail -15 server.log 2>/dev/null; exit 1; }

# ── 6. make sure the bot itself is running ──────────────────────────────────
if ! bot_up; then
  PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
  setsid nohup "$PY" bot.py >> bot.log 2>&1 &
  sleep 3
fi
bot_up || { echo "❌ Bot did not start — see bot.log"; tail -15 bot.log 2>/dev/null; exit 1; }

echo "✅ Stack is UP: model server healthy, bot running, watchdog supervising."
echo "   You can close this terminal — everything keeps running."
echo "   Logs: server.log · bot.log · watchdog.log"
