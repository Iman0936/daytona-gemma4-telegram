#!/usr/bin/env python3
"""
Telegram bot for Gemma 4 E4B served by llama.cpp (llama-server) on the same sandbox.

Config via environment variables:
  BOT_TOKEN     (required) token from BotFather
  ALLOWED_IDS   optional, comma-separated Telegram user IDs; empty = open to everyone
                (for a personal test bot, set it to your own ID so only you can use it)
  SYSTEM_PROMPT optional system prompt (without the thinking token; see THINKING)
  THINKING      optional, default "1" = thinking mode ON; set "0" to disable
  MAX_TOKENS    optional, default 32768 (thinking tokens count toward this budget;
                the real ceiling is the server context — see watchdog.sh)

Generation settings are Google's official recommendation for Gemma 4:
  temperature=1.0, top_p=0.95, top_k=64

Thinking mode is ON by default: per Google's docs it is activated by putting
the <|think|> token at the start of the system prompt. The model's internal
reasoning is shown in a collapsible quote box above the answer (while
streaming, only the tail of the thought is shown in a plain quote, since
Telegram re-collapses expandable quotes on every edit). Thought content
is kept out of the conversation history (Google's multi-turn rule).

Long memory: the bot keeps the last 100 exchanges per chat (trimmed to a
120000-character budget so history + answer always fit the 64K context) and
persists them to history.json next to this file, so the model's memory
survives bot and server restarts. /reset wipes a chat's memory.
llama-server reuses the cached prompt prefix between turns, so serving a
long history stays cheap — only the newest turn is processed each time.

Replies are STREAMED: llama-server emits tokens chunk by chunk and the bot
keeps editing one Telegram message (throttled to ~1 edit/second — Telegram's
practical edit limit), so the answer types itself out live. When a message
reaches Telegram's 4096-character limit it is sealed and the answer keeps
streaming in a NEW message — long replies are never cut off at the limit.
If the stream breaks mid-answer, whatever was generated is still delivered
to the user instead of being thrown away.
Telegram rate-limits edits and new messages during long streams
("RetryAfter"). Every edit/send here is retried after the wait Telegram
asks for, and the character bookkeeping only advances after a chunk is
confirmed on screen — a rate-limited chunk is never silently skipped,
which would otherwise punch a hole in the middle of the reply.
"""
import asyncio
import json
import os
import re
import time

import html
import httpx
from telegram import Update
from telegram.request import HTTPXRequest
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

API_URL = "http://127.0.0.1:8080/v1/chat/completions"
BOT_TOKEN = os.environ["BOT_TOKEN"]
ALLOWED_IDS = {
    int(x) for x in os.environ.get("ALLOWED_IDS", "").replace(" ", "").split(",") if x
}
THINKING = os.environ.get("THINKING", "1") != "0"
_BASE_SYSTEM = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Always reply in the same language as the user's message.",
)
SYSTEM_PROMPT = ("<|think|>" + _BASE_SYSTEM) if THINKING else _BASE_SYSTEM
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32768"))
HISTORY_TURNS = 100         # the model's memory: how many exchanges it can recall
MAX_HISTORY_CHARS = 120000  # guardrail: history + answer must fit the context
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.json")

# chat_id -> list of {"role": ..., "content": ...}
histories: dict[int, list[dict]] = {}

THOUGHT_BLOCK = re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL)
THINKING_TEXT = "🧠 دارم فکر می‌کنم..."
WORKING_TEXT = "⏳ دارم جواب رو می‌سازم..."


def _load_histories() -> None:
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            for chat_id, msgs in json.load(f).items():
                histories[int(chat_id)] = msgs
    except Exception:
        pass


def _save_histories() -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(histories, f, ensure_ascii=False)
    except Exception:
        pass


_load_histories()


def allowed(update: Update) -> bool:
    return not ALLOWED_IDS or (update.effective_user and update.effective_user.id in ALLOWED_IDS)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    histories.pop(update.effective_chat.id, None)
    _save_histories()
    await update.message.reply_text(
        "سلام! من با مدل Gemma 4 روی سرور Daytona کار می‌کنم. "
        "پیامت رو بفرست تا جواب بدم.\n"
        "با دستور /reset گفتگو از اول شروع می‌شود."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update):
        return
    histories.pop(update.effective_chat.id, None)
    _save_histories()
    await update.message.reply_text("حافظه گفتگو پاک شد. از اول شروع می‌کنیم.")


def _visible_text(content: str) -> str:
    """Final-answer text only: completed thought blocks removed, and
    anything from an unclosed thought tag onward cut off."""
    text = THOUGHT_BLOCK.sub("", content)
    open_idx = text.rfind("<|channel>thought")
    if open_idx != -1:
        text = text[:open_idx]
    return text.strip()


def _thought_text(content: str, reasoning: str) -> str:
    """The model's thinking: the reasoning_content stream when the server
    sends one, otherwise the thought embedded in content channel tags."""
    if reasoning.strip():
        return reasoning.strip()
    parts = []
    for m in THOUGHT_BLOCK.findall(content):
        parts.append(
            m.replace("<|channel>thought", "", 1).replace("<channel|>", "").strip()
        )
    open_idx = content.rfind("<|channel>thought")
    if open_idx != -1 and "<channel|>" not in content[open_idx:]:
        parts.append(content[open_idx + len("<|channel>thought"):].strip())
    return "\n".join(p for p in parts if p).strip()


def _cut_html(s: str, limit: int = 4000):
    """Split s into (chunk, rest) with chunk <= limit, without breaking
    the HTML: never cut inside a tag or an entity, and if a blockquote
    is left open in the chunk, close it there and reopen it in the rest."""
    if len(s) <= limit:
        return s, ""
    cut = limit
    lt = s.rfind("<", 0, cut)
    if lt != -1 and s.find(">", lt) >= cut:
        cut = lt
    amp = s.rfind("&", 0, cut)
    if amp != -1 and ";" not in s[amp:cut]:
        cut = amp
    chunk, rest = s[:cut], s[cut:]
    if chunk.count("<blockquote") > chunk.count("</blockquote>"):
        kind = "<blockquote expandable>" if "<blockquote expandable>" in chunk else "<blockquote>"
        chunk += "</blockquote>"
        rest = kind + rest
    return chunk, rest


def _chunk_html(s: str, limit: int = 4000) -> list:
    chunks = []
    while s:
        chunk, s = _cut_html(s, limit)
        chunks.append(chunk)
    return chunks


def _preview(content: str, reasoning: str, final: bool = False) -> str:
    """What the user sees (HTML): the model's thinking in a quote box on
    top, the answer below it - the display form of the Qwen bot. While
    streaming, Telegram re-collapses expandable quotes on every edit, so
    the live view is a plain quote holding only the thought's tail; the
    full expandable quote lands in the final message. The thinking
    itself is never altered, shortened or limited - only displayed."""
    thought = _thought_text(content, reasoning)
    text = _visible_text(content)
    parts = []
    if thought:
        if final:
            parts.append(f"<blockquote expandable>🧠 {html.escape(thought)}</blockquote>")
        else:
            tail = thought[-350:]
            if len(thought) > 350:
                tail = "…" + tail
            parts.append(f"<blockquote>🧠 {html.escape(tail)}</blockquote>")
    if text:
        parts.append(html.escape(text))
    if parts:
        return "\n".join(parts)
    if reasoning or content:
        return THINKING_TEXT
    return ""

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not allowed(update) or not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    history = histories.setdefault(chat_id, [])
    history.append({"role": "user", "content": update.message.text})
    # Trim memory: keep the newest HISTORY_TURNS exchanges and stay under the
    # char budget, always dropping whole user+assistant pairs from the front.
    while len(history) > HISTORY_TURNS * 2 or (
        sum(len(m["content"]) for m in history) > MAX_HISTORY_CHARS and len(history) > 2
    ):
        del history[:2]

    # Streaming: send a placeholder message, then keep EDITING it as tokens
    # arrive. Thoughts stay hidden behind the thinking indicator. When the
    # visible answer outgrows Telegram's 4096-char message limit, the full
    # message is sealed and streaming continues in a NEW message — the reply
    # never stops at the limit.
    current = await update.message.reply_text(THINKING_TEXT if THINKING else WORKING_TEXT)
    base = 0  # answer chars CONFIRMED on screen in earlier (full) messages

    async def _edit(msg, text: str) -> bool:
        """Edit a message, riding out Telegram flood limits. True = shown."""
        for _ in range(5):
            try:
                await msg.edit_text(text, parse_mode="HTML")
                return True
            except RetryAfter as e:
                await asyncio.sleep(min(e.retry_after + 1, 30))
            except BadRequest as e:
                return "not modified" in str(e).lower()  # already on screen
            except Exception:
                return False
        return False

    async def _send(text: str):
        """Send a new message, riding out Telegram flood limits."""
        for _ in range(5):
            try:
                return await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
            except RetryAfter as e:
                await asyncio.sleep(min(e.retry_after + 1, 30))
            except Exception:
                return None
        return None

    async def show(full_visible: str) -> None:
        # Seal full 4000-char chunks into new messages. base advances ONLY
        # after the seal edit AND the follow-up message both succeeded —
        # otherwise the chunk is retried on the next tick, never skipped.
        nonlocal current, base
        while len(full_visible) - base > 4000:
            piece, _ = _cut_html(full_visible[base:], 4000)
            if not await _edit(current, piece):
                return
            new_msg = await _send("…")
            if new_msg is None:
                return
            base += len(piece)
            current = new_msg
        tail = full_visible[base:]
        if tail:
            await _edit(current, tail)

    content = ""
    reasoning = ""
    last_edit = 0.0
    last_shown_len = 0
    try:
        payload = {
            "model": "gemma-4-e4b",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": MAX_TOKENS,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=3600) as client:  # a full-length answer can take ~30 min on CPU
            async with client.stream("POST", API_URL, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"]
                    except (ValueError, KeyError, IndexError):
                        continue
                    # llama-server may stream thoughts as reasoning_content,
                    # or inline in content inside the thought channel tags.
                    if delta.get("reasoning_content"):
                        reasoning += delta["reasoning_content"]
                    if delta.get("content"):
                        content += delta["content"]
                    if not (delta.get("content") or delta.get("reasoning_content")):
                        continue
                    # Telegram rate-limits edits — refresh at most every
                    # ~1.5 s, and only once ~20 new chars have arrived
                    # (or immediately when a message needs sealing).
                    now = time.monotonic()
                    preview = _preview(content, reasoning)
                    due = now - last_edit >= 1.5 and len(preview) - last_shown_len >= 20
                    if preview and (due or len(preview) - base > 4000):
                        last_edit = now
                        await show(preview)
                        last_shown_len = len(preview)
    except Exception:
        pass  # connection/stream broke — keep whatever partial answer arrived

    clean = _visible_text(content)
    if base == 0:
        # Nothing was sealed while streaming: the whole reply still lives
        # in the current message, so it can be replaced wholesale by the
        # final form (full thought in an expandable quote).
        chunks = _chunk_html(_preview(content, reasoning, final=True))
    else:
        # Part of the reply is already sealed in earlier messages and
        # cannot be rewritten. Finish in the streaming form instead:
        # replacing from chunk 0 here would resend the text from the
        # beginning and duplicate the answer (bug seen 2026-10-30).
        chunks = _chunk_html(_preview(content, reasoning)[base:])
    if not clean:
        history.pop()  # drop the failed user turn
        if content or reasoning:
            msg = "فکر کردن این سؤال کل بودجه توکن رو خورد و به جواب نرسید. سؤالت رو کوتاه‌تر بپرس."
        else:
            msg = "الان سرور مدل جواب نداد. چند ثانیه صبر کن و دوباره بفرست."
        await _edit(current, msg)
        return

    history.append({"role": "assistant", "content": clean})
    _save_histories()
    # Deliver whatever is left, in order: fill the current message first
    # (falling back to a fresh message if the edit will not land), then
    # continue in new messages until the whole answer is out.
    if chunks:
        if not await _edit(current, chunks[0]):
            await _send(chunks[0])
        for ch in chunks[1:]:
            await _send(ch)



# --- Liveness: heartbeat + Telegram-link monitor -------------------------
# Why: on 2026-10-30 this bot was found alive but silently dead for days -
# Telegram had closed its pooled connections (sockets stuck in CLOSE-WAIT)
# and the long-poll never recovered. Three layers against that death:
# 1) explicit HTTPX timeouts in main(): a dead connection now ERRORS
#    within seconds instead of hanging forever, and polling retries on a
#    fresh connection;
# 2) this monitor: probes Telegram through the bot's own connection every
#    60 s, refreshes heartbeat.txt, and exits after 3 failed probes so the
#    watchdog starts a fresh process;
# 3) watchdog.sh supervises heartbeat.txt: a wedged event loop stops
#    refreshing it, and the watchdog SIGKILLs and restarts the bot.
HEARTBEAT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heartbeat.txt")
_link_fails = 0


async def _liveness(app: Application) -> None:
    global _link_fails
    while True:
        try:
            await asyncio.wait_for(app.bot.get_me(), timeout=15)
            _link_fails = 0
        except Exception:
            _link_fails += 1
            if _link_fails >= 3:
                os._exit(1)  # link is dead; the watchdog starts a fresh process
        try:
            with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
                f.write(str(time.time()))
        except Exception:
            pass
        await asyncio.sleep(60)


async def _post_init(app: Application) -> None:
    asyncio.create_task(_liveness(app))


def main() -> None:
    request = HTTPXRequest(
        connection_pool_size=8, connect_timeout=10.0,
        read_timeout=20.0, write_timeout=20.0, pool_timeout=5.0,
    )
    get_updates_request = HTTPXRequest(
        connection_pool_size=2, connect_timeout=10.0,
        read_timeout=45.0, write_timeout=20.0, pool_timeout=5.0,
    )
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .request(request)
        .get_updates_request(get_updates_request)
        .post_init(_post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling()


if __name__ == "__main__":
    main()
