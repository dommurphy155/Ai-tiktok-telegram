import os
import re
import json
import sqlite3
import asyncio
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from config import (
    MAX_VIDEOS_PER_REQUEST,
    log,
    OPENAI_API_KEY,
)

# --- OpenAI setup ---
OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
if OPENAI_AVAILABLE:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        log("[VPS LOG] OpenAI API key is available and will be used.")
    except Exception:
        OPENAI_AVAILABLE = False
        log("[VPS LOG] OpenAI API key failed to initialize, AI features disabled.")
else:
    log("[VPS LOG] No OpenAI API key found, AI features disabled.")

# ---------------- DB helpers ----------------
def mark_urls_sent(conn: sqlite3.Connection, urls, video_ids=None):
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    for url in urls:
        vid = (video_ids and video_ids.get(url)) or (url.rstrip("/").split("/")[-1])
        try:
            cur.execute(
                "INSERT OR IGNORE INTO sent_videos (video_id, url, sent_at) VALUES (?, ?, ?)",
                (vid, url, ts)
            )
        except Exception:
            pass
    conn.commit()

def save_ai_memory(conn: sqlite3.Connection, user_id, query_text, result_urls):
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO ai_memory (user_id, query_text, result_urls, ts) VALUES (?, ?, ?, ?)",
        (str(user_id), query_text, json.dumps(result_urls), ts)
    )
    conn.commit()

async def load_ai_memory(conn: sqlite3.Connection, user_id, limit=50):
    """Return last queries/results for context"""
    cur = conn.cursor()
    cur.execute(
        "SELECT query_text, result_urls, ts FROM ai_memory WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (str(user_id), limit)
    )
    rows = cur.fetchall()
    mem = []
    for q, urls, ts in rows:
        try:
            urls_list = json.loads(urls)
        except Exception:
            urls_list = []
        mem.append({"query": q, "urls": urls_list, "ts": ts})
    return mem

# ---------------- Parse user input ----------------
async def parse_user_request(text: str, memory=None, last_sent_urls=None):
    """
    Uses AI to understand request, correct typos, extract query and count,
    generate confirmation prompt, and detect follow-ups like "more like last ones".
    Returns (query, count, suggested_prompt)
    """
    text = (text or "").strip()
    if not text:
        return None, 0, None

    # --- Detect follow-up requests ---
    if last_sent_urls and re.search(r"more (like|similar) (the )?(last|previous)", text, re.I):
        return "__FOLLOWUP__", len(last_sent_urls), f"Do you want me to send {len(last_sent_urls)} more videos similar to the last ones?"

    suggested_prompt = None

    if OPENAI_AVAILABLE:
        mem_text = ""
        if memory:
            mem_text = "\nUser history:\n" + "\n".join([f"{m['ts']}: {m['query']}" for m in memory])
        prompt = f"""
You are a smart assistant helping fetch TikTok videos.
User instruction: '''{text}'''
{mem_text}

Correct typos, understand intent, and extract:
1. A short search query (what to search for)
2. Number of videos (1-{MAX_VIDEOS_PER_REQUEST})

Also suggest a human-readable confirmation prompt, e.g. "I will search 5 videos for 'funny cat edits', is that okay?"

Return JSON:
{{
"query": "...",
"count": N,
"confirmation": "..."
}}
"""
        try:
            resp = await asyncio.to_thread(lambda: openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=150,
                temperature=0.0,
            ))
            raw = resp.choices[0].text.strip()
            j = None
            try:
                j = json.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", raw, re.S)
                if m:
                    j = json.loads(m.group(0))
            if j:
                query = j.get("query", "").strip()
                count = int(j.get("count") or 0)
                count = min(max(1, count), MAX_VIDEOS_PER_REQUEST)
                suggested_prompt = j.get("confirmation")
                return query, count, suggested_prompt
        except Exception as e:
            log(f"[AI PARSE ERROR] {e}")

    # fallback parser
    m = re.search(r"(\d+)\s*(?:videos|video|v)?", text, re.I)
    count = int(m.group(1)) if m else 3
    count = min(count, MAX_VIDEOS_PER_REQUEST)

    COMMON_WORDS = r"\b(send|me|need|please|find|the|that|i'm|i am|like|videos|video|of|for|now|what|you|to|do|is|about|okay|bloody|perfect|most|recent)\b"
    cleaned = re.sub(COMMON_WORDS, " ", text, flags=re.I)
    cleaned = re.sub(r"\d+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    words = [w for w in cleaned.split() if len(w) > 1]
    if m:
        num_index = text.lower().split().index(m.group(1))
        context_words = text.split()[num_index+1:num_index+6]
        words += [w for w in context_words if len(w) > 1]
    query = " ".join(words)[:120].strip()
    if not query:
        query = "fyp"
    suggested_prompt = f"I will search {count} videos for '{query}', is that okay?"
    return query, count, suggested_prompt

# ---------------- Telegram helpers ----------------
def make_markup():
    return InlineKeyboardMarkup([[InlineKeyboardButton("Next ▶️", callback_data="next")]])

# ---------------- AI Fresh URL filter ----------------
async def ai_filter_fresh_urls(conn, candidate_urls, desired_count):
    cur = conn.cursor()
    fresh = []
    for url in candidate_urls:
        vid = url.rstrip("/").split("/")[-1]
        cur.execute("SELECT 1 FROM sent_videos WHERE video_id = ?", (vid,))
        if not cur.fetchone():
            fresh.append(url)
        if len(fresh) >= desired_count:
            break

    if OPENAI_AVAILABLE and fresh:
        try:
            prompt = f"""
Given these TikTok URLs: {fresh}
Filter out duplicates, near-duplicates, or URLs that seem already known.
Return at most {desired_count} URLs in JSON array format.
"""
            resp = await asyncio.to_thread(lambda: openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=300,
                temperature=0.0,
            ))
            raw = resp.choices[0].text.strip()
            try:
                j = json.loads(raw)
                if isinstance(j, list):
                    fresh = j[:desired_count]
            except Exception:
                pass
        except Exception as e:
            log(f"[AI FILTER ERROR] {e}")
    return fresh[:desired_count]

# ---------------- High-level handler ----------------
async def handle_user_request(update: Update, context: ContextTypes.DEFAULT_TYPE, tiktok_collect_fn, downloader_fn, db_conn):
    user_text = update.message.text or ""
    user = update.effective_user
    await update.message.chat.send_action("typing")
    log(f"🤖 user asked: {user_text}")

    # --- Load conversation memory ---
    memory = await load_ai_memory(db_conn, user.id)
    last_sent_urls = memory[0]["urls"] if memory else None

    query, count, confirmation_prompt = await parse_user_request(user_text, memory, last_sent_urls)

    if not query or count <= 0:
        await update.message.reply_text("I couldn't understand your request. Try something like: `send 5 funny edits`")
        return

    # --- Handle follow-up requests ---
    if query == "__FOLLOWUP__" and last_sent_urls:
        candidate_urls = last_sent_urls
        await update.message.reply_text(confirmation_prompt)
    else:
        # --- Ask for confirmation ---
        if confirmation_prompt:
            await update.message.reply_text(confirmation_prompt)
            def check_reply(msg_update):
                return msg_update.effective_chat.id == update.effective_chat.id and msg_update.from_user.id == user.id
            try:
                reply_update = await context.bot.wait_for_message(timeout=30, check=check_reply)
                if not reply_update or not reply_update.text.lower() in ("yes", "y", "ok", "sure"):
                    await update.message.reply_text("❌ Search cancelled.")
                    return
            except asyncio.TimeoutError:
                await update.message.reply_text("⌛ Timeout, search cancelled.")
                return

        driver = context.bot_data.get("tiktok_driver")
        if not driver:
            await update.message.reply_text("⚠️ Scraper not available right now. Try again later.")
            return

        loop = asyncio.get_running_loop()
        try:
            candidate_urls = await loop.run_in_executor(
                None,
                lambda: tiktok_collect_fn(driver, [query], per_query=count, batch_limit=count)
            )
        except Exception as e:
            log(f"[ERROR] collecting URLs: {e}")
            await update.message.reply_text("❌ Failed to collect video links.")
            return

    if not candidate_urls:
        await update.message.reply_text("⚠️ No videos found for that query.")
        return

    # --- AI-powered fresh URL filter ---
    fresh = await ai_filter_fresh_urls(db_conn, candidate_urls, count)
    if not fresh:
        await update.message.reply_text("⚠️ No new videos (already sent these).")
        return

    await update.message.reply_text(f"⬇️ Downloading {len(fresh)} videos now...")

    try:
        downloaded_paths = await downloader_fn(fresh)
    except Exception as e:
        log(f"[ERROR] download step: {e}")
        await update.message.reply_text("❌ Download failed.")
        return

    if not downloaded_paths:
        await update.message.reply_text("❌ Downloads failed or returned no files.")
        return

    mark_urls_sent(db_conn, fresh)
    save_ai_memory(db_conn, user.id, user_text, fresh)

    sent = 0
    for path in downloaded_paths:
        try:
            with open(path, "rb") as f:
                await context.bot.send_video(chat_id=update.effective_chat.id, video=f)
            sent += 1
        except Exception as e:
            log(f"[WARNING] Failed sending {path}: {e}")
    await update.message.reply_text(f"✅ Sent {sent} videos.")
