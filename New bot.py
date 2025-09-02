import os
import re
import json
import sqlite3
import asyncio
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler, MessageHandler, filters

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

def save_valuable_words(conn, user_id, words):
    cur = conn.cursor()
    for w in words:
        try:
            cur.execute(
                "INSERT OR IGNORE INTO valuable_words (user_id, word) VALUES (?, ?)",
                (user_id, w)
            )
        except Exception:
            pass
    conn.commit()

async def load_valuable_words(conn, user_id):
    cur = conn.cursor()
    cur.execute(
        "SELECT word FROM valuable_words WHERE user_id=?",
        (user_id,)
    )
    rows = cur.fetchall()
    return [r[0] for r in rows]

# ---------------- GPT query expansion ----------------
async def expand_query_with_gpt(query: str, valuable_words=None, max_prompts=3):
    if not OPENAI_AVAILABLE:
        return [query]

    valuable_text = ""
    if valuable_words:
        valuable_text = "Previously valuable words: " + ", ".join(valuable_words)

    prompt = f"""
You are a helpful assistant generating high-quality search prompts for TikTok videos.
User input: "{query}"
{valuable_text}

Return up to {max_prompts} alternative search prompts that preserve the user's intent,
using synonyms and relevant related terms.
Return as a JSON array of strings.
"""
    try:
        resp = await openai.chat.completions.acreate(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content.strip()
        j = None
        try:
            j = json.loads(raw)
        except Exception:
            m = re.search(r"\[.*\]", raw, re.S)
            if m:
                j = json.loads(m.group(0))
        if j and isinstance(j, list):
            return j[:max_prompts]
    except Exception as e:
        log(f"[AI QUERY EXPANSION ERROR] {e}")
    return [query]

# ---------------- Parse user input ----------------
async def parse_user_request(text: str, memory=None, last_sent_urls=None, db_conn=None):
    text = (text or "").strip()
    if not text:
        return None, 0, None, None

    if last_sent_urls and re.search(r"more (like|similar) (the )?(last|previous)", text, re.I):
        return "__FOLLOWUP__", len(last_sent_urls), f"Do you want me to send {len(last_sent_urls)} more videos similar to the last ones?", None

    suggested_prompt = None
    query = None

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

Return JSON:
{{
"query": "...",
"count": N
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
                count = int(j.get("count") or 3)
                count = min(max(1, count), MAX_VIDEOS_PER_REQUEST)
        except Exception as e:
            log(f"[AI PARSE ERROR] {e}")

    # fallback
    if not query:
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
        # save valuable words
        if db_conn:
            await asyncio.to_thread(save_valuable_words, db_conn, "user_id_placeholder", words)

    # expand query with GPT
    valuable_words = await load_valuable_words(db_conn, "user_id_placeholder") if db_conn else None
    alt_prompts = await expand_query_with_gpt(query, valuable_words)
    suggested_prompt = f"I will search {count} videos for '{alt_prompts[0]}', is that okay?"

    return query, count, suggested_prompt, alt_prompts

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
CONFIRMATION = range(1)

async def handle_user_request(update: Update, context: ContextTypes.DEFAULT_TYPE, tiktok_collect_fn, downloader_fn, db_conn):
    user_text = update.message.text or ""
    user = update.effective_user
    await update.message.chat.send_action("typing")
    log(f"🤖 user asked: {user_text}")

    memory = await load_ai_memory(db_conn, user.id)
    last_sent_urls = memory[0]["urls"] if memory else None

    query, count, confirmation_prompt, alt_prompts = await parse_user_request(user_text, memory, last_sent_urls, db_conn)

    if not query or count <= 0:
        await update.message.reply_text("I couldn't understand your request. Try something like: `send 5 funny edits`")
        return

    # --- Handle follow-up ---
    if query == "__FOLLOWUP__" and last_sent_urls:
        candidate_urls = last_sent_urls
        await update.message.reply_text(confirmation_prompt)
    else:
        # --- Confirmation ---
        if confirmation_prompt:
            buttons = [
                [InlineKeyboardButton("✅ Yes", callback_data="confirm")],
                [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
            ]
            await update.message.reply_text(confirmation_prompt, reply_markup=InlineKeyboardMarkup(buttons))
            # Wait for user to press button
            context.user_data["awaiting_confirmation"] = True
            context.user_data["confirmation_result"] = None
            # Confirmation will be handled by a separate callback handler

            while context.user_data.get("awaiting_confirmation"):
                await asyncio.sleep(0.5)

            if context.user_data.get("confirmation_result") != "confirm":
                await update.message.reply_text("❌ Search cancelled.")
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

# ---------------- Confirmation button handler ----------------
async def confirmation_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if context.user_data.get("awaiting_confirmation"):
        context.user_data["confirmation_result"] = query.data
        context.user_data["awaiting_confirmation"] = False
