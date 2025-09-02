

# ========================
# START OF MAIN.PY
# ========================

import os
import asyncio
import sys
from functools import partial

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import (
    TELEGRA M_BOT_TOKEN as _unused,  # placeholder to avoid linter noise if imported
    TELEGRAM_BOT_TOKEN,
    NETSCAPE_COOKIES_FILE,
    TIKTOK_COOKIES_JSON,
    log,
    init_db,
    DB_CONN,
    MAX_VIDEOS_PER_REQUEST,
)
from tiktok import (
    setup_browser,
    load_cookies_from_file,
    apply_cookies,
    convert_json_to_netscape,
    collect_batch_urls,
)
import downloader  # we'll use downloader.download_batch wrapper
import bot as bot_module

# --- Start DB and browser ---
def start_state():
    # init sqlite
    conn = init_db()
    return conn

# simple downloader wrapper (blocking): takes list of urls -> returns list of filepath(s)
def blocking_downloader(urls):
    """
    For simplicity: synchronous loop calling yt-dlp per url.
    Returns list of saved file paths (skip failures).
    """
    out = []
    for u in urls:
        try:
            # delegate to downloader module: it has async functions; call them with a short-run loop
            # For simplicity we call downloader.download_video via asyncio run
            r = asyncio.run(downloader.download_video(u, os.path.join(os.getcwd(), "downloads")))
            if r:
                path, _meta = r
                out.append(path)
        except Exception as e:
            log(f"[WARNING] downloader failure for {u}: {e}")
    return out

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 TikTok AI helper started. Ask me e.g. 'send me 5 creed edits' (max 10)."
    )

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # delegate to bot handler; pass driver + blocking_downloader + db_conn
    await bot_module.handle_user_request(
        update,
        context,
        tiktok_collect_fn=collect_batch_urls,
        downloader_fn=blocking_downloader,
        db_conn=context.application.bot_data["db_conn"],
    )

def build_app(token, driver, db_conn):
    app = Application.builder().token(token).build()
    # store driver + db_conn for access in handlers
    app.bot_data["tiktok_driver"] = driver
    app.bot_data["db_conn"] = db_conn

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_message))
    return app

def main():
    if not TELEG RAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN env required")
        sys.exit(2)

    log("Starting TikTok downloader with Telegram bot...")
    # sqlite
    db_conn = start_state()

    # browser
    driver = setup_browser()
    cookies = load_cookies_from_file(TIKTOK_COOKIES_JSON)
    log("✅ Loading cookies from file...")
    log(f"✅ Loaded {len(cookies)} cookies")
    apply_cookies(driver, cookies)
    convert_json_to_netscape(TIKTOK_COOKIES_JSON, NETSCAPE_COOKIES_FILE)

    # wire bot
    app = build_app(TELEGRAM_BOT_TOKEN, driver, db_conn)
    log("🤖 telegram ai started")

    # run polling (blocking)
    app.run_polling()

if __name__ == "__main__":
    main()


# ========================
# END OF MAIN.PY
# ========================



# ========================
# START OF CONFIG.PY
# ========================

# config.py
from datetime import datetime
import os
import sqlite3

# ------------------- Env / Tunables -------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # optional

TIKTOK_COOKIES_JSON = os.getenv("TIKTOK_COOKIES_FILE", "tiktok_cookies.json")
NETSCAPE_COOKIES_FILE = os.getenv("NETSCAPE_COOKIES_FILE", "tiktok_cookies.txt")

OUTPUT_PATH = os.path.join(os.getcwd(), "downloads")
os.makedirs(OUTPUT_PATH, exist_ok=True)

PRELOAD_TARGET = int(os.getenv("PRELOAD_TARGET", "10"))
ROTATION_BATCH_SIZE = max(5, PRELOAD_TARGET)
VIDEO_CACHE_MAXLEN = int(os.getenv("VIDEO_CACHE_MAXLEN", "128"))
MAX_VIDEOS_PER_REQUEST = int(os.getenv("MAX_VIDEOS_PER_REQUEST", "10"))

SEARCH_QUERIES_FALLBACK = ["4k", "edit", "fyp", "funny", "movie"]

# ------------------- Logging -------------------
def log(msg: str):
    prefix = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{prefix} {msg}")

# ------------------- SQLite helpers -------------------
SQLITE_FILE = os.path.join(os.getcwd(), "bot_state.db")

def init_db():
    conn = sqlite3.connect(SQLITE_FILE, timeout=30)
    cur = conn.cursor()
    # table for sent videos to avoid duplicates
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sent_videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT UNIQUE,
        url TEXT,
        sent_at TEXT
    )
    """)
    # table for AI memory (queries / user context)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ai_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        query_text TEXT,
        result_urls TEXT,
        ts TEXT
    )
    """)
    conn.commit()
    return conn

# db connection will be created by main
DB_CONN = None


# ========================
# END OF CONFIG.PY
# ========================



# ========================
# START OF TIKTOK.PY
# ========================

# tiktok.py
"""
Selenium + helper functions for navigating TikTok, extracting video URLs,
and converting cookies for yt-dlp. Keep this file focused on browser actions.
"""

import json
import random
import re
import time
import os
import urllib.parse

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

from config import (
    TIKTOK_COOKIES_JSON,
    NETSCAPE_COOKIES_FILE,
    ROTATION_BATCH_SIZE,
    SEARCH_QUERIES_FALLBACK,
    log,
)

# ---------------- Browser setup ----------------
def setup_browser(geckodriver_path="/usr/bin/geckodriver", headless=True):
    opts = Options()
    if headless:
        opts.headless = True
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.set_preference("permissions.default.image", 2)
    opts.set_preference("media.autoplay.enabled", False)
    opts.set_preference("browser.cache.disk.enable", False)
    opts.set_preference("browser.cache.memory.enable", False)

    try:
        driver = webdriver.Firefox(service=Service(geckodriver_path), options=opts)
        log("✅ Firefox WebDriver started")
        return driver
    except WebDriverException as e:
        log(f"[FATAL] Failed to start webdriver: {e}")
        raise

def load_cookies_from_file(path=None):
    path = path or TIKTOK_COOKIES_JSON
    with open(path, "r") as f:
        cookies = json.load(f)
    return cookies

def apply_cookies(driver, cookies, url="https://www.tiktok.com/"):
    try:
        driver.get(url)
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except Exception:
        # proceed anyway
        pass

    # try JS bulk injection
    js = """
    const cookies = arguments[0];
    cookies.forEach(c => {
        try {
            document.cookie = `${c.name}=${c.value};domain=${c.domain||'.tiktok.com'};path=${c.path||'/'}${c.secure ? ';secure' : ''}${c.httpOnly ? ';httponly' : ''}`;
        } catch(e) {}
    });
    """
    try:
        driver.execute_script(js, cookies)
    except Exception:
        # fallback to add_cookie
        for c in cookies:
            try:
                driver.add_cookie({
                    "name": c["name"],
                    "value": c["value"],
                    "domain": c.get("domain", ".tiktok.com"),
                    "path": c.get("path", "/"),
                    "secure": c.get("secure", True),
                    "httpOnly": c.get("httpOnly", False),
                })
            except Exception:
                continue
    time.sleep(1)
    try:
        driver.refresh()
    except Exception:
        pass
    log("✅ Cookies applied")
    log("🔄 Refreshed browser to confirm cookies")

def convert_json_to_netscape(json_file, txt_file):
    with open(json_file, "r") as f:
        cookies = json.load(f)
    with open(txt_file, "w") as f:
        f.write("# Netscape HTTP Cookie File\n")
        for c in cookies:
            domain = c.get("domain", ".tiktok.com")
            include_subdomains = "TRUE"
            path = c.get("path", "/")
            secure = "TRUE" if c.get("secure", False) else "FALSE"
            expiry = str(int(c.get("expiry", 2147483647)))
            name = c["name"]
            value = c["value"]
            f.write("\t".join([domain, include_subdomains, path, secure, expiry, name, value]) + "\n")
    log(f"✅ Converted {os.path.basename(json_file)} into Netscape format for yt-dlp")

# ---------------- Extract video links ----------------
def _normalize_href(href):
    if not href:
        return None
    if not href.startswith("http"):
        href = "https://www.tiktok.com" + href
    href = href.split("?")[0]
    if re.match(r"https://www\.tiktok\.com/.+/video/\d+$", href):
        return href
    return None

def get_fresh_video_links_for_query(driver, query, desired_count=10, scroll_cycles=6, retries=2):
    """
    Navigate to a search URL for `query`, scroll, collect candidate video links.
    Returns up to desired_count unique links.
    """
    query_str = str(query or "")
    encoded = urllib.parse.quote(query_str.replace(",", " "))
    search_url = f"https://www.tiktok.com/search?q={encoded}"
    driver.get(search_url)
    time.sleep(2 + random.uniform(0.5, 1.5))
    log(f"🔄 Rotating search page: {query_str}")

    collected = set()
    attempt = 0
    while len(collected) < desired_count and attempt < retries:
        attempt += 1
        # scroll a few times
        for _ in range(scroll_cycles):
            driver.execute_script(f"window.scrollBy(0, {random.randint(800,1600)})")
            time.sleep(random.uniform(0.8, 1.6))
        try:
            wait = WebDriverWait(driver, 6)
            wait.until(EC.presence_of_element_located((By.XPATH, "//a[contains(@href,'/video/')]")))
        except TimeoutException:
            # try refresh and another attempt
            try:
                driver.refresh()
                time.sleep(1.2)
            except Exception:
                pass

        # try multiple XPaths to be robust
        xpaths = [
            "//a[contains(@href,'/video/')]",
            "//div[contains(@data-e2e,'search_video-item')]//a[contains(@href,'/video/')]",
            "//div[contains(@data-e2e,'recommend-list-item-container')]//a[contains(@href,'/video/')]",
            "//div[contains(@data-e2e,'user-post-item')]//a[contains(@href,'/video/')]",
            "//div[contains(@data-testid,'video')]/a[contains(@href,'/video/')]"
        ]
        for xp in xpaths:
            els = driver.find_elements(By.XPATH, xp)
            for e in els:
                href = _normalize_href(e.get_attribute("href"))
                if href:
                    collected.add(href)
                if len(collected) >= desired_count:
                    break
            if len(collected) >= desired_count:
                break

    return list(collected)[:desired_count]

def rotator_pick_queries():
    # use env config SEARCH_QUERIES if provided in main; fallback otherwise
    from config import SEARCH_QUERIES_FALLBACK
    return SEARCH_QUERIES_FALLBACK[:]

# Simple wrapper used by main: rotate queries and collect links up to batch size
def collect_batch_urls(driver, query_list, per_query=10, batch_limit=50):
    urls = []
    for q in query_list:
        found = get_fresh_video_links_for_query(driver, q, desired_count=per_query)
        for u in found:
            if u not in urls:
                urls.append(u)
            if len(urls) >= batch_limit:
                return urls
    return urls


# ========================
# END OF TIKTOK.PY
# ========================



# ========================
# START OF BOT.PY
# ========================

# bot.py
"""
Telegram handlers + AI parsing wrapper.
- Uses OpenAI (if OPENAI_API_KEY present) to parse user prompts.
- Fallback: simple regex parser.
"""

import os
import re
import json
import sqlite3
import asyncio
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from config import (
    TELEGRA M_BOT_TOKEN as _unused,  # placeholder to avoid linter noise if imported
    OPENAI_API_KEY,
    MAX_VIDEOS_PER_REQUEST,
    log,
    DB_CONN,
)
# above line keeps config imported; actual token used in main

# optional OpenAI client
OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
if OPENAI_AVAILABLE:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
    except Exception:
        OPENAI_AVAILABLE = False

# DB helper
def mark_urls_sent(conn: sqlite3.Connection, urls, video_ids=None):
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    for url in urls:
        vid = (video_ids and video_ids.get(url)) or (url.rstrip("/").split("/")[-1])
        try:
            cur.execute("INSERT OR IGNORE INTO sent_videos (video_id, url, sent_at) VALUES (?, ?, ?)",
                        (vid, url, ts))
        except Exception:
            pass
    conn.commit()

def save_ai_memory(conn: sqlite3.Connection, user_id, query_text, result_urls):
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO ai_memory (user_id, query_text, result_urls, ts) VALUES (?, ?, ?, ?)",
                (str(user_id), query_text, json.dumps(result_urls), ts))
    conn.commit()

# parse user natural language into (query_text, num_videos)
async def parse_user_request(text: str):
    text = (text or "").strip()
    if not text:
        return None, 0

    # if AI is available, ask it to extract: {query, count}
    if OPENAI_AVAILABLE:
        prompt = (
            "Extract a short search query and a number of videos from this user instruction.\n"
            "Return JSON like: {\"query\":\"...\",\"count\":N}\n\n"
            f"Instruction: '''{text}'''"
        )
        try:
            resp = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=60,
                temperature=0.0,
            )
            raw = resp.choices[0].text.strip()
            # attempt to find JSON in reply
            j = None
            try:
                j = json.loads(raw)
            except Exception:
                # try to extract substring that looks like json
                m = re.search(r"\{.*\}", raw, re.S)
                if m:
                    j = json.loads(m.group(0))
            if j:
                q = j.get("query", "").strip()
                c = int(j.get("count") or 0)
                return q, min(max(0, c), MAX_VIDEOS_PER_REQUEST)
        except Exception:
            pass

    # Fallback rule-based parsing:
    # Try to find a number in the message
    m = re.search(r"(\d+)\s*(?:videos|video|v)?", text, re.I)
    count = int(m.group(1)) if m else 3
    if count > MAX_VIDEOS_PER_REQUEST:
        count = MAX_VIDEOS_PER_REQUEST

    # Heuristic: remove common words and return the rest as query
    # e.g. "send me 5 creed edits" -> "creed edits"
    q = re.sub(r"\b(send|me|need|please|find|the|that|i'm|i am|like|videos|video|of|for)\b", " ", text, flags=re.I)
    q = re.sub(r"\d+", "", q)
    q = re.sub(r"[^\w\s]", " ", q)
    q = " ".join([w for w in q.split() if len(w) > 1])[:120].strip()
    if not q:
        q = "fyp"
    return q, count

# Reply keyboard helper
def make_markup():
    return InlineKeyboardMarkup([[InlineKeyboardButton("Next ▶️", callback_data="next")]])

# high-level handler invoked by main
async def handle_user_request(update: Update, context: ContextTypes.DEFAULT_TYPE, tiktok_collect_fn, downloader_fn, db_conn):
    """
    tiktok_collect_fn(driver, query, count) -> list of urls
    downloader_fn(urls, outdir) -> list of downloaded paths
    """
    user_text = update.message.text or ""
    user = update.effective_user
    await update.message.chat.send_action("typing")
    log(f"🤖 user asked: {user_text}")

    query, count = await parse_user_request(user_text)
    if not query or count <= 0:
        await update.message.reply_text("I couldn't understand what you want. Try: `send 5 creed edits`")
        return

    # enforce limit
    if count > MAX_VIDEOS_PER_REQUEST:
        await update.message.reply_text(f"⚠️ Max videos per request is {MAX_VIDEOS_PER_REQUEST}")
        return

    await update.message.reply_text(f"🔎 Searching TikTok for: \"{query}\" (will try to return {count} videos)")

    # call into tiktok (driver is handled in main)
    driver = context.bot_data.get("tiktok_driver")
    if not driver:
        await update.message.reply_text("⚠️ Scraper not available right now. Try again later.")
        return

    # collect URLs (do not block bot UI; run in executor)
    loop = asyncio.get_running_loop()
    try:
        urls = await loop.run_in_executor(None, lambda: tiktok_collect_fn(driver, [query], per_query=count, batch_limit=count))
    except Exception as e:
        log(f"[ERROR] collecting URLs: {e}")
        await update.message.reply_text("❌ Failed to collect video links.")
        return

    if not urls:
        await update.message.reply_text("⚠️ No videos found for that query.")
        return

    # deduplicate against DB
    cur = db_conn.cursor()
    fresh = []
    for u in urls:
        vid = u.rstrip("/").split("/")[-1]
        cur.execute("SELECT 1 FROM sent_videos WHERE video_id = ?", (vid,))
        if cur.fetchone():
            continue
        fresh.append(u)
        if len(fresh) >= count:
            break

    if not fresh:
        await update.message.reply_text("⚠️ No new videos (looks like I've already sent these).")
        return

    await update.message.reply_text(f"⬇️ Downloading {len(fresh)} videos now...")

    # Call downloader (blocking in executor)
    try:
        downloaded_paths = await loop.run_in_executor(None, lambda: downloader_fn(fresh))
    except Exception as e:
        log(f"[ERROR] download step: {e}")
        await update.message.reply_text("❌ Download failed.")
        return

    if not downloaded_paths:
        await update.message.reply_text("❌ Downloads failed or returned no files.")
        return

    # Mark URLs in DB
    mark_urls_sent(db_conn, fresh)
    save_ai_memory(db_conn, user.id, user_text, fresh)

    # Send videos serially (async)
    sent = 0
    for path in downloaded_paths:
        try:
            with open(path, "rb") as f:
                await context.bot.send_video(chat_id=update.effective_chat.id, video=f)
            sent += 1
        except Exception as e:
            log(f"[WARNING] Failed sending {path}: {e}")
    await update.message.reply_text(f"✅ Sent {sent} videos.")


# ========================
# END OF BOT.PY
# ========================

