

# ========================
# START OF MAIN.PY
# ========================

import os
import asyncio
import sys
from dotenv import load_dotenv
from functools import partial
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

os.environ["DISPLAY"] = ":99"

load_dotenv()

from config import (
    TELEGRAM_BOT_TOKEN,
    NETSCAPE_COOKIES_FILE,
    TIKTOK_COOKIES_JSON,
    log,
    init_db,
    DB_CONN,
    MAX_VIDEOS_PER_REQUEST,
    MAX_CONCURRENT_DOWNLOADS
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
download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

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
          "🎉 Your TikTok content assistant is ready!\n"
          "Send a message like ‘3 funny pranks’ and I’ll fetch fresh videos for you.\n\n"
          "⏳ Each batch takes a little time to gather and download, so please be patient.\n"
         "⚠️ Only one request at a time, with a maximum of 10 videos per message."
    )

# --- Async downloader wrapper ---
async def async_downloader(urls):
    """Download multiple videos concurrently, limited by MAX_CONCURRENT_DOWNLOADS."""
    out = []

    async def sem_download(url):
        async with download_semaphore:
            try:
                r = await downloader.download_video(url, os.path.join(os.getcwd(), "downloads"))
                if r:
                    path, _meta = r
                    out.append(path)
            except Exception as e:
                log(f"[WARNING] downloader failure for {url}: {e}")

    # create tasks
    tasks = [sem_download(u) for u in urls]
    await asyncio.gather(*tasks)
    return out

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # delegate to bot handler; pass driver + blocking_downloader + db_conn
    await bot_module.handle_user_request(
        update,
        context,
        tiktok_collect_fn=collect_batch_urls,
        downloader_fn=async_downloader,
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
    if not TELEGRAM_BOT_TOKEN:
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

from datetime import datetime
import os
import sqlite3
from dotenv import load_dotenv
load_dotenv()
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

MAX_CONCURRENT_DOWNLOADS = 10

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
def setup_browser():
    """Start Firefox WebDriver with optimized options."""
    firefox_options = Options()
    firefox_options.headless = True
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument("--disable-dev-shm-usage")
    firefox_options.add_argument("--width=800")
    firefox_options.add_argument("--height=600")

    firefox_options.set_preference("dom.ipc.processCount", 1)              # single process
    firefox_options.set_preference("browser.tabs.remote.autostart", False) # disable multiprocess
    firefox_options.set_preference("gfx.webrender.all", False)             # no GPU compositor
    firefox_options.set_preference("layers.acceleration.disabled", True)   # disable hardware accel
    firefox_options.set_preference("permissions.default.image", 2)         # block all images
    firefox_options.set_preference("media.autoplay.enabled", False)        # no video autoplay
    firefox_options.set_preference("media.hardware-video-decoding.enabled", False)
    firefox_options.set_preference("browser.cache.disk.enable", False)     # no disk cache
    firefox_options.set_preference("browser.cache.memory.enable", False)   # no memory cache
    firefox_options.set_preference("network.prefetch-next", False)         # disable prefetch
    firefox_options.set_preference("extensions.update.enabled", False)     # no addon updates
    firefox_options.set_preference("app.update.enabled", False)            # no app updates
    firefox_options.set_preference("toolkit.telemetry.enabled", False)     # no telemetry
    firefox_options.set_preference("datareporting.healthreport.uploadEnabled", False)

    try:
        driver = webdriver.Firefox(service=Service("/usr/bin/geckodriver"), options=firefox_options)
        return driver
    except WebDriverException as e:
        log(f"[ERROR] Failed to start Firefox WebDriver: {e}")
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

def get_fresh_video_links_for_query(driver, query, desired_count=10, scroll_cycles=0, retries=2):
    """
    Navigate to a search URL for `query`, scroll, collect candidate video links.
    Returns up to desired_count unique links.
    """
    query_str = str(query or "")
    encoded = urllib.parse.quote(query_str.replace(",", " "))
    search_url = f"https://www.tiktok.com/search?q={encoded}"
    driver.get(search_url)
    time.sleep(2 + random.uniform(0.5, 1.0))
    log(f"🔄 Rotating search page: {query_str}")

    collected = set()
    attempt = 0
    while len(collected) < desired_count and attempt < retries:
        attempt += 1
        # scroll a few times
        for _ in range(scroll_cycles):
            driver.execute_script(f"window.scrollBy(0, {random.randint(800,1600)})")
            time.sleep(random.uniform(0.8, 1.0))
        try:
            wait = WebDriverWait(driver, 6)
            wait.until(EC.presence_of_element_located((By.XPATH, "//a[contains(@href,'/video/')]")))
        except TimeoutException:
            # try refresh and another attempt
            try:
                driver.refresh()
                time.sleep(0.7)
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

# ---------------- Per-user DB helpers ----------------
def get_user_db_path(user_id):
    os.makedirs("user_dbs", exist_ok=True)
    return os.path.join("user_dbs", f"{user_id}.db")

def get_user_db_conn(user_id):
    db_path = get_user_db_path(user_id)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    init_user_db_conn(conn)  # always ensure tables exist
    return conn

def init_user_db_conn(conn):
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS ai_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        query_text TEXT,
        result_urls TEXT,
        ts TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS valuable_words (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        word TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS sent_videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT,
        url TEXT,
        sent_at TEXT
    )""")
    conn.commit()

def save_valuable_words_threadsafe(user_id, words):
    conn = get_user_db_conn(user_id)
    save_valuable_words(conn, user_id, words)
    conn.close()

async def load_valuable_words_threadsafe(user_id):
    conn = get_user_db_conn(user_id)
    words = await load_valuable_words(conn, user_id)
    conn.close()
    return words

async def load_ai_memory_threadsafe(user_id, limit=50):
    conn = get_user_db_conn(user_id)
    mem = await load_ai_memory(conn, user_id, limit)
    conn.close()
    return mem

def mark_urls_sent_threadsafe(user_id, urls, video_ids=None):
    conn = get_user_db_conn(user_id)
    mark_urls_sent(conn, urls, video_ids)
    conn.close()

def save_ai_memory_threadsafe(user_id, query_text, result_urls):
    conn = get_user_db_conn(user_id)
    save_ai_memory(conn, user_id, query_text, result_urls)
    conn.close()

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
        # <<< PATCHED: synchronous call wrapped in asyncio.to_thread >>>
        resp = await asyncio.to_thread(lambda: openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        ))
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
async def parse_user_request(text: str, memory=None, last_sent_urls=None, user_id=None):
    text = (text or "").strip()
    if not text:
        return None, 0, None, None

    if last_sent_urls and re.search(r"more (like|similar) (the )?(last|previous)", text, re.I):
        return "__FOLLOWUP__", len(last_sent_urls), f"Do you want me to send {len(last_sent_urls)} more videos similar to the last ones?", None

    suggested_prompt = None
    query = None

    db_conn = get_user_db_conn(user_id) if user_id else None

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
            cleaned_words = re.sub(r"\b(send|me|need|please|find|the|that|i'm|i am|like|videos|video|of|for|now|what|you|to|do|is|about|okay|bloody|perfect|most|recent)\b", " ", text, flags=re.I)
            cleaned_words = re.sub(r"\d+", " ", cleaned_words)
            cleaned_words = re.sub(r"[^\w\s]", " ", cleaned_words)
            words = [w for w in cleaned_words.split() if len(w) > 1]
            if db_conn:
                await asyncio.to_thread(save_valuable_words, db_conn, user_id, words)

            # <<< PATCHED: synchronous call wrapped in asyncio.to_thread >>>
            resp = await asyncio.to_thread(lambda: openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
            ))
            raw = resp.choices[0].message.content.strip()
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
        if db_conn:
            await asyncio.to_thread(save_valuable_words, db_conn, user_id, words)

    valuable_words = await load_valuable_words_threadsafe(user_id) if user_id else None
    alt_prompts = await expand_query_with_gpt(query, valuable_words)
    suggested_prompt = f"I will search {count} videos for '{alt_prompts[0]}', is that okay?"

    if db_conn:
        db_conn.close()

    return query, count, suggested_prompt, alt_prompts

# ---------------- Telegram helpers ----------------
def make_markup():
    return InlineKeyboardMarkup([[InlineKeyboardButton("Next ▶️", callback_data="next")]])

# ---------------- AI Fresh URL filter ----------------
async def ai_filter_fresh_urls(user_id, candidate_urls, desired_count):
    conn = get_user_db_conn(user_id)
    cur = conn.cursor()
    fresh = []
    for url in candidate_urls:
        vid = url.rstrip("/").split("/")[-1]
        cur.execute("SELECT 1 FROM sent_videos WHERE video_id = ?", (vid,))
        if not cur.fetchone():
            fresh.append(url)
        if len(fresh) >= desired_count:
            break
    conn.close()

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

# ---- MODIFIED handle_user_request: added db_conn keyword ----
async def handle_user_request(update: Update, context: ContextTypes.DEFAULT_TYPE, tiktok_collect_fn, downloader_fn, *, db_conn=None):
    user_text = update.message.text or ""
    user = update.effective_user
    user_id = user.id
    await update.message.chat.send_action("typing")
    log(f"🤖 user asked: {user_text}")

    memory = await load_ai_memory_threadsafe(user_id)
    last_sent_urls = memory[0]["urls"] if memory else None

    query, count, confirmation_prompt, alt_prompts = await parse_user_request(user_text, memory, last_sent_urls, user_id)

    if not query or count <= 0:
        await update.message.reply_text("I couldn't understand your request. Try something like: `send 5 funny edits`")
        return

    if query == "__FOLLOWUP__" and last_sent_urls:
        candidate_urls = last_sent_urls
        await update.message.reply_text(confirmation_prompt)
    else:
        if confirmation_prompt:
            buttons = [
                [InlineKeyboardButton("✅ Yes", callback_data="confirm")],
                [InlineKeyboardButton("❌ Cancel", callback_data="cancel")]
            ]
            await update.message.reply_text(confirmation_prompt, reply_markup=InlineKeyboardMarkup(buttons))
            context.user_data["awaiting_confirmation"] = True
            context.user_data["confirmation_result"] = None

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

    fresh = await ai_filter_fresh_urls(user_id, candidate_urls, count)
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

    mark_urls_sent_threadsafe(user_id, fresh)
    save_ai_memory_threadsafe(user_id, user_text, fresh)

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


# ========================
# END OF BOT.PY
# ========================

