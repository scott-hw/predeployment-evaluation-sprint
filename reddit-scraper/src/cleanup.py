"""
cleanup.py — Stage 5: Light text cleanup.

Fixes encoding artifacts, normalizes whitespace, strips Reddit-specific markup.
Writes cleaned text to body_clean column.
Does NOT correct spelling or grammar.

Usage:
    python src/cleanup.py
"""

import argparse
import html
import re
import sys
import pathlib
import logging

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.storage import get_db, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Reddit markdown patterns to strip/normalize
_LINK_RE = re.compile(r'\[([^\]]+)\]\([^)]+\)')           # [text](url) → text
_SUBREDDIT_RE = re.compile(r'/?r/\w+')                    # r/whatever — keep
_USER_RE = re.compile(r'/?u/\w+')                          # u/whoever — keep
_QUOTE_BLOCK_RE = re.compile(r'^>+\s*', re.MULTILINE)     # leading >
_HEADER_RE = re.compile(r'^#{1,6}\s+', re.MULTILINE)      # ## Header
_BOLD_ITALIC_RE = re.compile(r'\*{1,3}([^*]+)\*{1,3}')   # *bold* / **italic**
_STRIKETHROUGH_RE = re.compile(r'~~([^~]+)~~')            # ~~strike~~
_INLINE_CODE_RE = re.compile(r'`[^`]*`')                   # `code`
_CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')             # ```block```
_HORIZONTAL_RULE_RE = re.compile(r'^\s*[-*_]{3,}\s*$', re.MULTILINE)
_WHITESPACE_RE = re.compile(r'\s{2,}')                     # collapse runs of spaces
_NEWLINE_RE = re.compile(r'\n{3,}')                        # collapse excess newlines
# Common mojibake patterns
_MOJIBAKE = [
    ('â', "'"),   # â€™ → '
    ('â', '"'),   # â€œ → "
    ('â', '"'),   # â€  → "
    ('â', '—'),   # â€" → —
    ('â', '–'),   # â€" → –
    ('Â ', ' '),         # Â  → non-breaking space → space
]


def clean_text(text: str) -> str:
    if not text:
        return text

    # 1. HTML entity decoding (&amp; &lt; &#39; etc.)
    text = html.unescape(text)

    # 2. Mojibake repair (common UTF-8 double-encoded sequences)
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)

    # 3. Reddit markdown cleanup
    text = _CODE_BLOCK_RE.sub('', text)          # remove code blocks
    text = _INLINE_CODE_RE.sub('', text)          # remove inline code
    text = _LINK_RE.sub(r'\1', text)              # [text](url) → text
    text = _BOLD_ITALIC_RE.sub(r'\1', text)       # *text* → text
    text = _STRIKETHROUGH_RE.sub(r'\1', text)     # ~~text~~ → text
    text = _QUOTE_BLOCK_RE.sub('', text)          # strip > quote markers
    text = _HEADER_RE.sub('', text)               # strip ## headers
    text = _HORIZONTAL_RULE_RE.sub('', text)      # strip ---

    # 4. Whitespace normalization
    text = _WHITESPACE_RE.sub(' ', text)
    text = _NEWLINE_RE.sub('\n\n', text)
    text = text.strip()

    return text


def main():
    parser = argparse.ArgumentParser(description="Clean body text into body_clean column")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--all", action="store_true", help="Clean all records, not just questions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = get_db(cfg["paths"]["db_path"])

    if args.all:
        where = "body IS NOT NULL AND body_clean IS NULL"
    else:
        where = "is_question = TRUE AND body IS NOT NULL AND body_clean IS NULL"

    rows = conn.execute(f"SELECT id, body FROM records WHERE {where}").fetchall()
    log.info("Cleaning %d records...", len(rows))

    batch = []
    for row_id, body in rows:
        cleaned = clean_text(body)
        batch.append((cleaned, row_id))
        if len(batch) >= 2000:
            conn.executemany("UPDATE records SET body_clean = ? WHERE id = ?", batch)
            batch = []

    if batch:
        conn.executemany("UPDATE records SET body_clean = ? WHERE id = ?", batch)

    conn.commit()
    log.info("Done. Cleaned %d records.", len(rows))
    conn.close()


if __name__ == "__main__":
    main()
