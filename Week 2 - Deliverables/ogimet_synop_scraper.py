#!/usr/bin/env python3
"""
OGIMET Undecoded SYNOP Scraper
Fetches raw SYNOP reports from:  http://www.ogimet.com/cgi-bin/getsynop

CSV output columns:
    WMOIND, YEAR, MONTH, DAY, HOUR, MIN, REPORT
"""

import csv
import io
import logging
import sys
import time
from datetime import datetime, date, timedelta, timezone
from typing import Iterator, List, Tuple

import requests

# ==============================================================================
#  ✏️  CONFIGURATION  —  only edit this section
# ==============================================================================

# WMO station index.  Single string or a list for multiple stations.
#   Single   →  WMO_STATIONS = "98444"
#   Multiple →  WMO_STATIONS = ["98444", "98222"]
WMO_STATIONS = "98427"

# Year range.  END_YEAR = None  →  uses the current year automatically.
START_YEAR = 2000
END_YEAR   = None

# Output CSV filename (written next to this script).
# Set to None to print a short preview in the terminal instead.
OUTPUT_FILE = "synop_undecoded.csv"

# ==============================================================================
#  ⚙️  ADVANCED SETTINGS
# ==============================================================================

# Days per HTTP request.  Smaller = lighter server load = fewer 504s.
# Raise to 30 for speed once you confirm it works.
CHUNK_DAYS = 365

# Seconds to wait between requests.  OGIMET asks users to be polite.
DELAY_SECS = 20

# Retry logic
MAX_RETRIES = 3
RETRY_DELAY = 30   # seconds (doubles each retry)

# ==============================================================================
#  ── internals ──────────────────────────────────────────────────────────────
# ==============================================================================

BASE_URL = "http://www.ogimet.com/cgi-bin/getsynop"
COLUMNS  = ["WMOIND", "YEAR", "MONTH", "DAY", "HOUR", "MIN", "REPORT"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Mimic a real browser — OGIMET rejects bot-like user agents
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept"          : "text/plain, text/html, */*",
    "Accept-Language" : "en-US,en;q=0.9",
    "Connection"      : "keep-alive",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def _chunks(start_year: int, end_year: int, days: int) -> Iterator[Tuple[datetime, datetime]]:
    """Yield (begin, end) datetime pairs in `days`-sized steps."""
    now   = datetime.now(timezone.utc).replace(tzinfo=None)
    start = datetime(start_year, 1, 1, 0, 0)
    stop  = min(datetime(end_year, 12, 31, 23, 59), now)
    cur   = start
    while cur <= stop:
        end = min(cur + timedelta(days=days) - timedelta(minutes=1), stop)
        yield cur, end
        cur = end + timedelta(minutes=1)


# ── fetch ─────────────────────────────────────────────────────────────────────

def _get(wmo: str, begin: datetime, end: datetime) -> str | None:
    """
    Make one HTTP GET to getsynop and return the raw response text.

    Key parameter choices:
      block = full 5-digit WMO  →  acts as an exact-station prefix filter
              (OGIMET matches stations whose index STARTS WITH this value;
               a 5-digit value therefore selects exactly one station)
      header=yes  →  first line is the CSV column header row
    """
    params = {
        "block"  : wmo,          # e.g. "98444"
        "begin"  : _fmt(begin),  # e.g. "202401010000"
        "end"    : _fmt(end),    # e.g. "202401072359"
        "header" : "yes",
        "lang"   : "eng",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                BASE_URL, params=params,
                headers=HTTP_HEADERS,
                timeout=120,
            )

            if resp.status_code in (429, 503, 504):
                raise requests.HTTPError(
                    f"HTTP {resp.status_code} — backing off"
                )
            resp.raise_for_status()

            # Log the exact URL on the first attempt so you can test it manually
            if attempt == 1:
                log.debug("  URL: %s", resp.url)

            return resp.text

        except requests.RequestException as exc:
            wait = RETRY_DELAY * attempt
            log.warning("  Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                log.info("  Retrying in %ds ...", wait)
                time.sleep(wait)
            else:
                log.error("  Giving up: WMO=%s  %s→%s", wmo, _fmt(begin), _fmt(end))

    return None


# ── parse ─────────────────────────────────────────────────────────────────────

def _parse(text: str, wmo: str) -> List[dict]:
    """
    Parse the raw getsynop response into a list of row dicts.

    Handles two cases:
      A) Response includes a header row  (header=yes was honoured)
      B) Response is headerless CSV      (some OGIMET mirrors ignore header=yes)

    Also filters to only rows belonging to `wmo` (block= may return neighbours).
    """
    if not text:
        return []

    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []

    # Detect error / empty responses
    first_upper = lines[0].upper()
    if any(kw in first_upper for kw in
           ("ERROR", "NO DATA", "NOT FOUND", "HOST NOT", "FORBIDDEN")):
        log.debug("  Server message: %s", lines[0][:100])
        return []

    rows = []

    # Case A: header row present
    if "WMOIND" in first_upper or "YEAR" in first_upper:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            # Strip whitespace from every key and value
            clean = {k.strip(): v.strip() for k, v in row.items() if k}
            if clean.get("WMOIND") == wmo:
                rows.append({c: clean.get(c, "") for c in COLUMNS})

    # Case B: no header — parse positionally
    else:
        for line in lines:
            # Split on comma into at most 7 parts
            # (REPORT field can contain spaces but not commas, so this is safe)
            parts = line.split(",", 6)
            if len(parts) < 6:
                continue
            record = {COLUMNS[i]: parts[i].strip() for i in range(min(7, len(parts)))}
            if record.get("WMOIND") == wmo:
                rows.append(record)

    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def run():
    current_year = datetime.now(timezone.utc).year

    # Normalise WMO_STATIONS to a list
    wmo_list = [WMO_STATIONS] if isinstance(WMO_STATIONS, str) else list(WMO_STATIONS)

    start_year = int(START_YEAR)
    end_year   = current_year if END_YEAR is None else int(END_YEAR)

    if start_year > end_year:
        log.error("START_YEAR (%d) must be <= END_YEAR (%d).", start_year, end_year)
        sys.exit(1)
    if end_year > current_year:
        log.warning("END_YEAR %d is in the future; capping at %d.", end_year, current_year)
        end_year = current_year

    chunks    = list(_chunks(start_year, end_year, CHUNK_DAYS))
    total_req = len(wmo_list) * len(chunks)

    log.info("=" * 60)
    log.info("  Station(s)   : %s", ", ".join(wmo_list))
    log.info("  Period       : %d – %d", start_year, end_year)
    log.info("  Chunk size   : %d days  (%d chunks)", CHUNK_DAYS, len(chunks))
    log.info("  Total reqs   : %d  (delay %ds each)", total_req, DELAY_SECS)
    log.info("  Output       : %s", OUTPUT_FILE or "terminal preview")
    log.info("=" * 60)

    all_rows: List[dict] = []
    req_n = 0

    for wmo in wmo_list:
        log.info("── WMO %s ──────────────────────────────────────────", wmo)
        for begin, end in chunks:
            label = f"{begin:%Y-%m-%d} → {end:%Y-%m-%d}"
            log.info("  [%d/%d]  %s", req_n + 1, total_req, label)

            text = _get(wmo, begin, end)
            req_n += 1

            if text is not None:
                rows = _parse(text, wmo)
                log.info("    → %d record(s)", len(rows))
                all_rows.extend(rows)
            else:
                log.warning("    → skipped (no response)")

            if req_n < total_req:
                time.sleep(DELAY_SECS)

    log.info("=" * 60)
    log.info("  Total records : %d", len(all_rows))

    # ── write output ──────────────────────────────────────────────────────
    if OUTPUT_FILE:
        if all_rows:
            with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_rows)
            log.info("  Saved → %s", OUTPUT_FILE)
        else:
            log.warning("  No records collected — CSV not written.")
            log.warning("  Troubleshooting tips:")
            log.warning("    1. Open this URL in your browser to confirm data exists:")
            log.warning("       http://www.ogimet.com/cgi-bin/getsynop"
                        "?block=%s&begin=202401010000&end=202401080000&header=yes",
                        wmo_list[0])
            log.warning("    2. If the browser shows data, try reducing CHUNK_DAYS to 3.")
            log.warning("    3. If the browser shows 'No data', the station has no records.")
    else:
        if all_rows:
            print("\n" + ",".join(COLUMNS))
            for row in all_rows[:30]:
                print(",".join(str(row.get(c, "")) for c in COLUMNS))
            if len(all_rows) > 30:
                print(f"  ... and {len(all_rows) - 30} more rows.")
        else:
            print("No records found.")

    log.info("=" * 60)


if __name__ == "__main__":
    run()
