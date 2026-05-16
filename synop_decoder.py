#!/usr/bin/env python3
"""
SYNOP Decoder
=============
Decodes WMO SYNOP (FM-12) reports from a CSV input file.

Input CSV columns : WMOIND, YEAR, MONTH, DAY, HOUR, MIN, REPORT
Output CSV columns: station_id, datetime, lat, lon, temp, pressure,
                    humidity, wind_speed, wind_dir, cloud_cover,
                    visibility, rain_3h

Usage
-----
    python synop_decoder.py input.csv output.csv

Optional flags
--------------
    --sep <char>    Column separator of the input file  (default: auto-detect)
    --lat-lon <file> Path to a WMO station lat/lon CSV (columns: wmo_id,lat,lon)
                    Falls back to None if the file is absent or the station
                    is not listed.
"""

import csv
import sys
import argparse
import os
import re
from datetime import datetime
import calendar as _calendar


def _safe_int(val, default=0):
    """Extract the first integer from val; return default if none found."""
    m = re.search(r"\d+", str(val))
    return int(m.group()) if m else default


MISSING = "N/A"  # written to CSV for empty/missing values

# ---------------------------------------------------------------------------
# Optional: small bundled lookup table for common WMO stations.
# You can replace / extend this with a full station list CSV via --lat-lon.
# ---------------------------------------------------------------------------
STATION_COORDS: dict[str, tuple[float | None, float | None]] = {
    # fmt: off
    # WMO-ID : (lat, lon)
    "98444": (13.15, 123.74),   # Albay — Legazpi / Bicol International Airport
    "98446": (13.57, 124.20),   # Catanduanes — Virac Airport
    "98543": (12.36, 123.62),   # Masbate — Moises R. Espinosa Airport
    "98536": (12.58, 122.26),   # Romblon
    "98427": (13.96, 121.59),   # Tayabas
    "98434": (14.74, 121.64),   # Infanta
    "98440": (14.12, 122.98),   # Camarines Norte — Daet Airport
    # fmt: on
}


# ---------------------------------------------------------------------------
# SYNOP group decoders
# ---------------------------------------------------------------------------

def decode_irix(group: str) -> dict:
    """Section 0 indicator group  iRiXhVV  (5 chars).

    iR (precipitation indicator):
        0/1/2 – group 6 present somewhere  → no sentinel (use decoded value)
        3     – precip amount = 0, no group 6  → rain_sentinel = 0.0  (Case B)
        4     – station does not measure precip → rain_sentinel = None (Case A)
    """
    out = {}
    if len(group) != 5:
        return out
    ir = group[0]
    if ir == "3":
        out["rain_sentinel"] = 0.0   # "no rain reported" → 0.0 mm
    elif ir == "4":
        out["rain_sentinel"] = None  # truly missing (station doesn't measure)
    # iX – station type / present-weather indicator  (ignored)
    # h  – height of cloud base (code table 1600)
    h_code = group[2]
    h_table = {"0": 0, "1": 50, "2": 100, "3": 200, "4": 300,
               "5": 600, "6": 1000, "7": 1500, "8": 2000, "9": 2500, "/": None}
    out["cloud_base_m"] = h_table.get(h_code)
    # VV – visibility (code table 4377)
    vv = group[3:5]
    out["visibility_m"] = _decode_vv(vv)
    return out


def _decode_vv(vv: str) -> float | None:
    """Convert VV code (00-99) to metres (WMO Code Table 4377)."""
    if not vv.isdigit():
        return None   # corrupted token e.g. '?8', 'YP', '6A', '6T'
    v = int(vv)
    if 0 <= v <= 50:
        return v * 100           # 00-50 → 0 … 5 000 m
    elif 51 <= v <= 80:
        return (v - 50) * 1000 + 5000   # 51-80 → 6 000 … 35 000 m
    elif 81 <= v <= 89:
        return (v - 80) * 5000 + 35000  # 81-89 → 40 000 … 80 000 m
    elif v == 90:
        return None   # < 50 m (below scale)
    elif 91 <= v <= 98:
        # WMO Table 4377: specific low-visibility codes
        _vv_91_98 = {91: 50, 92: 200, 93: 500, 94: 1000,
                     95: 2000, 96: 4000, 97: 10000, 98: 40000}
        return float(_vv_91_98[v])
    elif v == 99:
        return None   # sky obscured / visibility not measurable
    return None


def decode_nddff(group: str) -> dict:
    """Cloud / wind group  Nddff  (5 chars).

    When ff=99 the actual wind speed exceeds 98 units and is encoded in the
    immediately following group 00fff (3-digit speed).  In that case
    wind_speed is set to None and wind_speed_needs_00fff=True is set so the
    caller knows to read the next group.
    """
    out = {}
    if len(group) != 5:
        return out
    # N – total cloud cover (oktas 0-8, / = obscured)
    n = group[0]
    if n == "/":
        out["cloud_cover"] = None
    elif n.isdigit():
        out["cloud_cover"] = int(n)   # oktas
    # dd – wind direction in tens of degrees (00-36, 00=calm, 99=variable)
    dd = group[1:3]
    if dd.isdigit():
        dd_val = int(dd)
        out["wind_dir"] = None if dd_val == 99 else dd_val * 10
    # ff – wind speed (m/s or knots depending on iW; assume m/s)
    # ff=99 means speed >= 99 units; actual value is in the next group 00fff
    ff = group[3:5]
    if ff.isdigit():
        if int(ff) == 99:
            out["wind_speed"] = None
            out["wind_speed_needs_00fff"] = True
        else:
            out["wind_speed"] = int(ff)
    return out


def decode_1sTTT(group: str) -> dict:
    """Air temperature group  1SnTTT."""
    out = {}
    if len(group) != 5 or group[0] != "1":
        return out
    sign = group[1]
    ttt = group[2:5]
    if ttt.isdigit() and sign in ("0", "1"):
        t = int(ttt) / 10.0
        out["temp"] = -t if sign == "1" else t
    return out


def decode_2sTdTdTd(group: str) -> dict:
    """Dew-point group  2SnTdTdTd → derive relative humidity."""
    out = {}
    if len(group) != 5 or group[0] != "2":
        return out
    sign = group[1]
    ttt = group[2:5]
    if ttt.isdigit() and sign in ("0", "1"):
        td = int(ttt) / 10.0
        if sign == "1":
            td = -td
        out["dewpoint"] = td
    return out


def decode_3PPPP(group: str) -> dict:
    """Station pressure group  3PPPP."""
    out = {}
    if len(group) != 5 or group[0] != "3":
        return out
    pppp = group[1:5]
    if pppp.isdigit():
        p = int(pppp) / 10.0
        # PPPP encodes tenths of hPa.  Values like "0980" → 98.0 are impossible
        # for real surface pressure; adding 1000 gives the correct 1098.0 hPa.
        # Threshold 500 hPa safely separates the ambiguous low-encoded values
        # from any legitimate reading (no surface station sits below ~500 hPa).
        if p < 500.0:
            p += 1000.0
        out["pressure_station"] = p
    return out


def decode_4PPPP(group: str) -> dict:
    """Sea-level pressure group  4PPPP."""
    out = {}
    if len(group) != 5 or group[0] != "4":
        return out
    pppp = group[1:5]
    if pppp.isdigit():
        p = int(pppp) / 10.0
        # Same threshold as decode_3PPPP: 500 hPa separates low-encoded
        # values from any physically plausible sea-level pressure.
        if p < 500.0:
            p += 1000.0
        out["pressure"] = p
    return out


def decode_6RRRt(group: str) -> dict:
    """Precipitation group  6RRRt."""
    out = {}
    if len(group) != 5 or group[0] != "6":
        return out
    rrr = group[1:4]
    t   = group[4]          # duration indicator
    if rrr.isdigit():
        r = int(rrr)
        if r == 990:
            rain = 0.1      # trace rainfall → 0.1 mm  (Case C)
        elif r >= 991:
            rain = (r - 990) / 10.0
        else:
            rain = float(r) / 10.0   # rrr is in tenths-of-mm per WMO standard
        out["rain_3h"] = rain
    return out


def decode_7wwW1W2(group: str) -> dict:
    """Present/past weather group  7wwW1W2  (informational, not in output)."""
    return {}


def decode_8NhClCmCh(group: str) -> dict:
    """Cloud group  8NhClCmCh."""
    out = {}
    if len(group) != 5 or group[0] != "8":
        return out
    # Nh – cloud cover of low / middle cloud (oktas)
    nh = group[1]
    if nh.isdigit():
        out["cloud_cover_low"] = int(nh)
    return out


# ---------------------------------------------------------------------------
# Humidity from temperature and dew-point (August-Roche-Magnus formula)
# ---------------------------------------------------------------------------

def _rh_magnus(T: float, Td: float) -> float:
    """Return relative humidity (%) from air temperature and dew-point (°C).

    Uses the August-Roche-Magnus approximation:
        RH = 100 * exp( a*Td/(b+Td) - a*T/(b+T) )
    with a=17.625, b=243.04 (valid roughly −40 °C to +60 °C).
    Result is clamped to [0, 100] to guard against floating-point overshoot.
    """
    import math
    a, b = 17.625, 243.04
    gamma_T  = a * T  / (b + T)
    gamma_Td = a * Td / (b + Td)
    rh = round(100.0 * math.exp(gamma_Td - gamma_T), 1)
    return max(0.0, min(100.0, rh))


# ---------------------------------------------------------------------------
# Main SYNOP parser
# ---------------------------------------------------------------------------

def parse_synop(report: str) -> dict:
    """
    Parse a single SYNOP FM-12 report string.
    Returns a dict with decoded fields (None where not reported).

    NOTE: This is the original v1 parser kept for reference.  Production code
    uses parse_synop_v2 which also scans Section 3 for precipitation.
    """
    result = {
        "temp": None,
        "pressure": None,
        "humidity": None,
        "wind_speed": None,
        "wind_dir": None,
        "cloud_cover": None,
        "visibility_m": None,
        "rain_3h": None,
    }

    # Clean up and split
    report = report.strip().rstrip("=").strip()
    tokens = report.split()

    if not tokens:
        return result

    # Skip section header tokens (AAXX, BBXX, OOXX) and YYGGiw
    idx = 0
    if tokens[idx] in ("AAXX", "BBXX", "OOXX"):
        idx += 1   # skip section id
    if idx < len(tokens):
        idx += 1   # skip YYGGiw (day/hour/wind-indicator)
    if idx < len(tokens):
        idx += 1   # skip station number (IIiii)

    # Parse remaining 5-char groups in section 1
    # Stop at section 2 / 3 markers (222, 333, 444, 555)
    dewpoint = None
    rain_sentinel = "UNSET"   # set from iR indicator in group 0
    while idx < len(tokens):
        g = tokens[idx]
        idx += 1

        if g in ("222", "333", "444", "555"):
            break

        if len(g) != 5:
            continue

        lead = g[0]

        if lead == "0":                          # iRiXhVV
            d = decode_irix(g)
            result["visibility_m"] = d.get("visibility_m")
            if "rain_sentinel" in d:
                rain_sentinel = d["rain_sentinel"]

        elif lead == "N" or (lead.isdigit() and g[1:3].isdigit() and g[3:5].isdigit() and lead in "012345678/"):
            # Nddff – must be first group after station id
            if lead in "012345678/":
                dd = g[1:3]
                ff = g[3:5]
                if (dd.isdigit() or dd == "//") and (ff.isdigit() or ff == "//"):
                    d = decode_nddff(g)
                    result.update({k: v for k, v in d.items() if k in result})

        elif lead == "1":                        # was a bare `if`, causing fall-through
            d = decode_1sTTT(g)
            if "temp" in d:
                result["temp"] = d["temp"]

        elif lead == "2":
            d = decode_2sTdTdTd(g)
            if "dewpoint" in d:
                dewpoint = d["dewpoint"]

        elif lead == "3":
            d = decode_3PPPP(g)
            # Only use station pressure if sea-level not yet found
            if "pressure_station" in d and result["pressure"] is None:
                result["pressure"] = d["pressure_station"]

        elif lead == "4":
            d = decode_4PPPP(g)
            if "pressure" in d:
                result["pressure"] = d["pressure"]

        elif lead == "5":
            pass  # pressure tendency – skip

        elif lead == "6":
            d = decode_6RRRt(g)
            if "rain_3h" in d:
                result["rain_3h"] = d["rain_3h"]

        elif lead == "7":
            pass  # present weather – skip

        elif lead == "8":
            d = decode_8NhClCmCh(g)
            # Use low-cloud oktas as cloud_cover if not yet set
            if result["cloud_cover"] is None and "cloud_cover_low" in d:
                result["cloud_cover"] = d["cloud_cover_low"]

        elif lead == "9":
            pass  # additional data – skip

    # Apply rain sentinel: if no group-6 was decoded, use iR indicator
    if result["rain_3h"] is None and rain_sentinel != "UNSET":
        result["rain_3h"] = rain_sentinel   # 0.0 (no rain) or None (truly missing)

    # Derive humidity
    if result["temp"] is not None and dewpoint is not None:
        if dewpoint > result["temp"]:
            print(
                f"  [WARN] Unphysical dew-point: Td={dewpoint} °C > T={result['temp']} °C "
                f"— humidity clamped to 100 %",
                file=sys.stderr,
            )
        result["humidity"] = _rh_magnus(result["temp"], dewpoint)

    return result


# ---------------------------------------------------------------------------
# Nddff group needs a dedicated second pass because the lead digit is
# ambiguous (it can be 0-8 or /).  Re-implement cleanly.
# ---------------------------------------------------------------------------

def _extract_nddff(tokens: list[str], start_idx: int) -> tuple[dict, int]:
    """
    Find and decode the Nddff group.  In FM-12 Section 1 it is always
    the FIRST 5-char data group after the station number.

    When ff=99 the wind speed exceeds 98 units and the actual value is
    encoded in the immediately following group 00fff (e.g. "00105" = 105 m/s
    or kt).  This function reads that extension group when present.

    Returns (decoded_dict, consumed_index).
    """
    out = {}
    if start_idx >= len(tokens):
        return out, start_idx
    g = tokens[start_idx]
    if len(g) == 5:
        n, dd, ff = g[0], g[1:3], g[3:5]
        n_ok  = n  in "012345678/"
        dd_ok = dd.isdigit() or dd == "//"
        ff_ok = ff.isdigit() or ff == "//"
        if n_ok and dd_ok and ff_ok:
            if n.isdigit():
                out["cloud_cover"] = int(n)
            if dd.isdigit():
                dd_val = int(dd)
                out["wind_dir"] = None if dd_val == 99 else dd_val * 10
            if ff.isdigit():
                ff_val = int(ff)
                if ff_val == 99:
                    # 3-digit wind speed follows in group 00fff
                    next_idx = start_idx + 1
                    if next_idx < len(tokens):
                        nxt = tokens[next_idx]
                        if len(nxt) == 5 and nxt[:2] == "00" and nxt[2:].isdigit():
                            out["wind_speed"] = int(nxt[2:])
                            return out, next_idx + 1   # consumed both groups
                    # 00fff group absent or malformed — leave wind_speed as None
                    out["wind_speed"] = None
                else:
                    out["wind_speed"] = ff_val
            return out, start_idx + 1
    return out, start_idx


def parse_synop_v2(report: str) -> dict:
    """
    Improved two-pass SYNOP parser.

    Handles:
    - Section 1 (mandatory groups)
    - Section 3 (333 block) for precipitation when iR=0 or iR=2
    - NIL= reports (no observation)
    - iR indicator for rain sentinel (Case A/B/C)
    """
    result = {
        "temp": None,
        "pressure": None,
        "humidity": None,
        "wind_speed": None,
        "wind_dir": None,
        "cloud_cover": None,
        "visibility_m": None,
        "rain_3h": None,
    }

    report = report.strip().rstrip("=").strip()
    tokens = report.split()
    if not tokens:
        return result

    idx = 0
    # Skip AAXX / BBXX / OOXX
    if idx < len(tokens) and tokens[idx] in ("AAXX", "BBXX", "OOXX"):
        idx += 1
    # Read YYGGiw — last char encodes wind unit:
    #   iW=0 or 1 → m/s   (no conversion needed)
    #   iW=3 or 4 → knots → multiply by 0.514444 to get m/s
    wind_in_knots = False
    if idx < len(tokens):
        yyggiw = tokens[idx]
        if len(yyggiw) >= 1:
            iw = yyggiw[-1]
            wind_in_knots = iw in ("3", "4")
        idx += 1
    # Skip IIiii (station number)
    if idx < len(tokens):
        idx += 1

    # ---- group 0: iRiXhVV
    # Always first in Section 1. iR tells us where to find precipitation group 6:
    #   iR=0 → group 6 in sections 1 AND 3
    #   iR=1 → group 6 in section 1 only
    #   iR=2 → group 6 in section 3 only  ← common cause of blank rain_3h!
    #   iR=3 → no precip (amount=0)        → rain_sentinel=0.0
    #   iR=4 → station doesn't measure     → rain_sentinel=None
    rain_sentinel = "UNSET"
    ir_code = None   # remember iR to know whether to scan section 3
    if idx < len(tokens) and len(tokens[idx]) == 5:
        d = decode_irix(tokens[idx])
        result["visibility_m"] = d.get("visibility_m")
        if "rain_sentinel" in d:
            rain_sentinel = d["rain_sentinel"]
        ir_code = tokens[idx][0]   # '0','1','2','3','4' or '/'
        idx += 1

    # ---- group Nddff
    nd, idx = _extract_nddff(tokens, idx)
    result.update({k: v for k, v in nd.items() if k in result})

    # Convert wind speed from knots to m/s when iW indicates knots (iW=3 or 4).
    # 1 knot = 0.514444 m/s; round to 1 decimal place.
    if wind_in_knots and result.get("wind_speed") is not None:
        result["wind_speed"] = round(result["wind_speed"] * 0.514444, 1)

    # ---- Section 1: remaining groups (stop at section markers)
    dewpoint = None
    while idx < len(tokens):
        g = tokens[idx]
        idx += 1

        if g in ("222", "333", "444", "555"):
            # If we hit 333, scan section 3 for precipitation
            if g == "333":
                _parse_section3(tokens, idx, result, ir_code)
            break
        if len(g) != 5:
            continue

        lead = g[0]
        if not lead.isdigit():
            continue

        if lead == "1":
            d = decode_1sTTT(g)
            if "temp" in d:
                result["temp"] = d["temp"]

        elif lead == "2":
            d = decode_2sTdTdTd(g)
            if "dewpoint" in d:
                dewpoint = d["dewpoint"]

        elif lead == "3":
            d = decode_3PPPP(g)
            if "pressure_station" in d and result["pressure"] is None:
                result["pressure"] = d["pressure_station"]

        elif lead == "4":
            d = decode_4PPPP(g)
            if "pressure" in d:
                result["pressure"] = d["pressure"]

        elif lead == "6":
            d = decode_6RRRt(g)
            if "rain_3h" in d:
                result["rain_3h"] = d["rain_3h"]

        elif lead == "8":
            d = decode_8NhClCmCh(g)
            if result["cloud_cover"] is None and "cloud_cover_low" in d:
                result["cloud_cover"] = d["cloud_cover_low"]

    # Apply rain sentinel: if no group-6 was decoded anywhere, use iR indicator
    if result["rain_3h"] is None and rain_sentinel != "UNSET":
        result["rain_3h"] = rain_sentinel   # 0.0 (no rain) or None (truly missing)

    # Derive humidity
    if result["temp"] is not None and dewpoint is not None:
        if dewpoint > result["temp"]:
            print(
                f"  [WARN] Unphysical dew-point: Td={dewpoint} °C > T={result['temp']} °C "
                f"— humidity clamped to 100 %",
                file=sys.stderr,
            )
        result["humidity"] = _rh_magnus(result["temp"], dewpoint)

    return result


def _parse_section3(tokens: list, start_idx: int, result: dict, ir_code: str) -> None:
    """
    Scan Section 3 (after 333 marker) for precipitation group 6RRRt.

    Called when iR=0 (precip in s1 AND s3) or iR=2 (precip in s3 ONLY).
    Also scans for min/max temperature groups (1SnTxTxTx / 2SnTnTnTn)
    which are commonly reported in section 3.
    """
    # Only parse rain from section 3 when iR explicitly indicates it's there.
    # Excluding None and "/" avoids double-counting when group 0 was absent or
    # malformed and group 6 may already have been decoded from section 1.
    want_rain = ir_code in ("0", "2")

    idx = start_idx
    while idx < len(tokens):
        g = tokens[idx]
        idx += 1

        # Stop at next section marker
        if g in ("222", "444", "555"):
            break
        if len(g) != 5:
            continue

        lead = g[0]
        if not lead.isdigit():
            continue

        # Group 6RRRt in section 3
        if lead == "6" and want_rain and result["rain_3h"] is None:
            d = decode_6RRRt(g)
            if "rain_3h" in d:
                result["rain_3h"] = d["rain_3h"]


# ---------------------------------------------------------------------------
# Station coordinate lookup
# ---------------------------------------------------------------------------

def load_station_coords(path: str) -> dict[str, tuple[float, float]]:
    coords = {}
    if not path or not os.path.isfile(path):
        return coords
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wmo = str(row.get("wmo_id", "")).strip()
            try:
                lat = float(row.get("lat", ""))
                lon = float(row.get("lon", ""))
                coords[wmo] = (lat, lon)
            except (ValueError, TypeError):
                pass
    return coords


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

OUTPUT_COLS = [
    "station_id", "date", "time",
    "lat", "lon",
    "temp", "pressure", "humidity",
    "wind_speed", "wind_dir",
    "cloud_cover", "visibility_m",
    "rain_3h",
]


def detect_separator(path: str) -> str:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(4096)
    counts = {s: sample.count(s) for s in (",", ";", "\t", "|")}
    return max(counts, key=counts.get)


def process(input_path: str, output_path: str,
            sep: str | None = None,
            station_coords_path: str | None = None) -> int:

    ext_coords = load_station_coords(station_coords_path) if station_coords_path else {}

    if sep is None:
        sep = detect_separator(input_path)

    decoded_rows = 0
    errors = 0

    with open(input_path, newline="", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin, delimiter=sep)
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()

        for lineno, row in enumerate(reader, start=2):   # 2 = first data line
            # Always write a row — even on error, output N/A for bad fields
            station_id = ""
            date_str   = ""
            time_str   = ""
            lat = lon  = MISSING
            decoded    = {}
            parse_error = None

            try:
                # ---- identifiers (case-insensitive column names) ----
                def _col(*names):
                    for n in names:
                        v = row.get(n)
                        if v is not None:
                            return str(v).strip()
                    return ""

                station_id = _col("WMOIND", "wmoind")
                year   = _col("YEAR",  "year")
                month  = _col("MONTH", "month")
                day    = _col("DAY",   "day")
                hour   = _col("HOUR",  "hour")
                minute = _col("MIN",   "min") or "0"
                report = _col("REPORT","report")

                # ---- datetime (robust: always produce a valid datetime string) ----
                # _safe_int extracts digits and clamps — NEVER raises, NEVER calls int() directly
                _y  = _safe_int(year,   2000)
                _mo = max(1, min(12,  _safe_int(month,  1)))
                _d  = max(1, min(_calendar.monthrange(_y, _mo)[1], _safe_int(day, 1)))
                _h  = max(0, min(23,  _safe_int(hour,   0)))   # "?8"→8, "YP"→0, "6A"→6, "6T"→6
                _mi = max(0, min(59,  _safe_int(minute, 0)))
                # This cannot raise — all inputs are clamped ints
                _dt     = datetime(_y, _mo, _d, _h, _mi)
                date_str = _dt.strftime("%Y-%m-%d")
                time_str = _dt.strftime("%H:%M:%S")

                # ---- coordinates ----
                _coords = ext_coords.get(station_id,
                          STATION_COORDS.get(station_id, (MISSING, MISSING)))
                lat, lon = _coords

                # ---- SYNOP decode ----
                report_upper = report.strip().rstrip("=").strip().upper()
                is_nil = report_upper.endswith("NIL") or report_upper == "NIL"

                if is_nil:
                    decoded = {}   # all fields → N/A
                else:
                    decoded = parse_synop_v2(report)

            except Exception as exc:
                parse_error = exc
                errors += 1
                import traceback as _tb
                print(f"  [WARN] line {lineno}: {exc}", file=sys.stderr)
                _tb.print_exc(file=sys.stderr)

            # ---- write row (N/A for any missing/failed numeric field) ----
            # _fmt: None/empty → "N/A", anything else → value as-is
            def _fmt(val):
                return MISSING if val is None or val == "" else val

            out_row = {
                "station_id":   station_id,
                "date":         date_str,
                "time":         time_str,
                "lat":          _fmt(lat),
                "lon":          _fmt(lon),
                "temp":         _fmt(decoded.get("temp")),
                "pressure":     _fmt(decoded.get("pressure")),
                "humidity":     _fmt(decoded.get("humidity")),
                "wind_speed":   _fmt(decoded.get("wind_speed")),
                "wind_dir":     _fmt(decoded.get("wind_dir")),
                "cloud_cover":  _fmt(decoded.get("cloud_cover")),
                "visibility_m": _fmt(decoded.get("visibility_m")),
                "rain_3h":      _fmt(decoded.get("rain_3h")),
            }
            writer.writerow(out_row)
            decoded_rows += 1

    return decoded_rows, errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decode SYNOP FM-12 reports from a CSV file."
    )
    parser.add_argument("input",  help="Input CSV file path")
    parser.add_argument("output", help="Output CSV file path")
    parser.add_argument("--sep",  default=None,
                        help="Column separator (auto-detected if omitted)")
    parser.add_argument("--lat-lon", dest="latlon", default=None,
                        help="Optional WMO station lat/lon CSV (columns: wmo_id,lat,lon)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Decoding {args.input}  →  {args.output}")
    n, err = process(args.input, args.output,
                     sep=args.sep, station_coords_path=args.latlon)
    print(f"Done. {n} rows decoded, {err} errors.")


if __name__ == "__main__":
    main()
