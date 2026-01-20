from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None


# =========================================================
# Regex / constants
# =========================================================

# Find a date anywhere inside a token (OCR may attach pipes etc.)
DATE_IN_TOKEN_RE = re.compile(
    r"(?P<d>\d{1,2})\s*/\s*(?P<m>\d{1,2})\s*/\s*(?P<y>\d{2,4})"
)

# Money token (allow commas, optional leading digit for .00, optional trailing punctuation)
MONEY_TOKEN_RE = re.compile(
    r"^\(?\s*(?:RM\s*)?(?P<num>(?:\d{1,3}(?:,\d{3})*|\d+)?\.\d{2})\s*\)?(?P<trail_sign>[+-])?\s*[\.,;:|]*\s*$",
    re.I,
)

HEADER_HINTS = ("DATE", "TARIKH", "DEBIT", "CREDIT", "BALANCE", "BAKI")
TOTAL_HINTS = ("TOTAL", "JUMLAH")
BF_HINTS = (
    "B/F",
    "BALANCE B/F",
    "BAKI B/F",
    "BAKI MULA",
    "OPENING",
    "BALANCE BROUGHT",
    "BALANCE BROUGHT FORWARD",
)

NON_TX_HINTS = (
    "ACCOUNT",
    "NO.",
    "STATEMENT",
    "PENYATA",
    "PAGE",
    "PIDM",
    "AFFIN",
    "BRANCH",
    "ADDRESS",
    "CUSTOMER",
    "CIF",
    "PERIOD",
    "TARIKH PENYATA",
    "STATEMENT DATE",
)


# =========================================================
# Helpers
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _to_iso_date(token: str) -> Optional[str]:
    """
    Accept:
      - 01/05/25
      - 1/5/2025
      - OCR-attached pipes like '01/05/25|' or '|01/05/25'
    """
    if not token:
        return None

    m = DATE_IN_TOKEN_RE.search(token)
    if not m:
        return None

    d = int(m.group("d"))
    mo = int(m.group("m"))
    y = int(m.group("y"))
    if y < 100:
        y += 2000

    try:
        return datetime(y, mo, d).strftime("%Y-%m-%d")
    except Exception:
        return None


def _clean_money_token(token: str) -> Optional[str]:
    if token is None:
        return None
    t = str(token).strip()
    if not t:
        return None

    # common OCR noise
    t = t.replace("O", "0").replace("o", "0")
    t = t.replace(" ", "")

    m = MONEY_TOKEN_RE.match(t)
    if not m:
        # Try stripping obvious punctuation and retry
        t2 = t.strip(".,;:|")
        m = MONEY_TOKEN_RE.match(t2)
        if not m:
            return None
        t = t2

    num = m.group("num") or ""
    if num.startswith("."):
        num = "0" + num

    # normalize commas
    num = num.replace(",", "")

    # parentheses negative (rare in Affin, but safe)
    is_paren_neg = t.startswith("(") and ")" in t

    sign = m.group("trail_sign")
    if is_paren_neg or sign == "-":
        return "-" + num
    return num


def _money_to_float(token: str) -> Optional[float]:
    s = _clean_money_token(token)
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _is_money_token(token: str) -> bool:
    return _clean_money_token(token) is not None


def _is_summary_row(row_upper: str) -> bool:
    return any(t in row_upper for t in TOTAL_HINTS)


def _is_bf_row(row_upper: str) -> bool:
    return any(t in row_upper for t in BF_HINTS)


def _looks_non_tx_row(row_upper: str) -> bool:
    return any(t in row_upper for t in NON_TX_HINTS)


# =========================================================
# Word extraction (PDF text first, OCR fallback)
# =========================================================
def _words_from_pdf(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    words = page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
        extra_attrs=["x0", "x1", "top", "bottom"],
    ) or []

    out: List[Dict[str, Any]] = []
    for w in words:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        out.append(
            {
                "text": t,
                "x0": float(w.get("x0", 0.0)),
                "x1": float(w.get("x1", 0.0)),
                "y0": float(w.get("top", 0.0)),
                "y1": float(w.get("bottom", 0.0)),
            }
        )
    return out


def _words_from_ocr(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    if pytesseract is None:
        return []

    # Crop away common header/footer to reduce OCR noise (Affin layout)
    try:
        w, h = float(page.width), float(page.height)
        top = 120
        bottom = h - 60
        crop = page.crop((0, top, w, bottom))
        img = crop.to_image(resolution=200).original
    except Exception:
        img = page.to_image(resolution=200).original

    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT,
        config="--psm 6",
    )

    n = len(data.get("text", []))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        x, y, ww, hh = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        out.append(
            {
                "text": txt,
                "x0": float(x),
                "x1": float(x + ww),
                "y0": float(y),
                "y1": float(y + hh),
            }
        )
    return out


def _get_page_words(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    w = _words_from_pdf(page)
    if w:
        return w
    return _words_from_ocr(page)


# =========================================================
# Row grouping
# =========================================================
def _cluster_rows(
    words: List[Dict[str, Any]], y_tol: float = 3.5
) -> List[Tuple[float, List[Dict[str, Any]]]]:
    """
    Returns list of (row_y, row_words) sorted top->bottom.
    """
    if not words:
        return []

    words.sort(key=lambda r: (r["y0"], r["x0"]))

    buckets: List[Dict[str, Any]] = []
    for w in words:
        placed = False
        for b in buckets:
            if abs(w["y0"] - b["y"]) <= y_tol:
                b["items"].append(w)
                b["y"] = (b["y"] * (len(b["items"]) - 1) + w["y0"]) / len(b["items"])
                placed = True
                break
        if not placed:
            buckets.append({"y": w["y0"], "items": [w]})

    out: List[Tuple[float, List[Dict[str, Any]]]] = []
    for b in sorted(buckets, key=lambda z: z["y"]):
        row_words = sorted(b["items"], key=lambda z: z["x0"])
        out.append((float(b["y"]), row_words))
    return out


def _row_text(row_words: List[Dict[str, Any]]) -> str:
    return _norm(" ".join(w["text"] for w in row_words))


def _row_has_date(row_words: List[Dict[str, Any]]) -> bool:
    for w in row_words[:8]:
        if _to_iso_date(w["text"]):
            return True
    return False


# =========================================================
# Column detection
# =========================================================
def _detect_columns(
    rows: List[Tuple[float, List[Dict[str, Any]]]]
) -> Optional[Dict[str, float]]:
    """
    Try to find header row containing Debit/Credit/Balance.
    OCR may distort 'Balance' so we look for 'BAL'/'BAKI' fragments too.
    """
    # 1) Header scan
    for _, rw in rows[:50]:
        txtu = _row_text(rw).upper()
        if not any(h in txtu for h in HEADER_HINTS):
            continue

        debit_x = credit_x = balance_x = None
        for w in rw:
            t = w["text"].upper()
            xc = (w["x0"] + w["x1"]) / 2.0
            if debit_x is None and ("DEBIT" in t or t == "DR"):
                debit_x = xc
            if credit_x is None and ("CREDIT" in t or t == "CR"):
                credit_x = xc
            if balance_x is None and ("BAL" in t or "BAKI" in t):
                balance_x = xc

        if debit_x and credit_x and balance_x:
            return {
                "debit_x": float(debit_x),
                "credit_x": float(credit_x),
                "balance_x": float(balance_x),
            }

    # 2) Fallback clustering: use only rows that have a date
    money_xs: List[float] = []
    for _, rw in rows:
        txtu = _row_text(rw).upper()
        if not _row_has_date(rw):
            continue
        if _is_summary_row(txtu):
            continue
        if _looks_non_tx_row(txtu):
            continue

        for w in rw:
            if _is_money_token(w["text"]):
                xc = (w["x0"] + w["x1"]) / 2.0
                money_xs.append(float(xc))

    if len(money_xs) < 10:
        return None

    money_xs.sort()

    clusters: List[List[float]] = []
    for x in money_xs:
        if not clusters or abs(x - clusters[-1][-1]) > 35:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    clusters.sort(key=lambda c: sum(c) / len(c))
    if not clusters:
        return None

    balance_x = sum(clusters[-1]) / len(clusters[-1])

    debit_x = credit_x = -1.0
    if len(clusters) >= 3:
        debit_x = sum(clusters[-3]) / len(clusters[-3])
        credit_x = sum(clusters[-2]) / len(clusters[-2])

    return {"debit_x": float(debit_x), "credit_x": float(credit_x), "balance_x": float(balance_x)}


def _pick_money_near(
    row_words: List[Dict[str, Any]], target_x: float, tol: float
) -> Optional[float]:
    if target_x <= 0:
        return None

    best: Optional[Tuple[float, float]] = None
    for w in row_words:
        if not _is_money_token(w["text"]):
            continue
        xc = (w["x0"] + w["x1"]) / 2.0
        dist = abs(xc - target_x)
        if dist > tol:
            continue
        val = _money_to_float(w["text"])
        if val is None:
            continue
        if best is None or dist < best[0]:
            best = (dist, float(val))
    return best[1] if best else None


def _rightmost_money(row_words: List[Dict[str, Any]]) -> Optional[float]:
    best: Optional[Tuple[float, float]] = None  # (x1, val)
    for w in row_words:
        if not _is_money_token(w["text"]):
            continue
        val = _money_to_float(w["text"])
        if val is None:
            continue
        x1 = float(w["x1"])
        if best is None or x1 > best[0]:
            best = (x1, float(val))
    return best[1] if best else None


# =========================================================
# Opening balance (header bootstrap)
# =========================================================
def _extract_opening_balance_from_pages(pdf: pdfplumber.PDF) -> Optional[float]:
    """
    If OCR can see the header amount 'BALANCE BROUGHT FORWARD', use it as a safer anchor
    than a potentially OCR-corrupted B/F row balance.
    """
    if pytesseract is None:
        return None

    for p in pdf.pages[:2]:
        try:
            img = p.to_image(resolution=180).original
        except Exception:
            continue

        text = pytesseract.image_to_string(img, config="--psm 6") or ""
        up = text.upper()
        if "BALANCE BROUGHT" not in up and "BAKI" not in up:
            continue

        # take the largest-looking money number near the header section
        candidates = re.findall(r"\d[\d,]*\.\d{2}", text)
        vals = []
        for c in candidates:
            try:
                vals.append(float(c.replace(",", "")))
            except Exception:
                pass
        if vals:
            # heuristically, the header balance is often one of the larger values on page
            return float(sorted(vals)[-1])

    return None


# =========================================================
# Public entry point (used by app.py)
# =========================================================
def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    """
    Works with app.py which passes an already-open pdfplumber.PDF object:
      parse_affin_bank(pdf, filename)

    Also supports standalone usage with bytes/path/file-like.
    """
    bank_name = "Affin Bank"
    txs: List[Dict[str, Any]] = []

    # If app.py already opened it
    if hasattr(pdf_input, "pages") and hasattr(pdf_input, "close"):
        pdf = pdf_input
        should_close = False
    else:
        should_close = True
        try:
            if isinstance(pdf_input, (bytes, bytearray)):
                pdf = pdfplumber.open(BytesIO(bytes(pdf_input)))
            elif hasattr(pdf_input, "getvalue"):
                pdf = pdfplumber.open(BytesIO(pdf_input.getvalue()))
            else:
                pdf = pdfplumber.open(pdf_input)
        except Exception as e:
            raise RuntimeError(f"Affin parser: cannot open PDF: {e}") from e

    prev_balance: Optional[float] = None
    seq = 0

    try:
        # bootstrap opening from header if possible
        opening = _extract_opening_balance_from_pages(pdf)
        if opening is not None:
            prev_balance = float(opening)

        for page_num, page in enumerate(pdf.pages, start=1):
            words = _get_page_words(page)
            if not words:
                continue

            rows = _cluster_rows(words, y_tol=3.5)
            if not rows:
                continue

            col = _detect_columns(rows)

            for row_y, row_words in rows:
                txt = _row_text(row_words)
                if not txt:
                    continue
                up = txt.upper()

                if _is_summary_row(up):
                    continue

                # Skip obvious non-tx lines unless they actually contain a date
                if _looks_non_tx_row(up) and not _row_has_date(row_words):
                    continue

                # Identify date token
                date_iso = None
                date_idx = None
                for i, w in enumerate(row_words[:8]):
                    d = _to_iso_date(w["text"])
                    if d:
                        date_iso = d
                        date_idx = i
                        break
                if not date_iso:
                    continue

                # Extract balance/debit/credit
                balance = None
                debit = None
                credit = None

                if col:
                    balance = _pick_money_near(row_words, col["balance_x"], tol=70.0)
                    if col["debit_x"] > 0:
                        debit = _pick_money_near(row_words, col["debit_x"], tol=55.0)
                    if col["credit_x"] > 0:
                        credit = _pick_money_near(row_words, col["credit_x"], tol=55.0)

                if balance is None:
                    balance = _rightmost_money(row_words)

                if balance is None:
                    continue

                # B/F rows are anchors, not transactions
                if _is_bf_row(up):
                    prev_balance = float(balance)
                    continue

                # Description: all non-money after date token
                desc_parts: List[str] = []
                for j, w in enumerate(row_words):
                    if date_idx is not None and j <= date_idx:
                        continue
                    t = (w["text"] or "").strip().strip("|")
                    if not t:
                        continue
                    if _is_money_token(t):
                        continue
                    desc_parts.append(t)
                description = _norm(" ".join(desc_parts))

                debit_f = float(debit or 0.0)
                credit_f = float(credit or 0.0)

                # If both debit and credit are missing (common), infer from balance delta
                if prev_balance is not None and debit is None and credit is None:
                    delta = round(float(balance) - float(prev_balance), 2)
                    if delta > 0:
                        credit_f = delta
                        debit_f = 0.0
                    elif delta < 0:
                        debit_f = abs(delta)
                        credit_f = 0.0

                # Reconcile and repair balance if OCR produced a digit-glitch
                if prev_balance is not None:
                    expected = round(float(prev_balance) + float(credit_f) - float(debit_f), 2)

                    # If OCR balance does not match expected, trust expected when debit/credit are present.
                    # This repairs common OCR digit glitches in the Balance column.
                    if abs(round(float(balance), 2) - expected) > 0.05:
                        # Only apply repair if the implied movement is reasonable.
                        if abs(expected - prev_balance) <= 5_000_000:
                            balance = expected

                seq += 1
                txs.append(
                    {
                        "date": date_iso,
                        "description": description,
                        "debit": round(float(debit_f), 2),
                        "credit": round(float(credit_f), 2),
                        "balance": round(float(balance), 2),
                        "page": int(page_num),
                        "bank": bank_name,
                        "source_file": source_file or "",
                        "_y": float(row_y),
                        "_seq": int(seq),
                    }
                )

                prev_balance = float(balance)

    finally:
        if should_close:
            try:
                pdf.close()
            except Exception:
                pass

    # Stable ordering (prevents month summary drift if PDF emits rows slightly out of order)
    txs.sort(
        key=lambda x: (
            x.get("date", ""),
            int(x.get("page", 0)),
            float(x.get("_y", 0.0)),
            int(x.get("_seq", 0)),
        )
    )

    for t in txs:
        t.pop("_y", None)
        t.pop("_seq", None)

    return txs
