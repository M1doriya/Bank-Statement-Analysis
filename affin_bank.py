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

DATE_IN_TOKEN_RE = re.compile(r"(?P<d>\d{1,2})\s*/\s*(?P<m>\d{1,2})\s*/\s*(?P<y>\d{2,4})")

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

# sanity guard: if a single row delta is absurd, treat as OCR/path error
MAX_REASONABLE_DELTA = 5_000_000.00


# =========================================================
# Helpers
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _to_iso_date(token: str) -> Optional[str]:
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

    # OCR noise
    t = t.replace("O", "0").replace("o", "0").replace(" ", "")

    m = MONEY_TOKEN_RE.match(t)
    if not m:
        t2 = t.strip(".,;:|")
        m = MONEY_TOKEN_RE.match(t2)
        if not m:
            return None
        t = t2

    num = (m.group("num") or "")
    if num.startswith("."):
        num = "0" + num
    num = num.replace(",", "")

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

    try:
        w, h = float(page.width), float(page.height)
        # slightly wider crop than before; Affin totals/header vary by month
        top = 90
        bottom = h - 50
        crop = page.crop((0, top, w, bottom))
        img = crop.to_image(resolution=220).original
    except Exception:
        img = page.to_image(resolution=220).original

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
def _cluster_rows(words: List[Dict[str, Any]], y_tol: float = 2.5) -> List[Tuple[float, List[Dict[str, Any]]]]:
    """
    Tighter y_tol reduces 'one transaction becomes multiple rows' OCR artifacts.
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
    for w in row_words[:10]:
        if _to_iso_date(w["text"]):
            return True
    return False


# =========================================================
# Column detection
# =========================================================
def _detect_columns(rows: List[Tuple[float, List[Dict[str, Any]]]]) -> Optional[Dict[str, float]]:
    """
    Find header row and return approximate x centers for debit/credit/balance.
    """
    for _, rw in rows[:60]:
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
            return {"debit_x": float(debit_x), "credit_x": float(credit_x), "balance_x": float(balance_x)}

    # Fallback: use three right-most money clusters from date rows
    money_xs: List[float] = []
    for _, rw in rows:
        up = _row_text(rw).upper()
        if not _row_has_date(rw):
            continue
        if _is_summary_row(up) or _looks_non_tx_row(up):
            continue
        for w in rw:
            if _is_money_token(w["text"]):
                money_xs.append(float((w["x0"] + w["x1"]) / 2.0))

    if len(money_xs) < 15:
        return None

    money_xs.sort()
    clusters: List[List[float]] = []
    for x in money_xs:
        if not clusters or abs(x - clusters[-1][-1]) > 30:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    clusters.sort(key=lambda c: sum(c) / len(c))
    if len(clusters) < 2:
        return None

    # pick right-most 3 clusters as [debit, credit, balance] if possible
    balance_x = sum(clusters[-1]) / len(clusters[-1])
    credit_x = sum(clusters[-2]) / len(clusters[-2])
    debit_x = sum(clusters[-3]) / len(clusters[-3]) if len(clusters) >= 3 else -1.0

    return {"debit_x": float(debit_x), "credit_x": float(credit_x), "balance_x": float(balance_x)}


def _classify_money_by_columns(
    row_words: List[Dict[str, Any]],
    col: Optional[Dict[str, float]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return (debit, credit, balance) from a row based on nearest column centers.
    More robust than picking one token near a target.
    """
    money_items: List[Tuple[float, float]] = []  # (xc, val)
    for w in row_words:
        if not _is_money_token(w["text"]):
            continue
        v = _money_to_float(w["text"])
        if v is None:
            continue
        xc = float((w["x0"] + w["x1"]) / 2.0)
        money_items.append((xc, float(v)))

    if not money_items:
        return None, None, None

    # If no columns, use heuristic: right-most is balance, remaining left-most are debit/credit unknown
    if not col:
        money_items.sort(key=lambda t: t[0])
        bal = money_items[-1][1]
        return None, None, bal

    debit_x = float(col.get("debit_x", -1.0))
    credit_x = float(col.get("credit_x", -1.0))
    balance_x = float(col.get("balance_x", -1.0))

    # Assign each token to nearest of (debit, credit, balance)
    debit_vals: List[float] = []
    credit_vals: List[float] = []
    balance_vals: List[float] = []

    for xc, v in money_items:
        # ignore obvious left-side amounts (narrative area) by requiring proximity to right columns
        # (Affin numeric columns are on the right)
        if balance_x > 0 and xc < (balance_x - 250):
            continue

        dists = []
        if debit_x > 0:
            dists.append(("debit", abs(xc - debit_x)))
        if credit_x > 0:
            dists.append(("credit", abs(xc - credit_x)))
        if balance_x > 0:
            dists.append(("balance", abs(xc - balance_x)))

        if not dists:
            continue

        label, dist = min(dists, key=lambda x: x[1])
        # tolerance gate: if too far from all columns, skip
        if dist > 85:
            continue

        if label == "debit":
            debit_vals.append(abs(v))
        elif label == "credit":
            credit_vals.append(abs(v))
        else:
            balance_vals.append(v)

    debit = round(sum(debit_vals), 2) if debit_vals else None
    credit = round(sum(credit_vals), 2) if credit_vals else None

    # balance: take the right-most/most plausible (many rows contain only one balance token)
    balance = None
    if balance_vals:
        # prefer the last seen balance token (often only one)
        balance = float(balance_vals[-1])
    else:
        # fallback: right-most money token
        money_items.sort(key=lambda t: t[0])
        balance = float(money_items[-1][1])

    return debit, credit, balance


# =========================================================
# Public entry point (used by app.py)
# =========================================================
def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    bank_name = "Affin Bank"
    txs: List[Dict[str, Any]] = []

    # app.py passes an open pdfplumber.PDF
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
    seen = set()

    try:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = _get_page_words(page)
            if not words:
                continue

            rows = _cluster_rows(words, y_tol=2.5)
            if not rows:
                continue

            col = _detect_columns(rows)

            i = 0
            while i < len(rows):
                row_y, row_words = rows[i]
                txt = _row_text(row_words)
                if not txt:
                    i += 1
                    continue

                up = txt.upper()

                # Skip summary/total blocks
                if _is_summary_row(up):
                    i += 1
                    continue

                # Find date in this row
                date_iso = None
                date_idx = None
                for j, w in enumerate(row_words[:10]):
                    d = _to_iso_date(w["text"])
                    if d:
                        date_iso = d
                        date_idx = j
                        break

                if not date_iso:
                    i += 1
                    continue

                # Merge continuation rows until next date (prevents split-row double counting)
                block_words = list(row_words)
                k = i + 1
                while k < len(rows) and not _row_has_date(rows[k][1]):
                    next_txtu = _row_text(rows[k][1]).upper()
                    if _is_summary_row(next_txtu):
                        break
                    # append continuation words
                    block_words.extend(rows[k][1])
                    k += 1

                # Re-sort merged block by x
                block_words.sort(key=lambda z: (z["y0"], z["x0"]))

                # Classify debit/credit/balance from merged block
                debit, credit, balance = _classify_money_by_columns(block_words, col)

                if balance is None:
                    i = k
                    continue

                # B/F row is an anchor (not a transaction)
                if _is_bf_row(up):
                    prev_balance = float(balance)
                    i = k
                    continue

                # Description: everything after date token excluding money tokens
                desc_parts: List[str] = []
                for idx, w in enumerate(block_words):
                    t = (w.get("text") or "").strip().strip("|")
                    if not t:
                        continue
                    if _is_money_token(t):
                        continue
                    # best-effort: ignore tokens before/at date token on first row
                    if idx <= (date_idx or 0) and w in row_words:
                        continue
                    # ignore obvious headers
                    tu = t.upper()
                    if tu in ("DATE", "TARIKH", "DEBIT", "CREDIT", "BALANCE", "BAKI"):
                        continue
                    desc_parts.append(t)

                description = _norm(" ".join(desc_parts))

                # If debit/credit absent, infer from balance delta with sanity check
                debit_f = float(debit) if debit is not None else 0.0
                credit_f = float(credit) if credit is not None else 0.0

                if prev_balance is not None and (debit is None and credit is None):
                    delta = round(float(balance) - float(prev_balance), 2)
                    if abs(delta) <= MAX_REASONABLE_DELTA:
                        if delta > 0:
                            credit_f = delta
                            debit_f = 0.0
                        elif delta < 0:
                            debit_f = abs(delta)
                            credit_f = 0.0

                # If both debit and credit exist but look suspicious (both > 0 on a single row),
                # prefer delta if it is small and consistent.
                if prev_balance is not None and debit_f > 0 and credit_f > 0:
                    delta = round(float(balance) - float(prev_balance), 2)
                    if abs(delta) <= MAX_REASONABLE_DELTA:
                        if delta > 0:
                            credit_f = abs(delta)
                            debit_f = 0.0
                        elif delta < 0:
                            debit_f = abs(delta)
                            credit_f = 0.0

                # Dedupe: (date, rounded amounts, rounded balance, page, short desc)
                key = (
                    date_iso,
                    round(debit_f, 2),
                    round(credit_f, 2),
                    round(float(balance), 2),
                    int(page_num),
                    description[:60],
                )
                if key in seen:
                    i = k
                    continue
                seen.add(key)

                txs.append(
                    {
                        "date": date_iso,
                        "description": description,
                        "debit": round(debit_f, 2),
                        "credit": round(credit_f, 2),
                        "balance": round(float(balance), 2),
                        "page": int(page_num),
                        "bank": bank_name,
                        "source_file": source_file or "",
                    }
                )

                prev_balance = float(balance)
                i = k

    finally:
        if should_close:
            try:
                pdf.close()
            except Exception:
                pass

    # stable ordering
    txs.sort(key=lambda t: (t.get("date", ""), int(t.get("page", 0)), t.get("description", "")))
    return txs
