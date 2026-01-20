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

MAX_REASONABLE_DELTA = 5_000_000.00

# For totals extraction
MONEY_IN_TEXT_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}")

TOTAL_DEBIT_PATTERNS = [
    re.compile(r"(TOTAL|JUMLAH)\s+DEBIT[^0-9]*(" + MONEY_IN_TEXT_RE.pattern + r")", re.I),
    re.compile(r"DEBIT\s+(TOTAL|JUMLAH)[^0-9]*(" + MONEY_IN_TEXT_RE.pattern + r")", re.I),
]
TOTAL_CREDIT_PATTERNS = [
    re.compile(r"(TOTAL|JUMLAH)\s+CREDIT[^0-9]*(" + MONEY_IN_TEXT_RE.pattern + r")", re.I),
    re.compile(r"CREDIT\s+(TOTAL|JUMLAH)[^0-9]*(" + MONEY_IN_TEXT_RE.pattern + r")", re.I),
]
OPENING_PATTERNS = [
    re.compile(r"(BALANCE|BAKI)\s*(B/F|BROUGHT\s+FORWARD|MULA)[^0-9]*(" + MONEY_IN_TEXT_RE.pattern + r")", re.I),
    re.compile(r"\bB/F\b[^0-9]*(" + MONEY_IN_TEXT_RE.pattern + r")", re.I),
]
CLOSING_PATTERNS = [
    re.compile(r"(BALANCE|BAKI)\s*(C/F|CARRIED\s+FORWARD|AKHIR)[^0-9]*(" + MONEY_IN_TEXT_RE.pattern + r")", re.I),
    re.compile(r"\bC/F\b[^0-9]*(" + MONEY_IN_TEXT_RE.pattern + r")", re.I),
    re.compile(r"(CLOSING|ENDING)\s+BALANCE[^0-9]*(" + MONEY_IN_TEXT_RE.pattern + r")", re.I),
]

FILENAME_MONTH_RE = re.compile(r"(?P<y>20\d{2})[^\d]?(?P<m>0[1-9]|1[0-2])")


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

    # common OCR noise
    t = t.replace("O", "0").replace("o", "0")
    t = t.replace(" ", "")

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

    # Crop header/footer a bit to reduce noise (tune if needed)
    try:
        w, h = float(page.width), float(page.height)
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

    # fallback: cluster right-side money tokens from date rows
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

    balance_x = sum(clusters[-1]) / len(clusters[-1])
    credit_x = sum(clusters[-2]) / len(clusters[-2])
    debit_x = sum(clusters[-3]) / len(clusters[-3]) if len(clusters) >= 3 else -1.0
    return {"debit_x": float(debit_x), "credit_x": float(credit_x), "balance_x": float(balance_x)}


def _classify_money_by_columns(
    row_words: List[Dict[str, Any]],
    col: Optional[Dict[str, float]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    money_items: List[Tuple[float, float]] = []
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

    if not col:
        money_items.sort(key=lambda t: t[0])
        bal = money_items[-1][1]
        return None, None, bal

    debit_x = float(col.get("debit_x", -1.0))
    credit_x = float(col.get("credit_x", -1.0))
    balance_x = float(col.get("balance_x", -1.0))

    debit_vals: List[float] = []
    credit_vals: List[float] = []
    balance_vals: List[float] = []

    for xc, v in money_items:
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
    balance = float(balance_vals[-1]) if balance_vals else float(sorted(money_items, key=lambda t: t[0])[-1][1])

    return debit, credit, balance


# =========================================================
# PUBLIC: Line-item transaction parser (PDF text + OCR fallback)
# =========================================================
def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    bank_name = "Affin Bank"
    txs: List[Dict[str, Any]] = []

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
                if _is_summary_row(up):
                    i += 1
                    continue
                if _looks_non_tx_row(up) and not _row_has_date(row_words):
                    i += 1
                    continue

                # find date
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

                # merge continuation rows until next date
                block_words = list(row_words)
                k = i + 1
                while k < len(rows) and not _row_has_date(rows[k][1]):
                    next_up = _row_text(rows[k][1]).upper()
                    if _is_summary_row(next_up):
                        break
                    block_words.extend(rows[k][1])
                    k += 1

                block_words.sort(key=lambda z: (z["y0"], z["x0"]))

                debit, credit, balance = _classify_money_by_columns(block_words, col)
                if balance is None:
                    i = k
                    continue

                # B/F anchor
                if _is_bf_row(up):
                    prev_balance = float(balance)
                    i = k
                    continue

                # description = text after date excluding money
                desc_parts: List[str] = []
                for idx, w in enumerate(block_words):
                    t = (w.get("text") or "").strip().strip("|")
                    if not t:
                        continue
                    if _is_money_token(t):
                        continue
                    if idx <= (date_idx or 0) and w in row_words:
                        continue
                    tu = t.upper()
                    if tu in ("DATE", "TARIKH", "DEBIT", "CREDIT", "BALANCE", "BAKI"):
                        continue
                    desc_parts.append(t)
                description = _norm(" ".join(desc_parts))

                debit_f = float(debit) if debit is not None else 0.0
                credit_f = float(credit) if credit is not None else 0.0

                # delta inference only if debit+credit missing
                if prev_balance is not None and debit is None and credit is None:
                    delta = round(float(balance) - float(prev_balance), 2)
                    if abs(delta) <= MAX_REASONABLE_DELTA:
                        if delta > 0:
                            credit_f = delta
                            debit_f = 0.0
                        elif delta < 0:
                            debit_f = abs(delta)
                            credit_f = 0.0

                # if both filled (suspicious), prefer delta if consistent
                if prev_balance is not None and debit_f > 0 and credit_f > 0:
                    delta = round(float(balance) - float(prev_balance), 2)
                    if abs(delta) <= MAX_REASONABLE_DELTA:
                        if delta > 0:
                            credit_f = abs(delta)
                            debit_f = 0.0
                        elif delta < 0:
                            debit_f = abs(delta)
                            credit_f = 0.0

                # dedupe (OCR-safe): do NOT include description/page/source_file
                key = (date_iso, round(debit_f, 2), round(credit_f, 2), round(float(balance), 2), bank_name)
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


# =========================================================
# PUBLIC: Statement totals extractor (ground truth for monthly totals)
# =========================================================
def _money_to_float_from_text(s: str) -> Optional[float]:
    if not s:
        return None
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def _infer_month_from_filename(filename: str) -> Optional[str]:
    if not filename:
        return None
    m = FILENAME_MONTH_RE.search(filename)
    if not m:
        return None
    y = int(m.group("y"))
    mo = int(m.group("m"))
    return f"{y:04d}-{mo:02d}"


def _page_text_pdf_or_ocr(page: pdfplumber.page.Page) -> str:
    txt = (page.extract_text() or "").strip()
    if len(txt) >= 200:
        return txt

    if pytesseract is None:
        return txt

    try:
        img = page.to_image(resolution=220).original
        ocr = pytesseract.image_to_string(img, config="--psm 6") or ""
        return ocr
    except Exception:
        return txt


def _find_first_amount(text: str, patterns: List[re.Pattern]) -> Optional[float]:
    if not text:
        return None
    for pat in patterns:
        m = pat.search(text)
        if not m:
            continue
        amt = m.group(m.lastindex) if m.lastindex else None
        f = _money_to_float_from_text(amt or "")
        if f is not None:
            return float(f)
    return None


def extract_affin_statement_totals(pdf_input: Any, source_file: str = "") -> Dict[str, Any]:
    """
    Extract Affin statement ground-truth totals:
      opening_balance, total_debit, total_credit, ending_balance, statement_month
    Scans first 2 + last 2 pages with PDF text, OCR fallback.
    """
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
            raise RuntimeError(f"Affin totals: cannot open PDF: {e}") from e

    try:
        n = len(pdf.pages)
        idxs: List[int] = []
        for i in [0, 1, max(0, n - 2), max(0, n - 1)]:
            if 0 <= i < n and i not in idxs:
                idxs.append(i)

        blob = ""
        for i in idxs:
            blob += "\n" + (_page_text_pdf_or_ocr(pdf.pages[i]) or "")
        blob = _norm(blob.replace("\x0c", " "))

        total_debit = _find_first_amount(blob, TOTAL_DEBIT_PATTERNS)
        total_credit = _find_first_amount(blob, TOTAL_CREDIT_PATTERNS)
        opening_balance = _find_first_amount(blob, OPENING_PATTERNS)
        ending_balance = _find_first_amount(blob, CLOSING_PATTERNS)

        # mild fallback: if B/F present but regex missed, take nearby number
        if opening_balance is None:
            m = re.search(r"(BALANCE|BAKI).*?\bB/F\b", blob, re.I)
            if m:
                tail = blob[m.start(): m.start() + 250]
                nums = MONEY_IN_TEXT_RE.findall(tail)
                if nums:
                    opening_balance = _money_to_float_from_text(nums[-1])

        if ending_balance is None:
            m = re.search(r"(BALANCE|BAKI).*?\bC/F\b", blob, re.I)
            if m:
                tail = blob[m.start(): m.start() + 250]
                nums = MONEY_IN_TEXT_RE.findall(tail)
                if nums:
                    ending_balance = _money_to_float_from_text(nums[-1])

        return {
            "bank": "Affin Bank",
            "source_file": source_file or "",
            "statement_month": _infer_month_from_filename(source_file),
            "opening_balance": opening_balance,
            "total_debit": total_debit,
            "total_credit": total_credit,
            "ending_balance": ending_balance,
        }

    finally:
        if should_close:
            try:
                pdf.close()
            except Exception:
                pass
