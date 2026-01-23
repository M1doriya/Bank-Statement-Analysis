# bank_rakyat.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None


# =========================================================
# Regex
# =========================================================

DATE_RANGE_RE = re.compile(
    r"(?:DARI/FROM)\s+(?P<d1>\d{2}/\d{2}/\d{4})\s+(?:SEHINGGA/UNTIL)\s+(?P<d2>\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)

# Money token like 9,450.66 or -116.02
MONEY_ANY_RE = re.compile(r"-?(?<!\d)(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}(?!\d)")
MONEY_STRICT_RE = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}$")

DATE_DMY_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")

# Transaction line (OCR) starts with DD/MM/YYYY
TX_LINE_RE = re.compile(r"^(?P<date>\d{2}/\d{2}/\d{4})\s+(?P<rest>.*)$")

# Month from filename fallback
FNAME_MONTH_RE = re.compile(
    r"\b(?P<mon>jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b.*?\b(?P<year>20\d{2})\b",
    re.IGNORECASE,
)

MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Scanned vs digital threshold
MIN_TEXT_CHARS = 40


# =========================================================
# Helpers
# =========================================================

def _safe_float_money(s: Any) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s2 = s.replace(",", "")
    if not MONEY_STRICT_RE.fullmatch(s2):
        return None
    try:
        return float(s2)
    except Exception:
        return None


def _to_iso_date_ddmmyyyy(dmy: str) -> Optional[str]:
    try:
        dt = datetime.strptime(dmy, "%d/%m/%Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _extract_statement_month_from_text(text: str) -> Optional[str]:
    m = DATE_RANGE_RE.search(text or "")
    if not m:
        return None
    try:
        end_dt = datetime.strptime(m.group("d2"), "%d/%m/%Y")
        return end_dt.strftime("%Y-%m")
    except Exception:
        return None


def _extract_statement_month_from_filename(filename: str) -> Optional[str]:
    if not filename:
        return None
    m = FNAME_MONTH_RE.search(filename.replace("_", " ").replace("-", " "))
    if not m:
        return None
    mon_s = (m.group("mon") or "").lower()
    year_s = m.group("year")
    mm = MONTH_MAP.get(mon_s)
    if not mm:
        return None
    try:
        yyyy = int(year_s)
        return f"{yyyy:04d}-{mm:02d}"
    except Exception:
        return None


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _strip_money_tokens(s: str) -> str:
    s = MONEY_ANY_RE.sub(" ", (s or ""))
    return _normalize_ws(s)


def _ocr_image_crop(
    page: pdfplumber.page.Page,
    *,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
    dpi: int = 250,
):
    w = float(page.width)
    h = float(page.height)
    left = w * left_ratio
    right = w * right_ratio
    top = h * top_ratio
    bottom = h * bottom_ratio
    crop = page.crop((left, top, right, bottom))
    try:
        return crop.to_image(resolution=dpi).original
    except Exception:
        return page.to_image(resolution=dpi).original


# =========================================================
# Statement totals (fast + scanned-safe)
# =========================================================

def extract_bank_rakyat_statement_totals(
    pdf: pdfplumber.PDF,
    source_file: str = "",
    *,
    compute_tx_count: bool = False,  # IMPORTANT: default False to avoid heavy OCR
) -> Dict[str, Optional[float]]:
    """
    Extract:
      - statement_month
      - opening_balance
      - total_debit
      - total_credit
      - ending_balance
      - transaction_count (optional / bounded)

    Strategy:
      1) Try embedded text on page 1 / last page (fast).
      2) If scanned, OCR only top band (month) + bottom band (totals).
      3) For totals, choose best 4 consecutive values satisfying:
           opening + credit - debit ≈ ending
         also tries swapped debit/credit.
    """
    out: Dict[str, Optional[float]] = {
        "statement_month": None,
        "opening_balance": None,
        "total_debit": None,
        "total_credit": None,
        "ending_balance": None,
        "transaction_count": None,
        "source_file": source_file,
    }

    if not pdf.pages:
        out["statement_month"] = _extract_statement_month_from_filename(source_file)
        return out

    candidate_pages = [pdf.pages[0]]
    if len(pdf.pages) > 1:
        candidate_pages.append(pdf.pages[-1])

    # 1) DIGITAL (embedded text) fast path
    for page in candidate_pages:
        text = page.extract_text(x_tolerance=1) or ""
        if len(text.strip()) >= MIN_TEXT_CHARS:
            out["statement_month"] = (
                _extract_statement_month_from_text(text)
                or _extract_statement_month_from_filename(source_file)
            )

            vals = [_safe_float_money(x) for x in MONEY_ANY_RE.findall(text)]
            vals = [v for v in vals if v is not None]

            if len(vals) >= 4:
                best = (None, None, None, None)
                best_score = float("inf")
                for i in range(0, len(vals) - 3):
                    a, b, c, d = vals[i], vals[i + 1], vals[i + 2], vals[i + 3]
                    s1 = abs((a + c - b) - d)  # opening + credit - debit
                    s2 = abs((a + b - c) - d)  # opening + debit? swapped
                    if s1 < best_score:
                        best_score = s1
                        best = (a, b, c, d)
                    if s2 < best_score:
                        best_score = s2
                        best = (a, c, b, d)  # normalize to (opening, debit, credit, ending)
                out["opening_balance"], out["total_debit"], out["total_credit"], out["ending_balance"] = best

            return out

    # 2) SCANNED (OCR limited) path
    if pytesseract is None:
        out["statement_month"] = _extract_statement_month_from_filename(source_file)
        return out

    # 2a) Month via OCR top band (quick)
    month = None
    for page in candidate_pages:
        top_img = _ocr_image_crop(page, left_ratio=0.0, top_ratio=0.0, right_ratio=1.0, bottom_ratio=0.25, dpi=200)
        try:
            top_txt = pytesseract.image_to_string(top_img, config="--psm 6") or ""
        except Exception:
            top_txt = ""
        month = _extract_statement_month_from_text(top_txt)
        if month:
            break
    out["statement_month"] = month or _extract_statement_month_from_filename(source_file)

    # 2b) Totals via OCR bottom band (quick + robust)
    best_tuple: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]] = (None, None, None, None)
    best_score = float("inf")

    for page in candidate_pages:
        bottom_img = _ocr_image_crop(page, left_ratio=0.0, top_ratio=0.65, right_ratio=1.0, bottom_ratio=1.0, dpi=250)
        try:
            raw = pytesseract.image_to_string(bottom_img, config="--psm 6") or ""
        except Exception:
            raw = ""

        vals = [_safe_float_money(x) for x in MONEY_ANY_RE.findall(raw)]
        vals = [v for v in vals if v is not None]
        if len(vals) < 4:
            continue

        for i in range(0, len(vals) - 3):
            a, b, c, d = vals[i], vals[i + 1], vals[i + 2], vals[i + 3]
            s1 = abs((a + c - b) - d)
            s2 = abs((a + b - c) - d)
            if s1 < best_score:
                best_score = s1
                best_tuple = (a, b, c, d)
            if s2 < best_score:
                best_score = s2
                best_tuple = (a, c, b, d)

    out["opening_balance"], out["total_debit"], out["total_credit"], out["ending_balance"] = best_tuple

    # 2c) Optional transaction count (bounded; off by default)
    if compute_tx_count:
        max_pages = min(len(pdf.pages), 5)
        total = 0
        for i in range(max_pages):
            page = pdf.pages[i]
            img = _ocr_image_crop(page, left_ratio=0.02, top_ratio=0.18, right_ratio=0.30, bottom_ratio=0.88, dpi=200)
            try:
                txt = pytesseract.image_to_string(img, config="--psm 6") or ""
            except Exception:
                txt = ""
            total += len(DATE_DMY_RE.findall(txt))
        out["transaction_count"] = total if total > 0 else None

    return out


# =========================================================
# Transactions parsing
# =========================================================

def _parse_bank_rakyat_digital(pdf: pdfplumber.PDF, filename: str) -> List[Dict]:
    """
    Digital fast path:
      - Uses table extraction + line fallback
      - Looks for rows where first cell contains DD/MM/YYYY
      - Outputs normalized schema with page/seq for stable ordering
    """
    txs: List[Dict] = []
    seq = 0

    # Common table extraction settings for bank statements
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 10,
        "min_words_vertical": 1,
        "min_words_horizontal": 1,
        "keep_blank_chars": False,
        "text_x_tolerance": 2,
        "text_y_tolerance": 2,
    }

    for page_num, page in enumerate(pdf.pages, start=1):
        # Try tables first
        tables = []
        try:
            tables = page.extract_tables(table_settings=table_settings) or []
        except Exception:
            tables = []

        for tbl in tables:
            for row in tbl or []:
                if not row:
                    continue
                cells = [(_normalize_ws(c) if c is not None else "") for c in row]
                if not cells:
                    continue

                # Find date in first non-empty cell (often col 0)
                date_cell = ""
                for c in cells[:2]:  # usually date is within first 2 cols
                    if DATE_DMY_RE.fullmatch(c):
                        date_cell = c
                        break
                if not date_cell:
                    continue

                date_iso = _to_iso_date_ddmmyyyy(date_cell)
                if not date_iso:
                    continue

                # Heuristic: last 3 numeric-looking cells are debit/credit/balance OR amount/balance
                money_cells = [c for c in cells if _safe_float_money(c) is not None]
                if not money_cells:
                    continue

                balance = _safe_float_money(money_cells[-1])
                debit = 0.0
                credit = 0.0

                if len(money_cells) >= 3:
                    debit_val = _safe_float_money(money_cells[-3]) or 0.0
                    credit_val = _safe_float_money(money_cells[-2]) or 0.0
                    debit = round(abs(debit_val), 2) if abs(debit_val) > 0.0001 else 0.0
                    credit = round(abs(credit_val), 2) if abs(credit_val) > 0.0001 else 0.0
                elif len(money_cells) == 2:
                    # amount + balance: infer sign using delta later in monthly summary; here keep as 0/0
                    pass

                # Description: join non-money non-date cells
                desc_parts: List[str] = []
                for c in cells:
                    if not c:
                        continue
                    if c == date_cell:
                        continue
                    if _safe_float_money(c) is not None:
                        continue
                    desc_parts.append(c)
                desc = _normalize_ws(" ".join(desc_parts))

                if balance is None:
                    continue

                txs.append(
                    {
                        "date": date_iso,
                        "description": desc,
                        "debit": float(debit),
                        "credit": float(credit),
                        "balance": round(float(balance), 2),
                        "page": int(page_num),
                        "seq": int(seq),
                        "bank": "Bank Rakyat",
                        "source_file": filename,
                        "scanned_ocr": False,
                    }
                )
                seq += 1

        # If no tables found, try text line fallback (still digital)
        if not tables:
            text = page.extract_text(x_tolerance=1) or ""
            for line in (text.splitlines() if text else []):
                line = _normalize_ws(line)
                if not line:
                    continue
                m = TX_LINE_RE.match(line)
                if not m:
                    continue
                date_iso = _to_iso_date_ddmmyyyy(m.group("date"))
                if not date_iso:
                    continue
                vals = [_safe_float_money(x) for x in MONEY_ANY_RE.findall(line)]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue
                balance = vals[-1]
                desc = _strip_money_tokens(m.group("rest") or "")
                txs.append(
                    {
                        "date": date_iso,
                        "description": desc,
                        "debit": 0.0,
                        "credit": 0.0,
                        "balance": round(float(balance), 2),
                        "page": int(page_num),
                        "seq": int(seq),
                        "bank": "Bank Rakyat",
                        "source_file": filename,
                        "scanned_ocr": False,
                    }
                )
                seq += 1

    # stable order
    txs = sorted(txs, key=lambda t: (t.get("date") or "", int(t.get("page") or 0), int(t.get("seq") or 0)))
    return txs


def _parse_bank_rakyat_scanned_ocr(pdf: pdfplumber.PDF, filename: str) -> List[Dict]:
    """
    Scanned fallback:
      - OCR ONLY the table band per page
      - Parse lines starting with DD/MM/YYYY
      - Prefer 3 trailing numeric columns: debit, credit, balance
      - If missing, infer debit/credit by balance delta
    """
    if pytesseract is None or not pdf.pages:
        return []

    totals = extract_bank_rakyat_statement_totals(pdf, filename, compute_tx_count=False)
    prev_balance: Optional[float] = totals.get("opening_balance")

    txs: List[Dict] = []
    seq = 0

    for page_num, page in enumerate(pdf.pages, start=1):
        # Table band crop (tuned ratios; adjust if your template differs)
        img = _ocr_image_crop(
            page,
            left_ratio=0.0,
            top_ratio=0.18,
            right_ratio=1.0,
            bottom_ratio=0.88,
            dpi=250,
        )
        try:
            raw = pytesseract.image_to_string(img, config="--psm 6") or ""
        except Exception:
            raw = ""

        for raw_line in raw.splitlines():
            line = _normalize_ws(raw_line)
            if not line:
                continue
            m = TX_LINE_RE.match(line)
            if not m:
                continue

            dmy = m.group("date")
            rest = m.group("rest") or ""
            date_iso = _to_iso_date_ddmmyyyy(dmy)
            if not date_iso:
                continue

            vals = [_safe_float_money(x) for x in MONEY_ANY_RE.findall(line)]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue

            balance = float(vals[-1])
            debit = 0.0
            credit = 0.0

            if len(vals) >= 3:
                debit_col = float(vals[-3])
                credit_col = float(vals[-2])
                debit = round(abs(debit_col), 2) if abs(debit_col) > 0.0001 else 0.0
                credit = round(abs(credit_col), 2) if abs(credit_col) > 0.0001 else 0.0

                if debit == 0.0 and credit == 0.0 and prev_balance is not None:
                    delta = round(balance - prev_balance, 2)
                    if delta > 0:
                        credit = abs(delta)
                    elif delta < 0:
                        debit = abs(delta)
            else:
                if prev_balance is not None:
                    delta = round(balance - prev_balance, 2)
                    if delta > 0:
                        credit = abs(delta)
                    elif delta < 0:
                        debit = abs(delta)

            desc = _strip_money_tokens(rest)

            txs.append(
                {
                    "date": date_iso,
                    "description": desc,
                    "debit": round(float(debit), 2),
                    "credit": round(float(credit), 2),
                    "balance": round(float(balance), 2),
                    "page": int(page_num),
                    "seq": int(seq),
                    "bank": "Bank Rakyat",
                    "source_file": filename,
                    "scanned_ocr": True,
                }
            )
            seq += 1
            prev_balance = balance

    txs = sorted(txs, key=lambda t: (t.get("date") or "", int(t.get("page") or 0), int(t.get("seq") or 0)))
    return txs


def parse_bank_rakyat(pdf: pdfplumber.PDF, filename: str) -> List[Dict]:
    """
    Public parser used by app.py:
      - Detect scanned vs digital cheaply (page 1 text length)
      - Digital: table extraction (fast, no OCR)
      - Scanned: OCR fallback (only when needed)
    """
    if not pdf.pages:
        return []

    t0 = pdf.pages[0].extract_text(x_tolerance=1) or ""
    is_scanned = len(t0.strip()) < MIN_TEXT_CHARS

    if not is_scanned:
        tx = _parse_bank_rakyat_digital(pdf, filename)
        if tx:
            return tx

    # OCR fallback (only when scanned, or when digital parsing yields nothing)
    return _parse_bank_rakyat_scanned_ocr(pdf, filename)
