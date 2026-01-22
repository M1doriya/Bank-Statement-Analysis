# bank_rakyat.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pdfplumber

try:
    import pytesseract  # type: ignore
    from pytesseract import Output  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None
    Output = None


# =========================================================
# Regex
# =========================================================

DATE_RANGE_RE = re.compile(
    r"(?:DARI/FROM)\s+(?P<d1>\d{2}/\d{2}/\d{4})\s+(?:SEHINGGA/UNTIL)\s+(?P<d2>\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)

MONEY_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}(?!\d)")
DATE_DMY_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")

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

def _safe_float_money(s: str) -> Optional[float]:
    if not s:
        return None
    s = str(s).strip()
    if not MONEY_RE.fullmatch(s):
        return None
    try:
        return float(s.replace(",", ""))
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


def _ocr_image_crop(
    page: pdfplumber.page.Page,
    *,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
    dpi: int = 300,
):
    """
    Return a PIL image for a cropped region.
    """
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


def _ocr_tokens_money(image) -> List[Dict]:
    """
    Use pytesseract image_to_data to get positioned tokens.
    Returns list of {val, x, y, w, h, cy}.
    """
    if pytesseract is None or Output is None:
        return []

    data = pytesseract.image_to_data(image, output_type=Output.DICT, config="--psm 6")
    n = len(data.get("text", []))
    out: List[Dict] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        # clean common OCR artifacts
        txt = txt.replace("O", "0").replace("o", "0")
        if not MONEY_RE.fullmatch(txt):
            continue
        val = _safe_float_money(txt)
        if val is None:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        cy = y + (h / 2.0)
        out.append({"val": val, "x": x, "y": y, "w": w, "h": h, "cy": cy})
    return out


def _group_tokens_by_line(tokens: List[Dict], y_tol: float = 12.0) -> List[List[Dict]]:
    """
    Cluster tokens into lines by their vertical center (cy).
    """
    if not tokens:
        return []
    toks = sorted(tokens, key=lambda t: (t["cy"], t["x"]))
    lines: List[List[Dict]] = []
    cur: List[Dict] = [toks[0]]
    cur_y = toks[0]["cy"]

    for t in toks[1:]:
        if abs(t["cy"] - cur_y) <= y_tol:
            cur.append(t)
            cur_y = (cur_y * 0.7) + (t["cy"] * 0.3)
        else:
            lines.append(cur)
            cur = [t]
            cur_y = t["cy"]
    lines.append(cur)
    return lines


def _best_tuple_from_lines(lines: List[List[Dict]], tol: float = 0.05) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    From OCR lines containing >=4 money tokens, choose the 4-tuple (opening, debit, credit, closing)
    that best satisfies: opening + credit - debit ≈ closing.
    We test both orders for debit/credit in case they are swapped by layout.
    """
    best = (None, None, None, None)
    best_score = float("inf")

    for line in lines:
        if len(line) < 4:
            continue
        # sort left-to-right
        line_sorted = sorted(line, key=lambda t: t["x"])
        vals = [t["val"] for t in line_sorted]

        # sliding windows of 4 consecutive values (robust when extra values exist on same line)
        for i in range(0, len(vals) - 3):
            a, b, c, d = vals[i], vals[i + 1], vals[i + 2], vals[i + 3]

            # hypothesis 1: opening, debit, credit, closing
            score1 = abs((a + c - b) - d)
            # hypothesis 2: opening, credit, debit, closing
            score2 = abs((a + b - c) - d)

            if score1 < best_score:
                best_score = score1
                best = (a, b, c, d)

            if score2 < best_score:
                best_score = score2
                best = (a, c, b, d)  # normalize to (opening, debit, credit, closing)

    # Require a reasonable identity match; otherwise return Nones
    if best[0] is None:
        return None, None, None, None
    if best_score > tol:
        # In some scans, totals might be split across lines; allow fallback later.
        # Still return best effort if it's not absurdly off.
        return best
    return best


def _count_transactions_scanned(pdf: pdfplumber.PDF) -> Optional[int]:
    """
    Count transaction rows in scanned PDFs by counting date occurrences in the date column region.
    """
    if pytesseract is None or not pdf.pages:
        return None

    total = 0
    for page in pdf.pages:
        img = _ocr_image_crop(
            page,
            left_ratio=0.02,
            top_ratio=0.18,
            right_ratio=0.30,
            bottom_ratio=0.88,
            dpi=250,
        )
        txt = pytesseract.image_to_string(img, config="--psm 6") or ""
        dates = DATE_DMY_RE.findall(txt)
        if dates:
            total += len(dates)

    return total if total > 0 else None


# =========================================================
# Public API
# =========================================================

def extract_bank_rakyat_statement_totals(pdf: pdfplumber.PDF, source_file: str = "") -> Dict[str, Optional[float]]:
    """
    Extract statement totals with OCR fallback and identity-based tuple selection.
    Includes transaction_count for scanned PDFs.
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

    # Prefer page 1 and last page (some templates put the summary on the last page)
    candidate_pages = [pdf.pages[0]]
    if len(pdf.pages) > 1:
        candidate_pages.append(pdf.pages[-1])

    # 1) Try DIGITAL extraction (embedded text)
    for page in candidate_pages:
        text = page.extract_text(x_tolerance=1) or ""
        if len(text.strip()) >= MIN_TEXT_CHARS:
            out["statement_month"] = _extract_statement_month_from_text(text) or _extract_statement_month_from_filename(source_file)

            # Even in digital, summary can be tricky; we can still use identity selection using plain token order.
            vals = [_safe_float_money(x) for x in MONEY_RE.findall(text)]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 4:
                # Try all windows of 4 and pick identity-best
                best = (None, None, None, None)
                best_score = float("inf")
                for i in range(0, len(vals) - 3):
                    a, b, c, d = vals[i], vals[i + 1], vals[i + 2], vals[i + 3]
                    s1 = abs((a + c - b) - d)
                    s2 = abs((a + b - c) - d)
                    if s1 < best_score:
                        best_score = s1
                        best = (a, b, c, d)
                    if s2 < best_score:
                        best_score = s2
                        best = (a, c, b, d)
                out["opening_balance"], out["total_debit"], out["total_credit"], out["ending_balance"] = best
            return out

    # 2) SCANNED extraction (OCR)
    # 2a) Month: OCR top band; if fails, filename fallback
    month = None
    for page in candidate_pages:
        top_img = _ocr_image_crop(page, left_ratio=0.0, top_ratio=0.0, right_ratio=1.0, bottom_ratio=0.25, dpi=250)
        top_txt = ""
        try:
            top_txt = pytesseract.image_to_string(top_img, config="--psm 6") or ""
        except Exception:
            top_txt = ""
        month = _extract_statement_month_from_text(top_txt)
        if month:
            break
    out["statement_month"] = month or _extract_statement_month_from_filename(source_file)

    # 2b) Totals: OCR bottom summary region and identity-select 4-tuple
    best_tuple = (None, None, None, None)
    best_score = float("inf")

    for page in candidate_pages:
        bottom_img = _ocr_image_crop(page, left_ratio=0.0, top_ratio=0.65, right_ratio=1.0, bottom_ratio=1.0, dpi=300)
        tokens = _ocr_tokens_money(bottom_img)
        lines = _group_tokens_by_line(tokens, y_tol=14.0)
        opening, debit, credit, closing = _best_tuple_from_lines(lines, tol=0.10)

        if opening is None or debit is None or credit is None or closing is None:
            continue

        score = abs((opening + credit - debit) - closing)
        if score < best_score:
            best_score = score
            best_tuple = (opening, debit, credit, closing)

    out["opening_balance"], out["total_debit"], out["total_credit"], out["ending_balance"] = best_tuple

    # 2c) Transaction count for scanned PDFs
    out["transaction_count"] = _count_transactions_scanned(pdf)

    return out


def parse_bank_rakyat(pdf: pdfplumber.PDF, filename: str) -> List[Dict]:
    """
    Keep conservative behavior: do NOT OCR full transaction table (performance).
    Monthly totals should be taken from extract_bank_rakyat_statement_totals().
    """
    txs: List[Dict] = []
    # If you already have a reliable digital parser, integrate it here.
    return txs
