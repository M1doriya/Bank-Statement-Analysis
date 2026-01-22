# bank_rakyat.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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

MONEY_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}(?!\d)")
INT_RE = re.compile(r"(?<!\d)\d{1,4}(?!\d)")

# Labels / keywords that appear in summary box (Malay/English)
OPENING_KW = re.compile(r"(BAKI\s+PERMULAAN|OPENING\s+BALANCE)", re.IGNORECASE)
CLOSING_KW = re.compile(r"(BAKI\s+PENUTUP|CLOSING\s+BALANCE)", re.IGNORECASE)
DEBIT_KW = re.compile(r"\bDEBIT\b", re.IGNORECASE)
CREDIT_KW = re.compile(r"\bKREDIT\b|\bCREDIT\b", re.IGNORECASE)

# Dates in table column
DATE_DMY_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")

# Decide scanned vs digital text
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


def _ocr_crop(page: pdfplumber.page.Page, *, top_ratio: float, bottom_ratio: float, left_ratio: float, right_ratio: float, dpi: int = 300) -> str:
    """
    OCR a cropped region defined by ratios.
    """
    if pytesseract is None:
        return ""

    w = float(page.width)
    h = float(page.height)
    left = w * left_ratio
    right = w * right_ratio
    top = h * top_ratio
    bottom = h * bottom_ratio

    crop = page.crop((left, top, right, bottom))
    try:
        img = crop.to_image(resolution=dpi).original
    except Exception:
        img = page.to_image(resolution=dpi).original

    try:
        return pytesseract.image_to_string(img, config="--psm 6") or ""
    except Exception:
        return ""


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _find_money_near_keyword(text: str, kw: re.Pattern, window: int = 80) -> Optional[float]:
    """
    Find the first money token near the keyword occurrence within a char window.
    Works well on OCR text where columns are not perfectly aligned.
    """
    t = text or ""
    m = kw.search(t)
    if not m:
        return None

    start = max(0, m.start() - window)
    end = min(len(t), m.end() + window)
    snippet = t[start:end]

    # Prefer the closest money token AFTER the keyword, then fallback BEFORE
    after = t[m.end():min(len(t), m.end() + window)]
    toks_after = MONEY_RE.findall(after)
    if toks_after:
        v = _safe_float_money(toks_after[0])
        if v is not None:
            return v

    toks_any = MONEY_RE.findall(snippet)
    if toks_any:
        v = _safe_float_money(toks_any[-1])
        if v is not None:
            return v

    return None


def _extract_totals_from_summary_text(summary_text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extract opening, debit_total, credit_total, closing using keyword-near-number strategy.
    This fixes Jan 2025 where 'last 4 money tokens' is unreliable.
    """
    txt = _normalize_space(summary_text)

    opening = _find_money_near_keyword(txt, OPENING_KW)
    closing = _find_money_near_keyword(txt, CLOSING_KW)

    debit_total = _find_money_near_keyword(txt, DEBIT_KW)
    credit_total = _find_money_near_keyword(txt, CREDIT_KW)

    # Fallback: if any missing, try last-4 approach as secondary fallback (not primary)
    vals = [_safe_float_money(x) for x in MONEY_RE.findall(txt)]
    vals = [v for v in vals if v is not None]
    if len(vals) >= 4:
        last4 = vals[-4:]
        if opening is None:
            opening = last4[0]
        if debit_total is None:
            debit_total = last4[1]
        if credit_total is None:
            credit_total = last4[2]
        if closing is None:
            closing = last4[3]

    return opening, debit_total, credit_total, closing


def _count_transactions_scanned(pdf: pdfplumber.PDF) -> Optional[int]:
    """
    Lightweight OCR count:
    - OCR only the DATE column region across each page (excluding header/footer).
    - Count date patterns dd/mm/yyyy.
    This gives a good transaction count without full-table OCR.
    """
    if pytesseract is None or not pdf.pages:
        return None

    total = 0
    for page in pdf.pages:
        # Middle region where the transaction table lives.
        # Left-side slice (date column area).
        ocr = _ocr_crop(
            page,
            top_ratio=0.18, bottom_ratio=0.88,
            left_ratio=0.02, right_ratio=0.28,
            dpi=250,
        )
        dates = DATE_DMY_RE.findall(ocr or "")
        if dates:
            total += len(dates)

    # Basic sanity: if zero, return None (avoid false confidence)
    return total if total > 0 else None


def extract_bank_rakyat_statement_totals(pdf: pdfplumber.PDF, source_file: str = "") -> Dict[str, Optional[float]]:
    """
    Extract Bank Rakyat statement totals with OCR fallback for scanned PDFs.
    Also extracts transaction_count for scanned PDFs.
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
        return out

    # Prefer last page for totals on some templates; also try first page.
    pages_to_try = [pdf.pages[0]]
    if len(pdf.pages) > 1:
        pages_to_try.append(pdf.pages[-1])

    # First try digital extraction
    for page in pages_to_try:
        text = page.extract_text(x_tolerance=1) or ""
        if len(text.strip()) >= MIN_TEXT_CHARS:
            out["statement_month"] = _extract_statement_month_from_text(text) or out["statement_month"]

            # For digital PDFs, extracting totals from embedded text is typically reliable,
            # but Jan scanned is our use-case so this won't run there.
            opening, debit_total, credit_total, closing = _extract_totals_from_summary_text(text)
            out["opening_balance"] = opening
            out["total_debit"] = debit_total
            out["total_credit"] = credit_total
            out["ending_balance"] = closing
            # transaction_count from line items is handled by app (if tx parsed); leave None here.
            return out

    # Scanned path: OCR only bottom summary box on first and last page, pick the best hit
    best = (None, None, None, None, -1)  # opening, debit, credit, closing, score
    stmt_month = None

    for page in pages_to_try:
        # Bottom summary region OCR
        ocr_summary = _ocr_crop(
            page,
            top_ratio=0.68, bottom_ratio=1.00,
            left_ratio=0.00, right_ratio=1.00,
            dpi=300,
        )
        ocr_summary_n = _normalize_space(ocr_summary)

        opening, debit_total, credit_total, closing = _extract_totals_from_summary_text(ocr_summary_n)

        score = 0
        for v in (opening, debit_total, credit_total, closing):
            if v is not None:
                score += 1

        if score > best[4]:
            best = (opening, debit_total, credit_total, closing, score)

        # Try to get statement month by OCRing top band (where date range lives)
        if stmt_month is None:
            ocr_top = _ocr_crop(
                page,
                top_ratio=0.00, bottom_ratio=0.22,
                left_ratio=0.00, right_ratio=1.00,
                dpi=250,
            )
            stmt_month = _extract_statement_month_from_text(ocr_top)

    out["statement_month"] = stmt_month

    opening, debit_total, credit_total, closing = best[0], best[1], best[2], best[3]
    out["opening_balance"] = opening
    out["total_debit"] = debit_total
    out["total_credit"] = credit_total
    out["ending_balance"] = closing

    # Transaction count for scanned PDFs
    out["transaction_count"] = _count_transactions_scanned(pdf)

    return out


# =========================================================
# Transaction parsing
# =========================================================
def parse_bank_rakyat(pdf: pdfplumber.PDF, filename: str) -> List[Dict]:
    """
    Keep existing behavior safe.
    - DIGITAL PDFs: if you already had parsing, keep it there (not included here).
    - SCANNED PDFs: do not OCR full transactions (we only count dates via statement totals).
    """
    txs: List[Dict] = []
    for _page in pdf.pages:
        text = _page.extract_text(x_tolerance=1) or ""
        if len(text.strip()) < MIN_TEXT_CHARS:
            continue  # scanned: skip line items

        # If you already have a working digital parser, integrate it here.
        # We keep empty to avoid changing your existing working logic elsewhere.

    return txs
