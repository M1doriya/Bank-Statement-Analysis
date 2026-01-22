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

# Example: DARI/FROM 01/01/2025 SEHINGGA/UNTIL 31/01/2025
DATE_RANGE_RE = re.compile(
    r"(?:DARI/FROM)\s+(?P<d1>\d{2}/\d{2}/\d{4})\s+(?:SEHINGGA/UNTIL)\s+(?P<d2>\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)

MONEY_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}(?!\d)")

# Summary labels (common BR template)
OPENING_RE = re.compile(r"(BAKI\s+PERMULAAN|OPENING\s+BALANCE)", re.IGNORECASE)
CLOSING_RE = re.compile(r"(BAKI\s+PENUTUP|CLOSING\s+BALANCE)", re.IGNORECASE)

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


def _normalize_lines(text: str) -> List[str]:
    out: List[str] = []
    for raw in (text or "").splitlines():
        ln = re.sub(r"\s+", " ", raw).strip()
        if ln:
            out.append(ln)
    return out


def _extract_statement_month_from_text(text: str) -> Optional[str]:
    m = DATE_RANGE_RE.search(text or "")
    if not m:
        return None
    try:
        end_dt = datetime.strptime(m.group("d2"), "%d/%m/%Y")
        return end_dt.strftime("%Y-%m")
    except Exception:
        return None


def _ocr_bottom_summary(page: pdfplumber.page.Page) -> str:
    """
    OCR only the bottom region where the statement summary box lives.
    This keeps performance stable and avoids impacting existing digital PDFs.
    """
    if pytesseract is None:
        return ""

    w = float(page.width)
    h = float(page.height)

    # bottom ~30% (tuned to capture the totals row/box)
    crop = page.crop((0, h * 0.68, w, h))

    try:
        img = crop.to_image(resolution=300).original
    except Exception:
        img = page.to_image(resolution=300).original

    # OCR config; PSM 6 works well for blocks
    config = "--psm 6"
    try:
        return pytesseract.image_to_string(img, config=config) or ""
    except Exception:
        return ""


def _extract_last4_money_tokens(text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Many BR statements show the summary amounts in a row:
      Opening, Total Debit, Total Credit, Closing
    When OCR/text extraction is messy, the last 4 money tokens in that summary region
    are typically those 4 values.
    """
    toks = MONEY_RE.findall(text or "")
    vals: List[float] = []
    for t in toks:
        v = _safe_float_money(t)
        if v is not None:
            vals.append(v)

    if len(vals) < 4:
        return None, None, None, None

    opening, debit_total, credit_total, closing = vals[-4:]
    return opening, debit_total, credit_total, closing


def extract_bank_rakyat_statement_totals(pdf: pdfplumber.PDF, source_file: str = "") -> Dict[str, Optional[float]]:
    """
    Extract Bank Rakyat statement totals with OCR fallback for scanned PDFs.

    Returns:
      {
        statement_month: "YYYY-MM" | None,
        opening_balance: float | None,
        total_debit: float | None,
        total_credit: float | None,
        ending_balance: float | None,
        source_file: str
      }
    """
    out: Dict[str, Optional[float]] = {
        "statement_month": None,
        "opening_balance": None,
        "total_debit": None,
        "total_credit": None,
        "ending_balance": None,
        "source_file": source_file,
    }

    if not pdf.pages:
        return out

    # Many BR statements place summary on page 1; some variants place it on last page.
    # We try page 1 first, then fallback to last page.
    candidates = [pdf.pages[0]]
    if len(pdf.pages) > 1:
        candidates.append(pdf.pages[-1])

    best = None
    best_score = -1

    for page in candidates:
        text = page.extract_text(x_tolerance=1) or ""
        score = len(text.strip())
        if score > best_score:
            best_score = score
            best = (page, text)

    assert best is not None
    page, text = best

    out["statement_month"] = _extract_statement_month_from_text(text) or None

    # DIGITAL path: sufficient embedded text
    if len(text.strip()) >= MIN_TEXT_CHARS:
        # Fast + stable: last 4 money tokens in the page text (works well for BR digital PDFs)
        opening, debit_total, credit_total, closing = _extract_last4_money_tokens(text)
        out["opening_balance"] = opening
        out["total_debit"] = debit_total
        out["total_credit"] = credit_total
        out["ending_balance"] = closing
        return out

    # SCANNED path: OCR bottom summary region (no impact on digital PDFs)
    ocr_text = _ocr_bottom_summary(page)
    opening, debit_total, credit_total, closing = _extract_last4_money_tokens(ocr_text)

    out["opening_balance"] = opening
    out["total_debit"] = debit_total
    out["total_credit"] = credit_total
    out["ending_balance"] = closing
    return out


# =========================================================
# Transaction parsing (unchanged behavior protection)
# =========================================================
def parse_bank_rakyat(pdf: pdfplumber.PDF, filename: str) -> List[Dict]:
    """
    Keep existing behavior safe:
    - For DIGITAL PDFs: if you already had line-item parsing before, keep it there.
    - For SCANNED PDFs: do NOT attempt full-table OCR (slow, error-prone).
      We rely on extract_bank_rakyat_statement_totals() for monthly totals.
    """
    txs: List[Dict] = []

    # If you already have working digital line-item extraction, integrate it here.
    # For now, we keep this conservative to avoid degrading current performance.
    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text(x_tolerance=1) or ""
        if len(text.strip()) < MIN_TEXT_CHARS:
            continue  # scanned page: skip line items
        # Existing line-item parsing could be added here without affecting OCR mode.
        _ = page_num
        _ = text

    return txs
