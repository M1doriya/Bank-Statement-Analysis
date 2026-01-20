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

DATE_IN_TOKEN_RE = re.compile(
    r"(?P<d>\d{1,2})\s*/\s*(?P<m>\d{1,2})\s*/\s*(?P<y>\d{2,4})"
)

MONEY_TOKEN_RE = re.compile(
    r"^\(?\s*(?:RM\s*)?(?P<num>(?:\d{1,3}(?:,\d{3})*|\d+)?\.\d{2})\s*\)?(?P<trail_sign>[+-])?\s*[\.,;:|]*\s*$",
    re.I,
)

# Generic money finder in free text
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

# Filename month inference: 2025_04.pdf, 2025-04.pdf, 04-2025, etc.
FILENAME_MONTH_RE = re.compile(r"(?P<y>20\d{2})[^\d]?(?P<m>0[1-9]|1[0-2])")


# =========================================================
# Small helpers
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


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
    """
    Best-effort text:
    - try embedded text first (fast)
    - fallback to OCR (slower)
    """
    txt = page.extract_text() or ""
    txt = txt.strip()
    if len(txt) >= 200:  # heuristic: enough text to likely include totals
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
        # amount is in the last capturing group
        amt = m.group(m.lastindex) if m.lastindex else None
        f = _money_to_float_from_text(amt or "")
        if f is not None:
            return float(f)
    return None


# =========================================================
# PUBLIC: Extract statement totals (source of truth for Affin monthly summary)
# =========================================================
def extract_affin_statement_totals(pdf_input: Any, source_file: str = "") -> Dict[str, Any]:
    """
    Extract Affin statement 'ground truth' totals:
      - opening_balance
      - total_debit
      - total_credit
      - ending_balance
      - statement_month (best effort from filename)

    Designed for OCR PDFs where line-item extraction is noisy.
    """
    # Open pdf if needed
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
        idxs = []
        # scan first two and last two pages (totals are usually there)
        for i in [0, 1, max(0, n - 2), max(0, n - 1)]:
            if 0 <= i < n and i not in idxs:
                idxs.append(i)

        blob = ""
        for i in idxs:
            blob += "\n" + (_page_text_pdf_or_ocr(pdf.pages[i]) or "")

        # normalize
        blob = blob.replace("\x0c", " ")
        blob = _norm(blob)

        total_debit = _find_first_amount(blob, TOTAL_DEBIT_PATTERNS)
        total_credit = _find_first_amount(blob, TOTAL_CREDIT_PATTERNS)
        opening_balance = _find_first_amount(blob, OPENING_PATTERNS)
        ending_balance = _find_first_amount(blob, CLOSING_PATTERNS)

        # Fallbacks if opening/closing not found explicitly:
        # - If we see "BALANCE B/F" but pattern missed, grab closest large money in same line
        if opening_balance is None:
            m = re.search(r"(BALANCE|BAKI).*?\bB/F\b.*?", blob, re.I)
            if m:
                # grab first money after it
                tail = blob[m.start(): m.start() + 250]
                nums = MONEY_IN_TEXT_RE.findall(tail)
                if nums:
                    opening_balance = _money_to_float_from_text(nums[-1])

        if ending_balance is None:
            m = re.search(r"(BALANCE|BAKI).*?\bC/F\b.*?", blob, re.I)
            if m:
                tail = blob[m.start(): m.start() + 250]
                nums = MONEY_IN_TEXT_RE.findall(tail)
                if nums:
                    ending_balance = _money_to_float_from_text(nums[-1])

        statement_month = _infer_month_from_filename(source_file) or None

        return {
            "bank": "Affin Bank",
            "source_file": source_file or "",
            "statement_month": statement_month,
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


# =========================================================
# Your existing parser entry point (leave as-is)
# =========================================================
def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    """
    Keep your current line-item extractor.
    Monthly totals for Affin should come from extract_affin_statement_totals() instead.
    """
    # If you already have a working parse_affin_bank, keep it.
    # This stub returns empty if you prefer to rely only on statement totals.
    #
    # To avoid disrupting your setup, paste your current parse_affin_bank implementation here.
    #
    # For now, we keep it minimal:
    return []
