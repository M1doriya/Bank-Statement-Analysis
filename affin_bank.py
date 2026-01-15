# affin_bank.py
"""
Affin Bank Malaysia - Current Account statement parser.

Root cause of "No transactions detected":
- Many Affin PDFs are IMAGE-ONLY (scanned). In that case pdfplumber.extract_text() returns empty,
  so a pure text-regex parser will never see any rows.

This parser:
1) Tries normal text extraction.
2) If a page has no extractable text, falls back to OCR (pytesseract) on that page image.
3) Parses transactions using a token-based approach that matches the Affin table layout:
   Date | Description | Debit | Credit | Balance
   - Handles values like ".00" (no leading zero)
   - Supports multi-line descriptions (continuation lines without a date)
   - Skips opening balance "B/F" row and trailing "TOTAL" summary rows
4) Outputs ISO dates (YYYY-MM-DD) for stable downstream monthly summaries.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


# -----------------------------
# Regex
# -----------------------------
DATE_TOKEN_RE = re.compile(r"^(?P<d>\d{1,2})/(?P<m>\d{2})/(?P<y>\d{2,4})$")

# Money tokens in Affin statements often look like:
#   900.00
#   0.00
#   .00              (common in credit/debit "empty" column)
#   73,625.00
MONEY_TOKEN_RE = re.compile(r"^(?:\d{1,3}(?:,\d{3})*|\d+)?\.\d{2}$")

# Lines that should stop/skip parsing
STOP_HINTS = (
    "TOTAL DEBIT",
    "TOTAL CREDIT",
    "TOTAL",
    "ACCOUNT",
    "PENYATA",
    "PAGE",
    "MEMBER",
    "PIDM",
)


def _money_to_float(tok: str) -> Optional[float]:
    if tok is None:
        return None
    s = str(tok).strip()
    if not s:
        return None
    # normalize ".00" -> "0.00"
    if s.startswith("."):
        s = "0" + s
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _date_to_iso(dmy: str) -> Optional[str]:
    m = DATE_TOKEN_RE.match(dmy.strip())
    if not m:
        return None
    dd = int(m.group("d"))
    mm = int(m.group("m"))
    yy = m.group("y")
    year = int(yy)
    if year < 100:
        year += 2000
    try:
        return datetime(year, mm, dd).strftime("%Y-%m-%d")
    except Exception:
        return None


def _looks_like_noise(line: str) -> bool:
    up = (line or "").upper().strip()
    if not up:
        return True
    return any(h in up for h in STOP_HINTS)


def _ocr_page_text(page: pdfplumber.page.Page) -> str:
    """
    OCR a pdfplumber page into text using pytesseract.

    Performance notes:
    - Full-page OCR at high DPI is slow; crop to likely table region and use moderate DPI.
    """
    if pytesseract is None:
        return ""

    try:
        # Crop away most headers/footers to speed OCR and reduce noise.
        # Coordinates are in PDF points (A4 ~ 595 x 842).
        w, h = float(page.width), float(page.height)
        table_top = 140
        table_bottom = h - 60
        cropped = page.crop((0, table_top, w, table_bottom))

        # Moderate resolution is usually sufficient for table rows.
        img = cropped.to_image(resolution=200).original

        # psm 6: assume a uniform block of text (good for tables/rows)
        return pytesseract.image_to_string(img, config="--psm 6") or ""
    except Exception:
        return ""


def _extract_text_or_ocr(page: pdfplumber.page.Page) -> str:
    # Try normal text extraction first
    text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
    if text.strip():
        return text

    # OCR fallback (for scanned PDFs)
    return _ocr_page_text(page)


def _finalize_tx(
    cur: Dict[str, Any],
    *,
    bank_name: str,
    source_file: str,
    page_num: int,
    prev_balance: Optional[float],
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Turn an in-progress transaction buffer into a canonical tx dict."""
    if not cur:
        return None, prev_balance

    date_iso = cur.get("date_iso")
    if not date_iso:
        return None, prev_balance

    desc = " ".join([p for p in cur.get("desc_parts", []) if p]).strip()
    if not desc:
        desc = cur.get("desc_head", "").strip()

    debit = cur.get("debit")
    credit = cur.get("credit")
    balance = cur.get("balance")

    # Skip B/F row but use it as anchor
    if "B/F" in (desc or "").upper():
        if balance is not None:
            prev_balance = balance
        return None, prev_balance

    # Skip totals/summary-like rows
    if any(k in (desc or "").upper() for k in ("TOTAL DEBIT", "TOTAL CREDIT", "TOTAL :")):
        return None, prev_balance

    # If we have balance but debit/credit missing/zero, infer from balance delta
    if balance is not None and prev_balance is not None:
        if (debit is None and credit is None) or (float(debit or 0) == 0.0 and float(credit or 0) == 0.0):
            delta = round(balance - prev_balance, 2)
            if delta > 0:
                credit = abs(delta)
                debit = 0.0
            elif delta < 0:
                debit = abs(delta)
                credit = 0.0

    debit = float(debit or 0.0)
    credit = float(credit or 0.0)

    # If still nothing meaningful, drop safely
    if debit == 0.0 and credit == 0.0 and balance is None:
        return None, prev_balance

    tx = {
        "date": date_iso,
        "description": desc.strip(),
        "debit": round(debit, 2),
        "credit": round(credit, 2),
        "balance": round(float(balance), 2) if balance is not None else None,
        "page": page_num,
        "bank": bank_name,
        "source_file": source_file or "",
    }

    if balance is not None:
        prev_balance = float(balance)

    return tx, prev_balance


def _parse_lines(
    lines: List[str],
    *,
    page_num: int,
    bank_name: str,
    source_file: str,
    prev_balance: Optional[float],
) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """Parse already-extracted lines (from text extraction or OCR)."""
    txs: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None

    for raw in lines:
        line = re.sub(r"\s+", " ", (raw or "")).strip()
        if not line:
            continue

        # Sometimes OCR returns pipe-like separators; remove obvious ones
        line = line.replace("|", " ").strip()

        if _looks_like_noise(line) and cur is None:
            continue

        tokens = line.split()

        # New transaction start: first token is a date
        date_iso = _date_to_iso(tokens[0]) if tokens else None
        if date_iso:
            # finalize previous buffer
            if cur is not None:
                tx, prev_balance = _finalize_tx(
                    cur,
                    bank_name=bank_name,
                    source_file=source_file,
                    page_num=page_num,
                    prev_balance=prev_balance,
                )
                if tx:
                    txs.append(tx)

            money_positions = [i for i, t in enumerate(tokens) if MONEY_TOKEN_RE.match(t)]
            debit = credit = balance = None

            if money_positions:
                balance = _money_to_float(tokens[money_positions[-1]])

                # Prefer last three money tokens: debit, credit, balance
                if len(money_positions) >= 3:
                    debit = _money_to_float(tokens[money_positions[-3]])
                    credit = _money_to_float(tokens[money_positions[-2]])
                elif len(money_positions) == 2:
                    debit = _money_to_float(tokens[money_positions[-2]])
                    credit = None

                first_money_idx = money_positions[0]
                desc_head = " ".join(tokens[1:first_money_idx]).strip()
            else:
                desc_head = " ".join(tokens[1:]).strip()

            cur = {
                "date_iso": date_iso,
                "desc_head": desc_head,
                "desc_parts": [desc_head] if desc_head else [],
                "debit": debit,
                "credit": credit,
                "balance": balance,
            }
            continue

        # Continuation line (no date)
        if cur is not None:
            money_positions = [i for i, t in enumerate(tokens) if MONEY_TOKEN_RE.match(t)]
            if money_positions:
                bal_val = _money_to_float(tokens[money_positions[-1]])
                if bal_val is not None:
                    cur["balance"] = bal_val

                if len(money_positions) >= 3:
                    cur["debit"] = _money_to_float(tokens[money_positions[-3]])
                    cur["credit"] = _money_to_float(tokens[money_positions[-2]])

                first_money_idx = money_positions[0]
                desc_part = " ".join(tokens[:first_money_idx]).strip()
                if desc_part:
                    cur["desc_parts"].append(desc_part)
            else:
                if not _looks_like_noise(line):
                    cur["desc_parts"].append(line)

    # finalize last buffer
    if cur is not None:
        tx, prev_balance = _finalize_tx(
            cur,
            bank_name=bank_name,
            source_file=source_file,
            page_num=page_num,
            prev_balance=prev_balance,
        )
        if tx:
            txs.append(tx)

    return txs, prev_balance


def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    """
    Entry used by app.py (via pdfplumber object):
      parse_affin_bank(pdf, filename) -> list of tx dicts

    Also supports being called with:
      - file-like
      - bytes
      - filesystem path
    """
    bank_name = "Affin Bank"
    transactions: List[Dict[str, Any]] = []
    prev_balance: Optional[float] = None

    def _parse_pdf(pdf: pdfplumber.PDF):
        nonlocal transactions, prev_balance
        for page_num, page in enumerate(pdf.pages, start=1):
            text = _extract_text_or_ocr(page)
            if not text.strip():
                continue

            lines = [ln for ln in text.splitlines() if ln and ln.strip()]
            page_txs, prev_balance = _parse_lines(
                lines,
                page_num=page_num,
                bank_name=bank_name,
                source_file=source_file,
                prev_balance=prev_balance,
            )
            transactions.extend(page_txs)

    if hasattr(pdf_input, "pages"):
        _parse_pdf(pdf_input)
        return transactions

    try:
        if hasattr(pdf_input, "seek"):
            try:
                pdf_input.seek(0)
            except Exception:
                pass

        with pdfplumber.open(pdf_input) as pdf:
            _parse_pdf(pdf)

    except Exception as e:
        raise RuntimeError(f"Affin Bank parser failed for '{source_file}': {e}") from e

    return transactions
