# affin_bank.py
"""
Affin Bank statement parser (text-first with OCR fallback).

Why this exists:
- Many Affin PDFs are scanned/image-only -> pdfplumber.extract_text() returns empty.
- Amount columns (Wang Masuk / Wang Keluar / Baki) require column-aware parsing.
- Multi-line descriptions are common.

Output schema (your app expects):
  date, description, debit, credit, balance, page, bank, source_file

Notes:
- Uses OCR only when a page has no meaningful text OR when text parsing yields nothing.
- Column detection:
    1) Try header words ("Wang Masuk", "Wang Keluar", "Baki") from OCR words
    2) Else infer 3 numeric columns by clustering x-positions of money tokens
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

# OCR dependencies (already in your environment per your requirements)
import pytesseract


# -----------------------------
# Regex
# -----------------------------
# Dates seen in statements: 01/08/2025, 1/8/25, sometimes 01-08-2025
DATE_RE = re.compile(r"^(?P<d>\d{1,2})[/-](?P<m>\d{1,2})[/-](?P<y>\d{2,4})$")

# Money: 1,234.56  (OCR may break commas/spaces)
MONEY_RE = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+(?:\.\d{2})$")

# Lines to ignore if they appear in description-only rows
HEADER_NOISE_RE = re.compile(
    r"\b(tarikh|uraian|wang|masuk|keluar|baki|page|muka\s*surat|statement|account)\b",
    re.I
)


# -----------------------------
# OCR cleanup maps
# -----------------------------
_OCR_CHAR_MAP = str.maketrans({
    "O": "0", "o": "0",
    "S": "5", "s": "5",
    "I": "1", "l": "1", "|": "1",
    "B": "8",
    "“": "", "”": "", "’": "", "`": "", "'": "",
    " ": "",
})


# -----------------------------
# Core parsing helpers
# -----------------------------
def _parse_date_token(tok: str) -> Optional[str]:
    tok = (tok or "").strip()
    m = DATE_RE.match(tok)
    if not m:
        return None

    d = int(m.group("d"))
    mo = int(m.group("m"))
    y = m.group("y")
    if len(y) == 2:
        # assume 20xx for 2-digit years in modern statements
        y_int = 2000 + int(y)
    else:
        y_int = int(y)

    try:
        dt = datetime(y_int, mo, d)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _clean_money_token(tok: str) -> Optional[float]:
    """Return float if token looks like money; else None.
    Handles OCR artifacts: O->0, S->5, spaces, stray punctuation.
    """
    if tok is None:
        return None
    s = str(tok).strip()
    if not s:
        return None

    # Normalize OCR artifacts and remove spaces
    s = s.translate(_OCR_CHAR_MAP)

    # Remove any stray non money punctuation (keep digits, comma, dot)
    s = re.sub(r"[^0-9,\.]", "", s)
    if not s:
        return None

    # Standardize commas
    # Example OCR: "1,23 4.56" becomes "1,234.56" after whitespace removal; keep commas.
    # Validate and parse
    if MONEY_RE.fullmatch(s):
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None

    # Sometimes OCR drops comma placement: "1234.56" is fine
    if re.fullmatch(r"^\d+\.\d{2}$", s):
        try:
            return float(s)
        except Exception:
            return None

    return None


def _group_words_by_line(words: List[Dict[str, Any]], y_tol: float = 6.0) -> List[List[Dict[str, Any]]]:
    """Group word boxes into lines using 'top' coordinate tolerance."""
    lines: List[List[Dict[str, Any]]] = []
    for w in words:
        txt = (w.get("text") or "").strip()
        if not txt:
            continue

        placed = False
        for ln in lines:
            if abs(float(ln[0]["top"]) - float(w["top"])) <= y_tol:
                ln.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    for ln in lines:
        ln.sort(key=lambda x: float(x.get("x0", 0.0)))
    lines.sort(key=lambda ln: float(ln[0].get("top", 0.0)))
    return lines


def _ocr_words_from_page(page: pdfplumber.page.Page, resolution: int = 300) -> Tuple[List[Dict[str, Any]], float]:
    """OCR the page image and return word-level boxes + page image width."""
    im = page.to_image(resolution=resolution).original
    width = float(im.size[0])

    data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))

    words: List[Dict[str, Any]] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue

        try:
            conf = float(data.get("conf", [])[i])
        except Exception:
            conf = -1.0

        # Conservative threshold: avoid garbage; keep real numeric tokens
        if conf != -1.0 and conf < 35:
            continue

        x = float(data["left"][i])
        y = float(data["top"][i])
        w = float(data["width"][i])
        h = float(data["height"][i])

        words.append({"text": txt, "x0": x, "x1": x + w, "top": y, "bottom": y + h})

    return words, width


def _detect_columns_from_header(lines: List[List[Dict[str, Any]]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Try to find x positions of Masuk/Keluar/Baki from header lines."""
    credit_x = debit_x = balance_x = None

    scan = lines[:40] if len(lines) > 40 else lines
    for ln in scan:
        texts = [w["text"].lower() for w in ln]
        # look for "wang masuk", "wang keluar", "baki"
        for idx, w in enumerate(ln):
            t = (w["text"] or "").strip().lower()
            if t == "masuk":
                credit_x = float(w["x0"])
            elif t == "keluar":
                debit_x = float(w["x0"])
            elif t == "baki":
                balance_x = float(w["x0"])

        if credit_x and debit_x and balance_x:
            return credit_x, debit_x, balance_x

    return credit_x, debit_x, balance_x


def _infer_columns_from_money_x(money_x: List[float], page_width: float) -> Tuple[float, float, float]:
    """
    Infer 3 column centers from x-positions of money tokens using robust quantiles.

    Approach:
    - Filter to right half of page (transaction amount columns are typically on right).
    - Use quantiles as proxies for 3 columns.
    """
    if not money_x:
        # fall back to proportional layout
        return page_width * 0.62, page_width * 0.76, page_width * 0.90

    xs = sorted(money_x)

    # Focus on right half (avoid left-side account numbers / dates)
    xs = [x for x in xs if x >= page_width * 0.45] or sorted(money_x)

    # Quantile-based centers
    def q(p: float) -> float:
        if not xs:
            return 0.0
        k = int(round((len(xs) - 1) * p))
        k = max(0, min(k, len(xs) - 1))
        return float(xs[k])

    credit_x = q(0.33)
    debit_x = q(0.66)
    balance_x = q(0.92)

    # Ensure ordering and separation
    centers = sorted([credit_x, debit_x, balance_x])
    credit_x, debit_x, balance_x = centers[0], centers[1], centers[2]

    # If clustering collapsed (all same x), fall back to proportions
    if abs(balance_x - credit_x) < 25:
        return page_width * 0.62, page_width * 0.76, page_width * 0.90

    return credit_x, debit_x, balance_x


def _assign_amounts_to_columns(
    money_items: List[Tuple[float, float]],
    credit_x: float,
    debit_x: float,
    balance_x: float
) -> Tuple[float, float, Optional[float]]:
    """
    Assign numeric tokens to credit/debit/balance by nearest column x.
    Returns (credit, debit, balance).

    Rules:
    - Balance is typically the right-most column; we allow multiple balance tokens but use the last one.
    - Credit/debit are summed if multiple tokens fall into those columns (rare, but safe).
    """
    credit = 0.0
    debit = 0.0
    balance: Optional[float] = None

    if not money_items:
        return 0.0, 0.0, None

    # Sort left->right
    money_items = sorted(money_items, key=lambda t: t[0])

    # Assign each token to nearest center
    for x0, val in money_items:
        dc = abs(x0 - credit_x)
        dd = abs(x0 - debit_x)
        db = abs(x0 - balance_x)

        if db <= min(dc, dd) and db <= 45:
            balance = float(val)
        elif dc <= dd:
            credit += float(val)
        else:
            debit += float(val)

    return round(credit, 2), round(debit, 2), (round(balance, 2) if balance is not None else None)


def _extract_transactions_from_lines(
    lines: List[List[Dict[str, Any]]],
    page_width: float,
    page_idx: int,
    source_file: str
) -> List[Dict[str, Any]]:
    """
    Reconstruct transactions:
    - A transaction starts at a line containing a date token early in the line.
    - Subsequent non-date lines are continuations of description (and may carry amount tokens).
    """
    bank_name = "Affin Bank"
    out: List[Dict[str, Any]] = []

    # Detect columns from header if possible
    hx_c, hx_d, hx_b = _detect_columns_from_header(lines)

    # Gather money x positions for clustering fallback
    money_x_positions: List[float] = []
    for ln in lines:
        for w in ln:
            val = _clean_money_token(w.get("text"))
            if val is not None:
                money_x_positions.append(float(w.get("x0", 0.0)))

    # Final column centers
    if hx_c and hx_d and hx_b:
        credit_x, debit_x, balance_x = float(hx_c), float(hx_d), float(hx_b)
    else:
        credit_x, debit_x, balance_x = _infer_columns_from_money_x(money_x_positions, page_width)

    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln:
            i += 1
            continue

        # detect date
        date_iso = None
        for w in ln[:5]:
            date_iso = _parse_date_token(w.get("text", ""))
            if date_iso:
                break

        if not date_iso:
            i += 1
            continue

        desc_parts: List[str] = []
        money_items: List[Tuple[float, float]] = []

        def consume_line(line_words: List[Dict[str, Any]]):
            nonlocal desc_parts, money_items
            for w in line_words:
                txt = (w.get("text") or "").strip()
                if not txt:
                    continue

                # skip date tokens
                if _parse_date_token(txt):
                    continue

                # money token
                mv = _clean_money_token(txt)
                if mv is not None:
                    money_items.append((float(w.get("x0", 0.0)), float(mv)))
                    continue

                # skip headers/noise
                if HEADER_NOISE_RE.search(txt):
                    continue

                desc_parts.append(txt)

        consume_line(ln)

        # continuation lines until next date
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            nxt_has_date = False
            for w in nxt[:5]:
                if _parse_date_token(w.get("text", "")):
                    nxt_has_date = True
                    break
            if nxt_has_date:
                break

            consume_line(nxt)
            j += 1

        description = " ".join(desc_parts).strip()
        credit, debit, balance = _assign_amounts_to_columns(money_items, credit_x, debit_x, balance_x)

        # drop empty/non-transaction rows
        if credit == 0.0 and debit == 0.0 and balance is None:
            i = j
            continue

        out.append({
            "date": date_iso,
            "description": description,
            "debit": float(debit),
            "credit": float(credit),
            "balance": balance,
            "page": int(page_idx),
            "bank": bank_name,
            "source_file": source_file,
            "row_type": "transaction",
        })

        i = j

    return out


# -----------------------------
# Main entry point (used by app.py)
# -----------------------------
def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    """
    Your app calls this as parse_affin_bank(pdfplumber_pdf, filename).
    We support both:
      - pdfplumber.PDF object
      - file-like object / path
    """
    def _run(pdf: pdfplumber.PDF) -> List[Dict[str, Any]]:
        all_tx: List[Dict[str, Any]] = []
        for page_idx, page in enumerate(pdf.pages, start=1):
            # Try extracting text; if meaningful, you could add a pure text parser.
            # However Affin statements are table-like and often scanned; OCR path is more reliable.
            text = (page.extract_text() or "").strip()

            # OCR fallback triggers if no text or text too small to be useful
            if not text or len(text) < 80:
                words, page_width = _ocr_words_from_page(page, resolution=300)
                if not words:
                    continue
                lines = _group_words_by_line(words, y_tol=6.0)
                page_tx = _extract_transactions_from_lines(lines, page_width, page_idx, source_file)
                all_tx.extend(page_tx)
            else:
                # Even when text exists, OCR still tends to be more consistent for column tables.
                # If you want "text-first", you can add a table/text parser here.
                # For now, we still OCR to avoid template variability.
                words, page_width = _ocr_words_from_page(page, resolution=300)
                if not words:
                    continue
                lines = _group_words_by_line(words, y_tol=6.0)
                page_tx = _extract_transactions_from_lines(lines, page_width, page_idx, source_file)
                all_tx.extend(page_tx)

        return all_tx

    # If caller passed an already-opened pdfplumber.PDF
    if hasattr(pdf_input, "pages"):
        return _run(pdf_input)

    # Otherwise open it ourselves
    try:
        try:
            pdf_input.seek(0)
        except Exception:
            pass
        with pdfplumber.open(pdf_input) as pdf:
            return _run(pdf)
    except Exception as e:
        raise RuntimeError(f"Affin Bank parser failed for '{source_file}': {e}") from e
