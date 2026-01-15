# affin_bank.py
"""
Affin Bank statement parser (OCR-first, line-by-line, column-zoned).

Fixes:
- Prevents "page block" merges (no multi-line stitching into a single transaction).
- Stabilizes credit/debit/balance extraction using x-zone columns.
- Filters summary/footer/header lines (e.g., TOTAL CREDIT/DEBIT, INT/HIBAH/PROFIT).
- Avoids treating B/F, C/F, and statement totals as transactions.

Output schema:
  date, description, debit, credit, balance, page, bank, source_file, row_type
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
import pytesseract


# -----------------------------
# Regex
# -----------------------------
DATE_RE = re.compile(r"^(?P<d>\d{1,2})[/-](?P<m>\d{1,2})[/-](?P<y>\d{2,4})$")
MONEY_RE = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+(?:\.\d{2})$")

# Lines that should never become transactions
SKIP_LINE_RE = re.compile(
    r"\b("
    r"tarikh|uraian|wang\s+masuk|wang\s+keluar|baki|muka\s+surat|page\b|"
    r"member\s+of|pidm|figures\s+and\s+balances|the\s+figures\s+and\s+balances|"
    r"total\s+credit|total\s+debit|int/hibah/profit|s/charge|service\s+charge|"
    r"balance\s*b/f|balance\s*c/f|\bb/f\b|\bc/f\b"
    r")\b",
    re.I
)


# -----------------------------
# OCR cleanup
# -----------------------------
_OCR_CHAR_MAP = str.maketrans({
    "O": "0", "o": "0",
    "S": "5", "s": "5",
    "I": "1", "l": "1", "|": "1",
    "“": "", "”": "", "’": "", "`": "", "'": "",
})


def _parse_date_token(tok: str) -> Optional[str]:
    tok = (tok or "").strip()
    m = DATE_RE.match(tok)
    if not m:
        return None
    d = int(m.group("d"))
    mo = int(m.group("m"))
    y = m.group("y")
    y_int = 2000 + int(y) if len(y) == 2 else int(y)
    try:
        return datetime(y_int, mo, d).strftime("%Y-%m-%d")
    except Exception:
        return None


def _clean_money_token(tok: str) -> Optional[float]:
    if tok is None:
        return None
    s = str(tok).strip()
    if not s:
        return None

    # Normalize common OCR substitutions
    s = s.translate(_OCR_CHAR_MAP)

    # Remove spaces and non-money chars (keep digits, comma, dot)
    s = s.replace(" ", "")
    s = re.sub(r"[^0-9,\.]", "", s)
    if not s:
        return None

    # Fix occasional OCR comma issues like "11327,.00" -> "11327.00"
    s = s.replace(",.", ".").replace(".,", ".")

    if MONEY_RE.fullmatch(s):
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None

    return None


def _ocr_words_from_page(page: pdfplumber.page.Page, resolution: int = 300) -> Tuple[List[Dict[str, Any]], float]:
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

        # Keep moderate quality tokens; too low creates garbage joins
        if conf != -1.0 and conf < 40:
            continue

        x = float(data["left"][i])
        y = float(data["top"][i])
        w = float(data["width"][i])
        h = float(data["height"][i])
        words.append({"text": txt, "x0": x, "x1": x + w, "top": y, "bottom": y + h})

    return words, width


def _group_words_by_line(words: List[Dict[str, Any]], y_tol: float = 5.0) -> List[List[Dict[str, Any]]]:
    """
    Group words into lines by 'top' coordinate.
    Smaller y_tol reduces accidental line merging (important for your failures).
    """
    lines: List[List[Dict[str, Any]]] = []
    for w in words:
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


def _line_text(line: List[Dict[str, Any]]) -> str:
    return " ".join((w.get("text") or "").strip() for w in line if (w.get("text") or "").strip()).strip()


def _extract_line_fields(
    line: List[Dict[str, Any]],
    page_width: float,
) -> Tuple[Optional[str], str, float, float, Optional[float]]:
    """
    Split a single line into:
      date | description | credit | debit | balance
    using fixed x-zones based on page width.

    These proportions are robust for Affin's tabular layout and avoid global clustering drift.
    """
    # Zones (tuned for typical Affin statement layout):
    # date: 0% - 18%
    # desc: 18% - 62%
    # credit: 62% - 75%
    # debit: 75% - 87%
    # balance: 87% - 100%
    x_date_end = page_width * 0.18
    x_desc_end = page_width * 0.62
    x_credit_end = page_width * 0.75
    x_debit_end = page_width * 0.87

    date_iso: Optional[str] = None
    desc_tokens: List[str] = []
    credit_vals: List[float] = []
    debit_vals: List[float] = []
    balance_vals: List[float] = []

    for w in line:
        txt = (w.get("text") or "").strip()
        if not txt:
            continue

        x0 = float(w.get("x0", 0.0))

        # Date zone
        if x0 <= x_date_end:
            d = _parse_date_token(txt)
            if d:
                date_iso = d
                continue
            # non-date junk in date zone -> ignore
            continue

        # Money?
        mv = _clean_money_token(txt)
        if mv is not None:
            if x0 <= x_credit_end:
                credit_vals.append(mv)
            elif x0 <= x_debit_end:
                debit_vals.append(mv)
            else:
                balance_vals.append(mv)
            continue

        # Description zone: keep only if not obvious noise
        if x0 <= x_desc_end:
            if not SKIP_LINE_RE.search(txt):
                desc_tokens.append(txt)

    description = " ".join(desc_tokens).strip()

    # In table rows, there should be max one amount per column; OCR might split it, so sum within column.
    credit = round(sum(credit_vals), 2) if credit_vals else 0.0
    debit = round(sum(debit_vals), 2) if debit_vals else 0.0
    balance = round(balance_vals[-1], 2) if balance_vals else None

    return date_iso, description, debit, credit, balance


def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    bank_name = "Affin Bank"

    def _run(pdf: pdfplumber.PDF) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for page_idx, page in enumerate(pdf.pages, start=1):
            words, page_width = _ocr_words_from_page(page, resolution=300)
            if not words:
                continue

            lines = _group_words_by_line(words, y_tol=5.0)

            for ln in lines:
                raw = _line_text(ln)
                if not raw:
                    continue

                # Fast skip for obvious headers/footers/summary lines
                if SKIP_LINE_RE.search(raw):
                    continue

                date_iso, desc, debit, credit, balance = _extract_line_fields(ln, page_width)

                if not date_iso:
                    continue

                # Must have at least an amount or a balance to be meaningful
                if debit == 0.0 and credit == 0.0 and balance is None:
                    continue

                # Sanity rule: a single row should not have BOTH debit and credit.
                # If it happens, it is almost certainly a column/OCR issue; keep the larger as the actual amount.
                if debit != 0.0 and credit != 0.0:
                    if debit >= credit:
                        credit = 0.0
                    else:
                        debit = 0.0

                out.append({
                    "date": date_iso,
                    "description": desc,
                    "debit": float(debit),
                    "credit": float(credit),
                    "balance": balance,
                    "page": int(page_idx),
                    "bank": bank_name,
                    "source_file": source_file,
                    "row_type": "transaction",
                })

        return out

    # Accept already-opened pdfplumber.PDF or file-like
    if hasattr(pdf_input, "pages"):
        return _run(pdf_input)

    try:
        try:
            pdf_input.seek(0)
        except Exception:
            pass
        with pdfplumber.open(pdf_input) as pdf:
            return _run(pdf)
    except Exception as e:
        raise RuntimeError(f"Affin Bank parser failed for '{source_file}': {e}") from e
