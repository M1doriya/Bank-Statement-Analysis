# affin_bank.py
"""
Affin Bank statement parser with OCR fallback.

Why needed:
- Many Affin PDFs are image-based (scanned). pdfplumber.extract_text() returns empty.
- This parser first tries normal text extraction; if empty, it OCRs each page image and reconstructs rows.

Output keys (consistent with your system):
  date, description, debit, credit, balance, page, bank, source_file
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

# OCR deps (already in your environment via requirements/system packages)
import pytesseract


# -----------------------------
# Regex
# -----------------------------
DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")     # 1/8/25 or 01/08/2025
MONEY_RE = re.compile(r"^[\d,]+\.\d{2}$")              # 1,234.56


# -----------------------------
# Helpers
# -----------------------------
def _clean_amount_token(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    # OCR sometimes yields ".00" as "00" or "0.00" etc. We only accept strict currency like 123.45
    # but allow leading comma/spacing issues.
    s = s.replace(" ", "")
    # common OCR junk
    s = s.replace("O", "0").replace("o", "0")  # sometimes O mistaken for 0
    s = s.replace("’", "").replace("`", "").replace("'", "")

    # normalize comma thousands
    s2 = s.replace(",", "")

    # allow "-.00" or similar? Affin format usually no negatives.
    if s2.startswith("-"):
        s2 = s2[1:]
        sign = -1.0
    else:
        sign = 1.0

    # accept standard money
    if re.fullmatch(r"\d+(\.\d{2})", s2):
        try:
            return sign * float(s2)
        except Exception:
            return None

    # if OCR gave something like "156325.31" it's okay, handled above;
    # if it gave "156,325.31" it's okay after replace.
    # Otherwise reject.
    return None


def _parse_date_token(tok: str) -> Optional[str]:
    tok = (tok or "").strip()
    if not DATE_RE.fullmatch(tok):
        return None

    # Support dd/mm/yy or dd/mm/yyyy
    parts = tok.split("/")
    if len(parts) != 3:
        return None

    d, m, y = parts
    if len(y) == 2:
        fmt = "%d/%m/%y"
    else:
        fmt = "%d/%m/%Y"

    try:
        return datetime.strptime(tok, fmt).strftime("%Y-%m-%d")
    except Exception:
        return None


def _group_words_by_line(words: List[Dict[str, Any]], y_tol: float = 6.0) -> List[List[Dict[str, Any]]]:
    """
    Group OCR words into lines based on their 'top' coordinate.
    """
    lines: List[List[Dict[str, Any]]] = []
    for w in words:
        if not w.get("text"):
            continue
        placed = False
        for ln in lines:
            if abs(ln[0]["top"] - w["top"]) <= y_tol:
                ln.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    for ln in lines:
        ln.sort(key=lambda x: x.get("x0", 0.0))
    lines.sort(key=lambda ln: ln[0].get("top", 0.0))
    return lines


def _ocr_words_from_page(page: pdfplumber.page.Page) -> Tuple[List[Dict[str, Any]], float]:
    """
    Run OCR and return word boxes with x0, top, text, and page width.
    """
    # Render page to image (PIL)
    # Increase resolution for better OCR reliability
    im = page.to_image(resolution=300).original
    width = float(im.size[0])

    data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)

    words: List[Dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data.get("conf", [])[i])
        except Exception:
            conf = -1.0

        # ignore extremely low confidence garbage; keep modest threshold
        if conf != -1.0 and conf < 35:
            continue

        x = float(data["left"][i])
        y = float(data["top"][i])
        w = float(data["width"][i])
        h = float(data["height"][i])
        words.append(
            {"text": txt, "x0": x, "x1": x + w, "top": y, "bottom": y + h}
        )

    return words, width


def _detect_amount_columns(lines: List[List[Dict[str, Any]]], page_width: float) -> Tuple[float, float, float]:
    """
    Detect approximate x positions for:
      credit ("Wang Masuk"),
      debit  ("Wang Keluar"),
      balance ("Baki")

    If OCR header detection fails, use robust fallback positions based on page width.
    """
    credit_x = debit_x = balance_x = None

    # Search first ~30 lines for header hints
    scan_lines = lines[:30] if len(lines) > 30 else lines
    for ln in scan_lines:
        texts = [w["text"].lower() for w in ln]
        for w in ln:
            t = w["text"].lower()
            if t == "masuk":
                credit_x = float(w["x0"])
            elif t == "keluar":
                debit_x = float(w["x0"])
            elif t == "baki":
                balance_x = float(w["x0"])

        # If we got all, stop
        if credit_x and debit_x and balance_x:
            break

    # Fallback to proportional columns if header isn't detected
    # Based on your sample layout: amount columns occupy right half of page.
    if credit_x is None:
        credit_x = page_width * 0.62
    if debit_x is None:
        debit_x = page_width * 0.76
    if balance_x is None:
        balance_x = page_width * 0.88

    return float(credit_x), float(debit_x), float(balance_x)


def _assign_money_to_columns(money_items: List[Tuple[float, float]], credit_x: float, debit_x: float, balance_x: float) -> Tuple[float, float, Optional[float]]:
    """
    money_items: list of (x0, value)
    returns credit, debit, balance
    """
    credit = 0.0
    debit = 0.0
    balance: Optional[float] = None

    # Heuristic: the rightmost money is usually balance.
    if money_items:
        # Sort by x
        money_items_sorted = sorted(money_items, key=lambda t: t[0])

        # Candidate balance: rightmost
        bx, bval = money_items_sorted[-1]
        # Only accept as balance if it's near the balance column
        if abs(bx - balance_x) <= max(35.0, 0.05 * balance_x):
            balance = float(bval)

        # For remaining amounts, assign to credit/debit via nearest column
        for x0, val in money_items_sorted[:-1]:
            dist_c = abs(x0 - credit_x)
            dist_d = abs(x0 - debit_x)
            dist_b = abs(x0 - balance_x)

            # if it is actually near the balance column, treat it as balance too
            if dist_b < min(dist_c, dist_d) and dist_b <= 35:
                balance = float(val)
                continue

            if dist_c <= dist_d:
                credit += float(val)
            else:
                debit += float(val)

    return round(credit, 2), round(debit, 2), (round(balance, 2) if balance is not None else None)


def _extract_transactions_from_lines(
    lines: List[List[Dict[str, Any]]],
    credit_x: float,
    debit_x: float,
    balance_x: float,
    page_idx: int,
    bank_name: str,
    source_file: str,
) -> List[Dict[str, Any]]:
    """
    Build transactions by:
    - starting a new transaction when a line begins with a date token
    - accumulating description across following lines until next date
    - collecting money tokens per transaction and assigning to columns
    """
    txs: List[Dict[str, Any]] = []

    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln:
            i += 1
            continue

        # Identify first date token in line
        date_iso = None
        date_x = None
        for w in ln[:4]:  # date is typically very early in the line
            maybe = _parse_date_token(w["text"])
            if maybe:
                date_iso = maybe
                date_x = float(w["x0"])
                break

        if not date_iso:
            i += 1
            continue

        # Gather description + money from this line and subsequent continuation lines
        desc_tokens: List[str] = []
        money_items: List[Tuple[float, float]] = []

        def consume_line(line_words: List[Dict[str, Any]]):
            nonlocal desc_tokens, money_items
            for w in line_words:
                txt = (w.get("text") or "").strip()
                if not txt:
                    continue

                # Skip the date token itself
                if _parse_date_token(txt):
                    continue

                # Money token?
                val = _clean_amount_token(txt)
                if val is not None:
                    money_items.append((float(w.get("x0", 0.0)), float(val)))
                    continue

                # Otherwise description (ignore obvious headers)
                low = txt.lower()
                if low in {"tarikh", "uraian", "wang", "masuk", "keluar", "baki", "page"}:
                    continue
                desc_tokens.append(txt)

        consume_line(ln)

        j = i + 1
        while j < len(lines):
            next_ln = lines[j]
            # stop if next line starts a new dated transaction
            is_next_date = False
            for w in next_ln[:4]:
                if _parse_date_token(w.get("text", "")):
                    is_next_date = True
                    break
            if is_next_date:
                break
            consume_line(next_ln)
            j += 1

        description = " ".join(desc_tokens).strip()

        credit, debit, balance = _assign_money_to_columns(money_items, credit_x, debit_x, balance_x)

        # Skip rows that look like statement headers / no-amount lines
        if credit == 0.0 and debit == 0.0 and balance is None:
            i = j
            continue

        txs.append(
            {
                "date": date_iso,
                "description": description,
                "debit": float(debit),
                "credit": float(credit),
                "balance": balance,
                "page": int(page_idx),
                "bank": bank_name,
                "source_file": source_file or "",
                "row_type": "transaction",
            }
        )

        i = j

    return txs


# -----------------------------
# Main entry
# -----------------------------
def parse_affin_bank(pdf_input, source_file: str = "") -> List[Dict[str, Any]]:
    """
    Accepts either:
      - a pdfplumber.PDF object (preferred, how your app calls it), OR
      - a file-like object / streamlit UploadedFile

    Returns list[dict] with keys:
      date, description, debit, credit, balance, page, bank, source_file
    """
    bank_name = "Affin Bank"
    out: List[Dict[str, Any]] = []

    def parse_pdf(pdf: pdfplumber.PDF):
        nonlocal out
        for page_idx, page in enumerate(pdf.pages, start=1):
            # 1) Try native text extraction first
            text = (page.extract_text() or "").strip()
            if text:
                # If you later have text-based Affin PDFs, you can add a text-regex parser here.
                # For now, we proceed with OCR anyway because the statement is columnar and
                # OCR-based line reconstruction is more consistent across templates.
                pass

            # 2) OCR fallback (works for scanned PDFs like your sample)
            words, page_width = _ocr_words_from_page(page)
            if not words:
                continue

            lines = _group_words_by_line(words, y_tol=6.0)
            credit_x, debit_x, balance_x = _detect_amount_columns(lines, page_width)

            page_txs = _extract_transactions_from_lines(
                lines=lines,
                credit_x=credit_x,
                debit_x=debit_x,
                balance_x=balance_x,
                page_idx=page_idx,
                bank_name=bank_name,
                source_file=source_file,
            )
            out.extend(page_txs)

    # If the caller already opened pdfplumber
    if hasattr(pdf_input, "pages"):
        parse_pdf(pdf_input)
        return out

    # Else open it ourselves
    try:
        try:
            pdf_input.seek(0)
        except Exception:
            pass

        with pdfplumber.open(pdf_input) as pdf:
            parse_pdf(pdf)

    except Exception as e:
        raise RuntimeError(f"Affin Bank parser failed for '{source_file}': {e}") from e

    return out
