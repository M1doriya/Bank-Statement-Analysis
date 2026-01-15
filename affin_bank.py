# affin_bank.py
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


# -----------------------------
# Regex
# -----------------------------
DATE_RE = re.compile(r"^(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{2,4})$")
MONEY_RE = re.compile(r"^(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}$")

HEADER_HINTS = ("DATE", "TARIKH", "DEBIT", "CREDIT", "BALANCE", "BAKI")
TOTAL_HINTS = ("TOTAL", "JUMLAH")
BF_HINTS = ("B/F", "BALANCE B/F", "BAKI B/F", "BAKI MULA", "OPENING")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _to_iso_date(tok: str) -> Optional[str]:
    m = DATE_RE.match(tok.strip())
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


def _money_to_float(tok: str) -> Optional[float]:
    if tok is None:
        return None
    s = tok.strip().replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


# -----------------------------
# Word extraction
# -----------------------------
def _words_from_pdf(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    words = page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
        extra_attrs=["x0", "x1", "top", "bottom"],
    ) or []
    out = []
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
    """
    OCR fallback: returns word boxes similar to pdfplumber.extract_words().
    Requires pytesseract.
    """
    if pytesseract is None:
        return []

    # Render page image
    img = page.to_image(resolution=250).original

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 6")
    n = len(data.get("text", []))

    out: List[Dict[str, Any]] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append(
            {
                "text": txt,
                "x0": float(x),
                "x1": float(x + w),
                "y0": float(y),
                "y1": float(y + h),
            }
        )
    return out


def _get_page_words(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    w = _words_from_pdf(page)
    if w:
        return w
    return _words_from_ocr(page)


# -----------------------------
# Row grouping + column detection
# -----------------------------
def _cluster_rows(words: List[Dict[str, Any]], y_tol: float = 3.0) -> List[List[Dict[str, Any]]]:
    if not words:
        return []

    # sort top->bottom, left->right
    words.sort(key=lambda r: (r["y0"], r["x0"]))

    rows: List[Dict[str, Any]] = []
    for w in words:
        placed = False
        for r in rows:
            if abs(w["y0"] - r["y"]) <= y_tol:
                r["items"].append(w)
                # update centroid
                r["y"] = (r["y"] * (len(r["items"]) - 1) + w["y0"]) / len(r["items"])
                placed = True
                break
        if not placed:
            rows.append({"y": w["y0"], "items": [w]})

    out: List[List[Dict[str, Any]]] = []
    for r in sorted(rows, key=lambda z: z["y"]):
        items = sorted(r["items"], key=lambda z: z["x0"])
        out.append(items)
    return out


def _row_text(row: List[Dict[str, Any]]) -> str:
    return _norm(" ".join(w["text"] for w in row))


def _detect_columns(rows: List[List[Dict[str, Any]]]) -> Optional[Dict[str, float]]:
    """
    Try to detect debit/credit/balance column x positions.
    Strategy:
      1) Find header row containing DEBIT/CREDIT/BALANCE; take their x centers.
      2) Fallback: cluster money-token x positions; rightmost cluster = balance.
    """
    # 1) header scan
    for row in rows[:20]:
        txt = _row_text(row).upper()
        if not any(h in txt for h in HEADER_HINTS):
            continue

        def find_x(keyword: str) -> Optional[float]:
            for w in row:
                if w["text"].upper() == keyword:
                    return (w["x0"] + w["x1"]) / 2.0
            return None

        debit_x = find_x("DEBIT")
        credit_x = find_x("CREDIT")
        balance_x = find_x("BALANCE")

        if debit_x and credit_x and balance_x:
            return {"debit_x": float(debit_x), "credit_x": float(credit_x), "balance_x": float(balance_x)}

    # 2) fallback clustering from money tokens
    money_words = []
    for row in rows:
        for w in row:
            if MONEY_RE.match(w["text"]):
                money_words.append(w)

    if len(money_words) < 5:
        return None

    xs = sorted((w["x0"] + w["x1"]) / 2.0 for w in money_words)

    # crude clustering by gaps
    clusters: List[List[float]] = []
    for x in xs:
        if not clusters or abs(x - clusters[-1][-1]) > 35:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    # we expect 2-3 money columns; rightmost is balance
    clusters.sort(key=lambda c: sum(c) / len(c))
    if not clusters:
        return None

    balance_x = sum(clusters[-1]) / len(clusters[-1])

    # if we have >=3 clusters, assume [debit, credit, balance]
    debit_x = credit_x = None
    if len(clusters) >= 3:
        debit_x = sum(clusters[-3]) / len(clusters[-3])
        credit_x = sum(clusters[-2]) / len(clusters[-2])
    elif len(clusters) == 2:
        # cannot know which is debit/credit reliably; we will use balance delta fallback later
        debit_x = None
        credit_x = None

    return {"debit_x": float(debit_x) if debit_x else -1.0,
            "credit_x": float(credit_x) if credit_x else -1.0,
            "balance_x": float(balance_x)}


def _pick_money_near(words: List[Dict[str, Any]], target_x: float, tol: float = 40.0) -> Optional[float]:
    if target_x <= 0:
        return None
    candidates = []
    for w in words:
        if not MONEY_RE.match(w["text"]):
            continue
        xc = (w["x0"] + w["x1"]) / 2.0
        if abs(xc - target_x) <= tol:
            val = _money_to_float(w["text"])
            if val is not None:
                candidates.append((abs(xc - target_x), -xc, val))
    if not candidates:
        return None
    candidates.sort()
    return float(candidates[0][2])


def _rightmost_money(words: List[Dict[str, Any]]) -> Optional[float]:
    money = []
    for w in words:
        if MONEY_RE.match(w["text"]):
            v = _money_to_float(w["text"])
            if v is not None:
                money.append((w["x1"], v))
    if not money:
        return None
    return float(max(money, key=lambda t: t[0])[1])


def _is_summary_row(row_txt_upper: str) -> bool:
    return any(t in row_txt_upper for t in TOTAL_HINTS)


def _is_bf_row(row_txt_upper: str) -> bool:
    return any(t in row_txt_upper for t in BF_HINTS)


# -----------------------------
# Main parser
# -----------------------------
def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    bank_name = "Affin Bank"
    transactions: List[Dict[str, Any]] = []

    # open robustly
    if isinstance(pdf_input, (bytes, bytearray)):
        pdf_bytes = bytes(pdf_input)
        pdf_f = BytesIO(pdf_bytes)
        pdf = pdfplumber.open(pdf_f)
    else:
        try:
            # streamlit upload
            if hasattr(pdf_input, "getvalue"):
                pdf = pdfplumber.open(BytesIO(pdf_input.getvalue()))
            else:
                pdf = pdfplumber.open(pdf_input)
        except Exception as e:
            raise RuntimeError(f"Affin parser: cannot open PDF: {e}") from e

    prev_balance: Optional[float] = None

    try:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = _get_page_words(page)
            if not words:
                continue

            rows = _cluster_rows(words, y_tol=3.5)
            if not rows:
                continue

            col = _detect_columns(rows)

            for row in rows:
                txt = _row_text(row)
                if not txt:
                    continue
                up = txt.upper()

                # Skip summary/totals blocks
                if _is_summary_row(up):
                    continue

                # Find date token (first token that matches DD/MM/YY)
                date_iso = None
                date_word_idx = None
                for i, w in enumerate(row[:6]):  # date is always early
                    d = _to_iso_date(w["text"])
                    if d:
                        date_iso = d
                        date_word_idx = i
                        break
                if not date_iso:
                    continue

                # Extract amounts by columns when possible
                balance = None
                debit = None
                credit = None

                if col:
                    balance = _pick_money_near(row, col["balance_x"], tol=55.0)
                    # for debit/credit columns, tighter tol to prevent cross-column pickup
                    debit = _pick_money_near(row, col["debit_x"], tol=45.0) if col["debit_x"] > 0 else None
                    credit = _pick_money_near(row, col["credit_x"], tol=45.0) if col["credit_x"] > 0 else None

                # Hard fallback: balance = rightmost money token on the row
                if balance is None:
                    balance = _rightmost_money(row)

                if balance is None:
                    continue

                # Opening/BF line: anchor only, do not emit
                if _is_bf_row(up):
                    prev_balance = float(balance)
                    continue

                # If debit/credit not confidently extracted, infer from balance delta
                if prev_balance is not None and (debit is None and credit is None):
                    delta = round(float(balance) - float(prev_balance), 2)
                    if delta > 0:
                        credit = delta
                        debit = 0.0
                    elif delta < 0:
                        debit = abs(delta)
                        credit = 0.0
                    else:
                        debit = 0.0
                        credit = 0.0
                else:
                    debit = float(debit or 0.0)
                    credit = float(credit or 0.0)

                    # sanity: if both are present due to OCR noise, use delta if available
                    if prev_balance is not None and debit > 0 and credit > 0:
                        delta = round(float(balance) - float(prev_balance), 2)
                        if delta > 0:
                            credit = float(delta)
                            debit = 0.0
                        elif delta < 0:
                            debit = float(abs(delta))
                            credit = 0.0
                        else:
                            debit = credit = 0.0

                # Build description: words between date and first money-column area
                # (simple heuristic: exclude money tokens and the date token)
                desc_parts = []
                for j, w in enumerate(row):
                    if date_word_idx is not None and j <= date_word_idx:
                        continue
                    if MONEY_RE.match(w["text"]):
                        continue
                    t = w["text"].strip("|")
                    if t:
                        desc_parts.append(t)
                description = _norm(" ".join(desc_parts))

                # Final anomaly gate: reject clearly broken balance jumps
                if prev_balance is not None:
                    # If delta is extremely large but row debit/credit is small, likely wrong balance
                    delta_abs = abs(float(balance) - float(prev_balance))
                    if delta_abs > 2_000_000:  # safety valve; tune if needed
                        # skip and do not update prev_balance
                        continue

                tx = {
                    "date": date_iso,
                    "description": description,
                    "debit": round(float(debit), 2),
                    "credit": round(float(credit), 2),
                    "balance": round(float(balance), 2),
                    "page": int(page_num),
                    "bank": bank_name,
                    "source_file": source_file or "",
                }
                transactions.append(tx)
                prev_balance = float(balance)

    finally:
        try:
            pdf.close()
        except Exception:
            pass

    return transactions
