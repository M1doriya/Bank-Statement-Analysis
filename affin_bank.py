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
MONEY_RE = re.compile(r"^(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}$")  # requires leading digit

HEADER_HINTS = ("DATE", "TARIKH", "DEBIT", "CREDIT", "BALANCE", "BAKI")
TOTAL_HINTS = ("TOTAL", "JUMLAH")
BF_HINTS = ("B/F", "BALANCE B/F", "BAKI B/F", "BAKI MULA", "OPENING", "B/F BALANCE")

# Skip-only noise (do not over-aggressively skip because it may drop tx rows)
NOISE_HINTS = ("PAGE", "STATEMENT", "PENYATA", "ACCOUNT", "AFFIN", "PIDM", "MEMBER")


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
    # Some OCRs can output ".00"; normalize to "0.00"
    if s.startswith("."):
        s = "0" + s
    try:
        return float(s)
    except Exception:
        return None


def _is_summary_row(row_txt_upper: str) -> bool:
    return any(t in row_txt_upper for t in TOTAL_HINTS)


def _is_bf_row(row_txt_upper: str) -> bool:
    return any(t in row_txt_upper for t in BF_HINTS)


def _is_noise_row(row_txt_upper: str) -> bool:
    return any(t in row_txt_upper for t in NOISE_HINTS)


# -----------------------------
# Word extraction
# -----------------------------
def _words_from_pdf(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    words = page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
        extra_attrs=["x0", "x1", "top", "bottom"],
    ) or []
    out: List[Dict[str, Any]] = []
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
    """
    Group words into rows by similar y0 (top).
    """
    if not words:
        return []

    # sort top->bottom, left->right
    words.sort(key=lambda r: (r["y0"], r["x0"]))

    buckets: List[Dict[str, Any]] = []
    for w in words:
        placed = False
        for b in buckets:
            if abs(w["y0"] - b["y"]) <= y_tol:
                b["items"].append(w)
                # update centroid
                b["y"] = (b["y"] * (len(b["items"]) - 1) + w["y0"]) / len(b["items"])
                placed = True
                break
        if not placed:
            buckets.append({"y": w["y0"], "items": [w]})

    out: List[List[Dict[str, Any]]] = []
    for b in sorted(buckets, key=lambda z: z["y"]):
        out.append(sorted(b["items"], key=lambda z: z["x0"]))
    return out


def _row_text(row: List[Dict[str, Any]]) -> str:
    return _norm(" ".join(w["text"] for w in row))


def _detect_columns(rows: List[List[Dict[str, Any]]]) -> Optional[Dict[str, float]]:
    """
    Try to detect debit/credit/balance column x positions.

    Strategy:
      1) Find a header row containing DEBIT/CREDIT/BALANCE and use their x centers.
      2) Fallback: cluster money-token x positions; rightmost cluster = balance.
    """
    # 1) header scan
    for row in rows[:30]:
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

    # 2) fallback from money tokens
    money_words: List[Dict[str, Any]] = []
    for row in rows:
        for w in row:
            t = w["text"]
            # allow ".00" OCR by normalizing for the check
            chk = ("0" + t) if t.startswith(".") else t
            if MONEY_RE.match(chk):
                money_words.append(w)

    if len(money_words) < 8:
        return None

    xs = sorted((w["x0"] + w["x1"]) / 2.0 for w in money_words)

    clusters: List[List[float]] = []
    for x in xs:
        if not clusters or abs(x - clusters[-1][-1]) > 35:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    clusters.sort(key=lambda c: sum(c) / len(c))
    if not clusters:
        return None

    balance_x = sum(clusters[-1]) / len(clusters[-1])

    # if we have >=3 clusters, assume [debit, credit, balance]
    debit_x = credit_x = None
    if len(clusters) >= 3:
        debit_x = sum(clusters[-3]) / len(clusters[-3])
        credit_x = sum(clusters[-2]) / len(clusters[-2])

    return {
        "debit_x": float(debit_x) if debit_x else -1.0,
        "credit_x": float(credit_x) if credit_x else -1.0,
        "balance_x": float(balance_x),
    }


def _pick_money_near(words: List[Dict[str, Any]], target_x: float, tol: float = 45.0) -> Optional[float]:
    if target_x <= 0:
        return None
    candidates: List[Tuple[float, float]] = []  # (distance, value)
    for w in words:
        t = w["text"]
        chk = ("0" + t) if t.startswith(".") else t
        if not MONEY_RE.match(chk):
            continue
        xc = (w["x0"] + w["x1"]) / 2.0
        if abs(xc - target_x) <= tol:
            val = _money_to_float(t)
            if val is not None:
                candidates.append((abs(xc - target_x), float(val)))
    if not candidates:
        return None
    candidates.sort(key=lambda z: z[0])
    return candidates[0][1]


def _rightmost_money(words: List[Dict[str, Any]]) -> Optional[float]:
    best: Optional[Tuple[float, float]] = None  # (x1, value)
    for w in words:
        t = w["text"]
        chk = ("0" + t) if t.startswith(".") else t
        if MONEY_RE.match(chk):
            v = _money_to_float(t)
            if v is None:
                continue
            x1 = float(w["x1"])
            if best is None or x1 > best[0]:
                best = (x1, float(v))
    return best[1] if best else None


# -----------------------------
# Public entry point (used by app.py)
# -----------------------------
def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    """
    IMPORTANT: In your app.py, _parse_with_pdfplumber already does pdfplumber.open(...)
    and passes a pdfplumber.PDF object into this function.

    Therefore this function must accept BOTH:
      - pdfplumber.PDF (already opened)
      - bytes/path/file-like (standalone usage)
    """
    bank_name = "Affin Bank"
    transactions: List[Dict[str, Any]] = []

    # ------------------------------------------------------
    # Detect if we were given an already-open pdfplumber.PDF
    # ------------------------------------------------------
    if hasattr(pdf_input, "pages") and hasattr(pdf_input, "close"):
        pdf = pdf_input
        should_close = False
    else:
        should_close = True
        try:
            if isinstance(pdf_input, (bytes, bytearray)):
                pdf = pdfplumber.open(BytesIO(bytes(pdf_input)))
            elif hasattr(pdf_input, "getvalue"):
                # streamlit UploadedFile
                pdf = pdfplumber.open(BytesIO(pdf_input.getvalue()))
            else:
                # file path or file-like
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

                # Skip obvious summaries
                if _is_summary_row(up):
                    continue

                # Light noise skip (only if row has no date-like token)
                # We do not skip aggressively because it may drop real transactions.
                if _is_noise_row(up) and not any(_to_iso_date(w["text"]) for w in row[:6]):
                    continue

                # Detect date near the start of the row
                date_iso = None
                date_word_idx = None
                for i, w in enumerate(row[:6]):
                    d = _to_iso_date(w["text"])
                    if d:
                        date_iso = d
                        date_word_idx = i
                        break
                if not date_iso:
                    continue

                # Amounts by column
                balance = None
                debit = None
                credit = None

                if col:
                    balance = _pick_money_near(row, col["balance_x"], tol=60.0)
                    if col["debit_x"] > 0:
                        debit = _pick_money_near(row, col["debit_x"], tol=45.0)
                    if col["credit_x"] > 0:
                        credit = _pick_money_near(row, col["credit_x"], tol=45.0)

                # Fallback balance
                if balance is None:
                    balance = _rightmost_money(row)
                if balance is None:
                    continue

                # Opening/BF: anchor only
                if _is_bf_row(up):
                    prev_balance = float(balance)
                    continue

                # Infer debit/credit if missing
                if prev_balance is not None and debit is None and credit is None:
                    delta = round(float(balance) - float(prev_balance), 2)
                    if delta > 0:
                        credit = delta
                        debit = 0.0
                    elif delta < 0:
                        debit = abs(delta)
                        credit = 0.0
                    else:
                        debit = credit = 0.0
                else:
                    debit = float(debit or 0.0)
                    credit = float(credit or 0.0)

                    # If OCR accidentally picked both, reconcile using delta when possible
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

                # Description: all non-money tokens after the date
                desc_parts: List[str] = []
                for j, w in enumerate(row):
                    if date_word_idx is not None and j <= date_word_idx:
                        continue
                    t = (w["text"] or "").strip().strip("|")
                    if not t:
                        continue
                    chk = ("0" + t) if t.startswith(".") else t
                    if MONEY_RE.match(chk):
                        continue
                    desc_parts.append(t)

                description = _norm(" ".join(desc_parts))

                # Safety valve: skip absurd balance jumps (usually OCR row-mix)
                if prev_balance is not None:
                    delta_abs = abs(float(balance) - float(prev_balance))
                    if delta_abs > 2_000_000:
                        # Do not update prev_balance
                        continue

                transactions.append(
                    {
                        "date": date_iso,
                        "description": description,
                        "debit": round(float(debit), 2),
                        "credit": round(float(credit), 2),
                        "balance": round(float(balance), 2),
                        "page": int(page_num),
                        "bank": bank_name,
                        "source_file": source_file or "",
                    }
                )

                prev_balance = float(balance)

    finally:
        if should_close:
            try:
                pdf.close()
            except Exception:
                pass

    return transactions
