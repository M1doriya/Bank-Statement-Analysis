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
# allow ".00" (OCR sometimes drops leading zero); normalize before matching
MONEY_RE = re.compile(r"^(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}$")

HEADER_HINTS = ("DATE", "TARIKH", "DEBIT", "CREDIT", "BALANCE", "BAKI")
TOTAL_HINTS = ("TOTAL", "JUMLAH")
BF_HINTS = ("B/F", "BALANCE B/F", "BAKI B/F", "BAKI MULA", "OPENING", "B/F BALANCE")


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
    if not s:
        return None
    # normalize ".00" -> "0.00"
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
    if pytesseract is None:
        return []

    # Render page
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
# Row grouping
# -----------------------------
def _cluster_rows(words: List[Dict[str, Any]], y_tol: float = 3.5) -> List[List[Dict[str, Any]]]:
    if not words:
        return []
    words.sort(key=lambda r: (r["y0"], r["x0"]))

    buckets: List[Dict[str, Any]] = []
    for w in words:
        placed = False
        for b in buckets:
            if abs(w["y0"] - b["y"]) <= y_tol:
                b["items"].append(w)
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


def _row_has_date(row: List[Dict[str, Any]]) -> bool:
    for w in row[:7]:
        if _to_iso_date(w["text"]):
            return True
    return False


# -----------------------------
# Column detection (FIXED for Apr/Aug)
# -----------------------------
def _detect_columns(rows: List[List[Dict[str, Any]]]) -> Optional[Dict[str, float]]:
    """
    Detect debit/credit/balance column x-centers.

    Priority:
      1) header row containing DEBIT/CREDIT/BALANCE
      2) fallback: cluster money x-centers BUT only using candidate transaction rows (rows with a date)
         (this is the key fix for April/August where summary blocks pollute clustering)
    """
    # 1) header scan
    for row in rows[:40]:
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

    # 2) fallback clustering from ONLY date-rows
    money_xs: List[float] = []
    for row in rows:
        if not _row_has_date(row):
            continue
        if _is_summary_row(_row_text(row).upper()):
            continue

        for w in row:
            t = (w["text"] or "").strip()
            if not t:
                continue
            chk = ("0" + t) if t.startswith(".") else t
            if MONEY_RE.match(chk):
                xc = (w["x0"] + w["x1"]) / 2.0
                money_xs.append(float(xc))

    if len(money_xs) < 10:
        return None

    money_xs.sort()

    # cluster by x-gap
    clusters: List[List[float]] = []
    for x in money_xs:
        if not clusters or abs(x - clusters[-1][-1]) > 35:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    clusters.sort(key=lambda c: sum(c) / len(c))
    if not clusters:
        return None

    # rightmost cluster is balance
    balance_x = sum(clusters[-1]) / len(clusters[-1])

    debit_x = credit_x = -1.0
    if len(clusters) >= 3:
        debit_x = sum(clusters[-3]) / len(clusters[-3])
        credit_x = sum(clusters[-2]) / len(clusters[-2])

    return {"debit_x": float(debit_x), "credit_x": float(credit_x), "balance_x": float(balance_x)}


def _pick_money_near(words: List[Dict[str, Any]], target_x: float, tol: float = 45.0) -> Optional[float]:
    if target_x <= 0:
        return None
    best: Optional[Tuple[float, float]] = None  # (dist, val)
    for w in words:
        t = (w["text"] or "").strip()
        chk = ("0" + t) if t.startswith(".") else t
        if not MONEY_RE.match(chk):
            continue
        xc = (w["x0"] + w["x1"]) / 2.0
        dist = abs(xc - target_x)
        if dist <= tol:
            val = _money_to_float(t)
            if val is None:
                continue
            if best is None or dist < best[0]:
                best = (dist, float(val))
    return best[1] if best else None


def _rightmost_money(words: List[Dict[str, Any]]) -> Optional[float]:
    best: Optional[Tuple[float, float]] = None  # (x1, value)
    for w in words:
        t = (w["text"] or "").strip()
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
# Public entry point
# -----------------------------
def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    """
    Works with your app.py which passes an already-open pdfplumber.PDF.

    Accepts:
      - pdfplumber.PDF (already opened), OR
      - bytes/path/file-like (standalone)
    """
    bank_name = "Affin Bank"
    transactions: List[Dict[str, Any]] = []

    # If app.py already opened it with pdfplumber
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

                if _is_summary_row(up):
                    continue

                # Identify date
                date_iso = None
                date_word_idx = None
                for i, w in enumerate(row[:7]):
                    d = _to_iso_date(w["text"])
                    if d:
                        date_iso = d
                        date_word_idx = i
                        break
                if not date_iso:
                    continue

                # Read amounts
                balance = None
                debit = None
                credit = None

                if col:
                    balance = _pick_money_near(row, col["balance_x"], tol=65.0)
                    if col["debit_x"] > 0:
                        debit = _pick_money_near(row, col["debit_x"], tol=50.0)
                    if col["credit_x"] > 0:
                        credit = _pick_money_near(row, col["credit_x"], tol=50.0)

                if balance is None:
                    balance = _rightmost_money(row)
                if balance is None:
                    continue

                # Opening/BF row anchors the running balance
                if _is_bf_row(up):
                    prev_balance = float(balance)
                    continue

                # Build description (non-money tokens after date)
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

                # Infer debit/credit by delta if we didn't reliably extract them
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

                    # If OCR picked both, reconcile with delta (more reliable)
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

                # Safety valve: skip absurd jumps (almost always OCR row-mix)
                if prev_balance is not None:
                    if abs(float(balance) - float(prev_balance)) > 2_000_000:
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
