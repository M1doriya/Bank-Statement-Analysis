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


# =========================================================
# Regex / constants
# =========================================================
DATE_RE = re.compile(r"^(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{2,4})$")

# Allow OCR ".00" by normalizing before matching
MONEY_RE = re.compile(r"^(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}$")

HEADER_HINTS = ("DATE", "TARIKH", "DEBIT", "CREDIT", "BALANCE", "BAKI")
TOTAL_HINTS = ("TOTAL", "JUMLAH")
BF_HINTS = ("B/F", "BALANCE B/F", "BAKI B/F", "BAKI MULA", "OPENING", "B/F BALANCE")

# Some statements contain money numbers in headers/boxes; avoid using those rows for clustering
NON_TX_HINTS = (
    "ACCOUNT", "NO.", "STATEMENT", "PENYATA", "PAGE", "MEMBER", "PIDM", "AFFIN",
    "BRANCH", "ADDRESS", "CUSTOMER", "CIF", "DATE OF", "PERIOD"
)


# =========================================================
# Helpers
# =========================================================
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
    s = (tok or "").strip().replace(",", "")
    if not s:
        return None
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


def _looks_non_tx_row(row_txt_upper: str) -> bool:
    return any(t in row_txt_upper for t in NON_TX_HINTS)


def _is_money_token(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    chk = ("0" + t) if t.startswith(".") else t
    return bool(MONEY_RE.match(chk))


# =========================================================
# Word extraction (pdf text first, OCR fallback)
# =========================================================
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

    # Crop away most headers/footers to reduce OCR noise (common Affin layout)
    try:
        w, h = float(page.width), float(page.height)
        top = 120
        bottom = h - 60
        crop = page.crop((0, top, w, bottom))
        img = crop.to_image(resolution=250).original
    except Exception:
        img = page.to_image(resolution=250).original

    data = pytesseract.image_to_data(
        img, output_type=pytesseract.Output.DICT, config="--psm 6"
    )
    n = len(data.get("text", []))

    out: List[Dict[str, Any]] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        x, y, ww, hh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append(
            {
                "text": txt,
                "x0": float(x),
                "x1": float(x + ww),
                "y0": float(y),
                "y1": float(y + hh),
            }
        )
    return out


def _get_page_words(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    w = _words_from_pdf(page)
    if w:
        return w
    return _words_from_ocr(page)


# =========================================================
# Row grouping
# =========================================================
def _cluster_rows(words: List[Dict[str, Any]], y_tol: float = 3.5) -> List[Tuple[float, List[Dict[str, Any]]]]:
    """
    Returns list of (row_y, row_words) sorted top->bottom.
    """
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

    out: List[Tuple[float, List[Dict[str, Any]]]] = []
    for b in sorted(buckets, key=lambda z: z["y"]):
        row_words = sorted(b["items"], key=lambda z: z["x0"])
        out.append((float(b["y"]), row_words))
    return out


def _row_text(row_words: List[Dict[str, Any]]) -> str:
    return _norm(" ".join(w["text"] for w in row_words))


def _row_has_date(row_words: List[Dict[str, Any]]) -> bool:
    for w in row_words[:7]:
        if _to_iso_date(w["text"]):
            return True
    return False


# =========================================================
# Column detection (key fix for Apr/Aug: use only date-rows)
# =========================================================
def _detect_columns(rows: List[Tuple[float, List[Dict[str, Any]]]]) -> Optional[Dict[str, float]]:
    # 1) Header scan
    for _, rw in rows[:40]:
        txtu = _row_text(rw).upper()
        if not any(h in txtu for h in HEADER_HINTS):
            continue

        def find_x(keyword: str) -> Optional[float]:
            for w in rw:
                if w["text"].upper() == keyword:
                    return (w["x0"] + w["x1"]) / 2.0
            return None

        debit_x = find_x("DEBIT")
        credit_x = find_x("CREDIT")
        balance_x = find_x("BALANCE")
        if debit_x and credit_x and balance_x:
            return {"debit_x": float(debit_x), "credit_x": float(credit_x), "balance_x": float(balance_x)}

    # 2) Fallback clustering (ONLY candidate transaction rows = rows with date)
    money_xs: List[float] = []
    for _, rw in rows:
        txtu = _row_text(rw).upper()
        if not _row_has_date(rw):
            continue
        if _is_summary_row(txtu):
            continue
        if _looks_non_tx_row(txtu):
            continue

        for w in rw:
            if _is_money_token(w["text"]):
                xc = (w["x0"] + w["x1"]) / 2.0
                money_xs.append(float(xc))

    if len(money_xs) < 10:
        return None

    money_xs.sort()

    clusters: List[List[float]] = []
    for x in money_xs:
        if not clusters or abs(x - clusters[-1][-1]) > 35:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    clusters.sort(key=lambda c: sum(c) / len(c))
    if not clusters:
        return None

    balance_x = sum(clusters[-1]) / len(clusters[-1])

    debit_x = credit_x = -1.0
    if len(clusters) >= 3:
        debit_x = sum(clusters[-3]) / len(clusters[-3])
        credit_x = sum(clusters[-2]) / len(clusters[-2])

    return {"debit_x": float(debit_x), "credit_x": float(credit_x), "balance_x": float(balance_x)}


def _pick_money_near(row_words: List[Dict[str, Any]], target_x: float, tol: float) -> Optional[float]:
    if target_x <= 0:
        return None
    best: Optional[Tuple[float, float]] = None
    for w in row_words:
        if not _is_money_token(w["text"]):
            continue
        xc = (w["x0"] + w["x1"]) / 2.0
        dist = abs(xc - target_x)
        if dist > tol:
            continue
        val = _money_to_float(w["text"])
        if val is None:
            continue
        if best is None or dist < best[0]:
            best = (dist, float(val))
    return best[1] if best else None


def _rightmost_money(row_words: List[Dict[str, Any]]) -> Optional[float]:
    best: Optional[Tuple[float, float]] = None  # (x1, val)
    for w in row_words:
        if not _is_money_token(w["text"]):
            continue
        val = _money_to_float(w["text"])
        if val is None:
            continue
        x1 = float(w["x1"])
        if best is None or x1 > best[0]:
            best = (x1, float(val))
    return best[1] if best else None


# =========================================================
# Post-parse validation + repair (fixes Streamlit wrong totals)
# =========================================================
def _is_garbage_row(tx: Dict[str, Any]) -> bool:
    desc = (tx.get("description") or "").strip()
    debit = float(tx.get("debit") or 0.0)
    credit = float(tx.get("credit") or 0.0)

    # empty/fragment descriptions and no amounts
    if debit == 0.0 and credit == 0.0:
        if desc == "":
            return True
        if len(desc) <= 3 and all(ch in ".:-|/\\ " or ch.isdigit() for ch in desc):
            return True
        if desc.replace(".", "").replace("-", "").isdigit():
            return True

    # very short non-informative desc + missing balance usually indicates header fragments
    if len(desc) <= 2 and (tx.get("balance") is None):
        return True

    return False


def _validate_and_repair(transactions: List[Dict[str, Any]], tol: float = 0.05) -> List[Dict[str, Any]]:
    """
    Ensures:
      prev_balance + credit - debit ≈ balance
    If not, tries swapping debit/credit to see if it reconciles.
    Invalid rows get balance set to None (so min/max/ending don't get poisoned).
    Obvious garbage rows dropped.
    """
    # Sort with best available keys for correct sequencing
    transactions.sort(key=lambda x: (x.get("date", ""), int(x.get("page", 0)), float(x.get("_y", 0.0)), int(x.get("_seq", 0))))

    cleaned: List[Dict[str, Any]] = []
    prev_bal: Optional[float] = None

    for tx in transactions:
        if _is_garbage_row(tx):
            continue

        debit = float(tx.get("debit") or 0.0)
        credit = float(tx.get("credit") or 0.0)
        bal = tx.get("balance", None)
        bal = float(bal) if bal is not None else None

        # No balance => can't validate; keep but do not affect min/max/ending
        if bal is None:
            tx["balance"] = None
            cleaned.append(tx)
            continue

        # If this row is first anchor (no prev), accept it as anchor
        if prev_bal is None:
            cleaned.append(tx)
            prev_bal = bal
            continue

        expected = prev_bal + credit - debit
        if abs(expected - bal) <= tol:
            cleaned.append(tx)
            prev_bal = bal
            continue

        # try swapping debit/credit
        expected_swapped = prev_bal + debit - credit
        if abs(expected_swapped - bal) <= tol:
            tx["debit"], tx["credit"] = credit, debit
            cleaned.append(tx)
            prev_bal = bal
            continue

        # Still inconsistent => mark balance invalid so it doesn't poison min/max/ending
        tx["balance"] = None
        cleaned.append(tx)
        # do NOT update prev_bal

    return cleaned


# =========================================================
# Public entry point (used by app.py)
# =========================================================
def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    """
    Works with your app.py which passes an already-open pdfplumber.PDF object:
      parse_affin_bank(pdf, filename)

    Also supports standalone usage with bytes/path/file-like.
    """
    bank_name = "Affin Bank"
    transactions: List[Dict[str, Any]] = []

    # If app.py already opened it
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
    seq_global = 0

    try:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = _get_page_words(page)
            if not words:
                continue

            rows = _cluster_rows(words, y_tol=3.5)
            if not rows:
                continue

            col = _detect_columns(rows)

            # Parse rows
            for row_y, row_words in rows:
                txt = _row_text(row_words)
                if not txt:
                    continue
                up = txt.upper()

                if _is_summary_row(up):
                    continue
                # Keep BF rows (anchor)
                # Skip obvious non-tx lines unless they actually contain a date
                if _looks_non_tx_row(up) and not _row_has_date(row_words):
                    continue

                # Identify date
                date_iso = None
                date_idx = None
                for i, w in enumerate(row_words[:7]):
                    d = _to_iso_date(w["text"])
                    if d:
                        date_iso = d
                        date_idx = i
                        break
                if not date_iso:
                    continue

                # Read balance/debit/credit from columns if possible
                balance = None
                debit = None
                credit = None

                if col:
                    # Balance column: allow a slightly wider tolerance
                    balance = _pick_money_near(row_words, col["balance_x"], tol=65.0)
                    if col["debit_x"] > 0:
                        debit = _pick_money_near(row_words, col["debit_x"], tol=50.0)
                    if col["credit_x"] > 0:
                        credit = _pick_money_near(row_words, col["credit_x"], tol=50.0)

                # Fallback: rightmost money token
                if balance is None:
                    balance = _rightmost_money(row_words)

                if balance is None:
                    continue

                # B/F row anchors balance; do not emit as transaction
                if _is_bf_row(up):
                    prev_balance = float(balance)
                    continue

                # Description: all non-money after date
                desc_parts: List[str] = []
                for j, w in enumerate(row_words):
                    if date_idx is not None and j <= date_idx:
                        continue
                    t = (w["text"] or "").strip().strip("|")
                    if not t:
                        continue
                    if _is_money_token(t):
                        continue
                    desc_parts.append(t)
                description = _norm(" ".join(desc_parts))

                # If debit/credit missing, infer from delta
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

                # Safety valve: skip absurd balance jumps (OCR row mix); do not update anchor
                if prev_balance is not None:
                    if abs(float(balance) - float(prev_balance)) > 2_000_000:
                        continue

                seq_global += 1
                tx = {
                    "date": date_iso,
                    "description": description,
                    "debit": round(float(debit), 2),
                    "credit": round(float(credit), 2),
                    "balance": round(float(balance), 2),
                    "page": int(page_num),
                    "bank": bank_name,
                    "source_file": source_file or "",
                    # internal sort keys for repair/aggregation correctness
                    "_y": float(row_y),
                    "_seq": int(seq_global),
                }
                transactions.append(tx)

                # update prev balance with parsed balance (anchor for next delta inference)
                prev_balance = float(balance)

    finally:
        if should_close:
            try:
                pdf.close()
            except Exception:
                pass

    # Repair/validate (this is what fixes Streamlit wrong totals/min/max/ending)
    transactions = _validate_and_repair(transactions, tol=0.05)

    # IMPORTANT: remove internal keys so downstream JSON/export is clean
    for tx in transactions:
        tx.pop("_y", None)
        tx.pop("_seq", None)

    return transactions
