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

DATE_IN_TOKEN_RE = re.compile(r"(?P<d>\d{1,2})\s*/\s*(?P<m>\d{1,2})\s*/\s*(?P<y>\d{2,4})")
MONEY_TOKEN_RE = re.compile(
    r"^\(?\s*(?:RM\s*)?(?P<num>(?:\d{1,3}(?:,\d{3})*|\d+)?\.\d{2})\s*\)?(?P<trail_sign>[+-])?\s*[\.,;:|]*\s*$",
    re.I,
)
MONEY_IN_TEXT_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}")

HEADER_HINTS = ("DATE", "TARIKH", "DEBIT", "CREDIT", "BALANCE", "BAKI")
NON_TX_HINTS = (
    "ACCOUNT",
    "NO.",
    "STATEMENT",
    "PENYATA",
    "PAGE",
    "PIDM",
    "AFFIN",
    "BRANCH",
    "ADDRESS",
    "CUSTOMER",
    "CIF",
    "PERIOD",
    "TARIKH PENYATA",
    "STATEMENT DATE",
)

BF_HINTS = (
    "B/F",
    "BALANCE B/F",
    "BAKI B/F",
    "BAKI MULA",
    "BAKI AWAL",
    "OPENING",
    "BALANCE BROUGHT",
    "BALANCE BROUGHT FORWARD",
)

FILENAME_MONTH_RE = re.compile(r"(?P<y>20\d{2})[^\d]?(?P<m>0[1-9]|1[0-2])")
MAX_REASONABLE_DELTA = 5_000_000.00


# =========================================================
# Basic helpers
# =========================================================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _infer_month_from_filename(filename: str) -> Optional[str]:
    if not filename:
        return None
    m = FILENAME_MONTH_RE.search(filename)
    if not m:
        return None
    return f"{int(m.group('y')):04d}-{int(m.group('m')):02d}"


def _to_iso_date(token: str) -> Optional[str]:
    if not token:
        return None
    m = DATE_IN_TOKEN_RE.search(token)
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


def _clean_money_token(token: str) -> Optional[str]:
    if token is None:
        return None
    t = str(token).strip()
    if not t:
        return None

    t = t.replace("O", "0").replace("o", "0").replace(" ", "")

    m = MONEY_TOKEN_RE.match(t)
    if not m:
        t2 = t.strip(".,;:|")
        m = MONEY_TOKEN_RE.match(t2)
        if not m:
            return None
        t = t2

    num = (m.group("num") or "").replace(",", "")
    if num.startswith("."):
        num = "0" + num

    sign = m.group("trail_sign")
    paren_neg = t.startswith("(") and ")" in t
    if paren_neg or sign == "-":
        return "-" + num
    return num


def _money_to_float(token: str) -> Optional[float]:
    s = _clean_money_token(token)
    if s is None:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _is_money_token(token: str) -> bool:
    return _clean_money_token(token) is not None


def _looks_non_tx_row(up: str) -> bool:
    return any(h in up for h in NON_TX_HINTS)


def _is_bf_row(up: str) -> bool:
    return any(h in up for h in BF_HINTS)


# =========================================================
# Extraction: words (PDF text first, OCR fallback)
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

    try:
        w, h = float(page.width), float(page.height)
        crop = page.crop((0, 80, w, h - 50))
        img = crop.to_image(resolution=240).original
    except Exception:
        img = page.to_image(resolution=240).original

    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT,
        config="--psm 6",
    )

    n = len(data.get("text", []))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        x, y, ww, hh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append({"text": txt, "x0": float(x), "x1": float(x + ww), "y0": float(y), "y1": float(y + hh)})
    return out


def _get_page_words(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    w = _words_from_pdf(page)
    if w:
        return w
    return _words_from_ocr(page)


# =========================================================
# Row clustering (y grouping)
# =========================================================
def _cluster_rows(words: List[Dict[str, Any]], y_tol: float = 2.8) -> List[Tuple[float, List[Dict[str, Any]]]]:
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
        out.append((float(b["y"]), sorted(b["items"], key=lambda z: z["x0"])))
    return out


def _row_text(row_words: List[Dict[str, Any]]) -> str:
    return _norm(" ".join(w["text"] for w in row_words))


def _row_has_date(row_words: List[Dict[str, Any]]) -> bool:
    for w in row_words[:10]:
        if _to_iso_date(w["text"]):
            return True
    return False


# =========================================================
# Column detection for Debit/Credit/Balance
# =========================================================
def _detect_columns(rows: List[Tuple[float, List[Dict[str, Any]]]]) -> Optional[Dict[str, float]]:
    # scan for a header row
    for _, rw in rows[:80]:
        up = _row_text(rw).upper()
        if not any(h in up for h in HEADER_HINTS):
            continue

        debit_x = credit_x = balance_x = None
        for w in rw:
            t = w["text"].upper()
            xc = (w["x0"] + w["x1"]) / 2.0
            if debit_x is None and ("DEBIT" in t or t == "DR"):
                debit_x = xc
            if credit_x is None and ("CREDIT" in t or t == "CR"):
                credit_x = xc
            if balance_x is None and ("BAL" in t or "BAKI" in t):
                balance_x = xc

        if debit_x and credit_x and balance_x:
            return {"debit_x": float(debit_x), "credit_x": float(credit_x), "balance_x": float(balance_x)}

    # fallback: cluster money x positions on date rows
    money_xs: List[float] = []
    for _, rw in rows:
        if not _row_has_date(rw):
            continue
        up = _row_text(rw).upper()
        if _looks_non_tx_row(up):
            continue
        for w in rw:
            if _is_money_token(w["text"]):
                money_xs.append(float((w["x0"] + w["x1"]) / 2.0))

    if len(money_xs) < 12:
        return None

    money_xs.sort()
    clusters: List[List[float]] = []
    for x in money_xs:
        if not clusters or abs(x - clusters[-1][-1]) > 30:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    clusters.sort(key=lambda c: sum(c) / len(c))
    if len(clusters) < 2:
        return None

    balance_x = sum(clusters[-1]) / len(clusters[-1])
    credit_x = sum(clusters[-2]) / len(clusters[-2])
    debit_x = sum(clusters[-3]) / len(clusters[-3]) if len(clusters) >= 3 else -1.0
    return {"debit_x": float(debit_x), "credit_x": float(credit_x), "balance_x": float(balance_x)}


def _classify_money_by_columns(row_words: List[Dict[str, Any]], col: Optional[Dict[str, float]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    money_items: List[Tuple[float, float]] = []
    for w in row_words:
        if not _is_money_token(w["text"]):
            continue
        v = _money_to_float(w["text"])
        if v is None:
            continue
        xc = float((w["x0"] + w["x1"]) / 2.0)
        money_items.append((xc, float(v)))

    if not money_items:
        return None, None, None

    money_items.sort(key=lambda t: t[0])

    if not col:
        # if we cannot detect columns, assume rightmost is balance
        return None, None, money_items[-1][1]

    debit_x = float(col.get("debit_x", -1))
    credit_x = float(col.get("credit_x", -1))
    balance_x = float(col.get("balance_x", -1))

    debit_vals: List[float] = []
    credit_vals: List[float] = []
    balance_vals: List[float] = []

    for xc, v in money_items:
        # ignore left noise
        if balance_x > 0 and xc < (balance_x - 250):
            continue

        candidates = []
        if debit_x > 0:
            candidates.append(("debit", abs(xc - debit_x)))
        if credit_x > 0:
            candidates.append(("credit", abs(xc - credit_x)))
        if balance_x > 0:
            candidates.append(("balance", abs(xc - balance_x)))
        if not candidates:
            continue

        label, dist = min(candidates, key=lambda x: x[1])
        if dist > 90:
            continue

        if label == "debit":
            debit_vals.append(abs(v))
        elif label == "credit":
            credit_vals.append(abs(v))
        else:
            balance_vals.append(v)

    debit = round(sum(debit_vals), 2) if debit_vals else None
    credit = round(sum(credit_vals), 2) if credit_vals else None
    balance = balance_vals[-1] if balance_vals else money_items[-1][1]
    return debit, credit, float(balance)


# =========================================================
# Statement totals extraction (used for monthly summary)
# =========================================================
def _page_text_pdf_or_ocr(page: pdfplumber.page.Page) -> str:
    txt = (page.extract_text() or "").strip()
    if len(txt) >= 120:
        return txt
    if pytesseract is None:
        return txt
    try:
        img = page.to_image(resolution=260).original
        return pytesseract.image_to_string(img, config="--psm 6") or txt
    except Exception:
        return txt


def _rightmost_money_in_line(line: str) -> Optional[float]:
    nums = MONEY_IN_TEXT_RE.findall(line or "")
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except Exception:
        return None


def _scan_lines_for_totals(text: str) -> Dict[str, Optional[float]]:
    out = {"opening_balance": None, "total_debit": None, "total_credit": None, "ending_balance": None}
    if not text:
        return out
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        up = line.upper()

        # debit (out)
        if out["total_debit"] is None:
            if ((("TOTAL" in up or "JUMLAH" in up) and "DEBIT" in up) or ("JUMLAH" in up and ("PENGELUARAN" in up or "KELUAR" in up))):
                v = _rightmost_money_in_line(line)
                if v is not None:
                    out["total_debit"] = float(v)
                    continue

        # credit (in)
        if out["total_credit"] is None:
            if ((("TOTAL" in up or "JUMLAH" in up) and "CREDIT" in up) or ("JUMLAH" in up and ("PEMASUKAN" in up or "MASUK" in up))):
                v = _rightmost_money_in_line(line)
                if v is not None:
                    out["total_credit"] = float(v)
                    continue

        # opening
        if out["opening_balance"] is None:
            if ("B/F" in up) or ("BROUGHT FORWARD" in up) or ("BAKI" in up and ("AWAL" in up or "MULA" in up)):
                v = _rightmost_money_in_line(line)
                if v is not None:
                    out["opening_balance"] = float(v)
                    continue

        # ending/closing
        if out["ending_balance"] is None:
            if ("C/F" in up) or ("CARRIED FORWARD" in up) or ("ENDING BALANCE" in up) or ("CLOSING BALANCE" in up) or ("BAKI" in up and ("AKHIR" in up or "PENUTUP" in up or "TUTUP" in up)):
                v = _rightmost_money_in_line(line)
                if v is not None:
                    out["ending_balance"] = float(v)
                    continue

    return out


def extract_affin_statement_totals(pdf_input: Any, source_file: str = "") -> Dict[str, Any]:
    if hasattr(pdf_input, "pages") and hasattr(pdf_input, "close"):
        pdf = pdf_input
        should_close = False
    else:
        should_close = True
        if isinstance(pdf_input, (bytes, bytearray)):
            pdf = pdfplumber.open(BytesIO(bytes(pdf_input)))
        elif hasattr(pdf_input, "getvalue"):
            pdf = pdfplumber.open(BytesIO(pdf_input.getvalue()))
        else:
            pdf = pdfplumber.open(pdf_input)

    try:
        n = len(pdf.pages)
        idxs: List[int] = []
        for i in [0, 1, max(0, n - 2), max(0, n - 1)]:
            if 0 <= i < n and i not in idxs:
                idxs.append(i)

        best = {"opening_balance": None, "total_debit": None, "total_credit": None, "ending_balance": None}

        for i in idxs:
            text = _page_text_pdf_or_ocr(pdf.pages[i]).replace("\x0c", " ")
            found = _scan_lines_for_totals(text)
            for k in best.keys():
                if best[k] is None and found.get(k) is not None:
                    best[k] = found.get(k)

        return {
            "bank": "Affin Bank",
            "source_file": source_file or "",
            "statement_month": _infer_month_from_filename(source_file),
            "opening_balance": best["opening_balance"],
            "total_debit": best["total_debit"],
            "total_credit": best["total_credit"],
            "ending_balance": best["ending_balance"],
        }
    finally:
        if should_close:
            try:
                pdf.close()
            except Exception:
                pass


# =========================================================
# Public parser: transactions
# =========================================================
def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    """
    Extract transactions by detecting columns, not by matching whole lines.
    OCR is used only when the PDF has no extractable text.

    Returns list[dict] with keys:
      date, description, debit, credit, balance, page, bank, source_file
    """
    bank_name = "Affin Bank"
    txs: List[Dict[str, Any]] = []

    if hasattr(pdf_input, "pages") and hasattr(pdf_input, "close"):
        pdf = pdf_input
        should_close = False
    else:
        should_close = True
        if isinstance(pdf_input, (bytes, bytearray)):
            pdf = pdfplumber.open(BytesIO(bytes(pdf_input)))
        elif hasattr(pdf_input, "getvalue"):
            pdf = pdfplumber.open(BytesIO(pdf_input.getvalue()))
        else:
            pdf = pdfplumber.open(pdf_input)

    prev_balance: Optional[float] = None
    seq = 0

    try:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = _get_page_words(page)
            if not words:
                continue

            rows = _cluster_rows(words, y_tol=2.8)
            if not rows:
                continue

            col = _detect_columns(rows)

            i = 0
            while i < len(rows):
                row_y, row_words = rows[i]
                txt = _row_text(row_words)
                if not txt:
                    i += 1
                    continue
                up = txt.upper()

                # skip non-tx rows unless they contain date
                if _looks_non_tx_row(up) and not _row_has_date(row_words):
                    i += 1
                    continue

                # find date
                date_iso = None
                date_idx = None
                for j, w in enumerate(row_words[:10]):
                    d = _to_iso_date(w["text"])
                    if d:
                        date_iso = d
                        date_idx = j
                        break
                if not date_iso:
                    i += 1
                    continue

                # merge continuation lines (wrapped description)
                block_words = list(row_words)
                k = i + 1
                while k < len(rows) and not _row_has_date(rows[k][1]):
                    nxt_up = _row_text(rows[k][1]).upper()
                    if any(h in nxt_up for h in HEADER_HINTS):
                        break
                    block_words.extend(rows[k][1])
                    k += 1
                block_words.sort(key=lambda z: (z["y0"], z["x0"]))

                debit, credit, balance = _classify_money_by_columns(block_words, col)
                if balance is None:
                    i = k
                    continue

                # treat B/F as anchor
                if _is_bf_row(up):
                    prev_balance = float(balance)
                    i = k
                    continue

                # description: all non-money after date token
                desc_parts: List[str] = []
                for ww in block_words:
                    t = (ww.get("text") or "").strip().strip("|")
                    if not t:
                        continue
                    if _is_money_token(t):
                        continue
                    # remove the date token itself
                    if _to_iso_date(t) == date_iso:
                        continue
                    desc_parts.append(t)

                description = _norm(" ".join(desc_parts))

                debit_f = float(debit) if debit is not None else 0.0
                credit_f = float(credit) if credit is not None else 0.0

                # infer when missing
                if prev_balance is not None:
                    delta = round(float(balance) - float(prev_balance), 2)
                    if abs(delta) <= MAX_REASONABLE_DELTA:
                        if debit is None and credit is None:
                            if delta > 0:
                                credit_f = abs(delta)
                                debit_f = 0.0
                            elif delta < 0:
                                debit_f = abs(delta)
                                credit_f = 0.0
                        # if both present (often OCR double-detected), force to delta
                        elif debit_f > 0 and credit_f > 0:
                            if delta > 0:
                                credit_f = abs(delta)
                                debit_f = 0.0
                            elif delta < 0:
                                debit_f = abs(delta)
                                credit_f = 0.0

                seq += 1
                txs.append(
                    {
                        "date": date_iso,
                        "description": description,
                        "debit": round(float(debit_f), 2),
                        "credit": round(float(credit_f), 2),
                        "balance": round(float(balance), 2),
                        "page": int(page_num),
                        "bank": bank_name,
                        "source_file": source_file or "",
                        "_y": float(row_y),
                        "_seq": int(seq),
                    }
                )

                prev_balance = float(balance)
                i = k

    finally:
        if should_close:
            try:
                pdf.close()
            except Exception:
                pass

    # stable sort
    txs.sort(key=lambda x: (x.get("date", ""), int(x.get("page") or 0), float(x.get("_y") or 0), int(x.get("_seq") or 0)))
    for t in txs:
        t.pop("_y", None)
        t.pop("_seq", None)
    return txs
