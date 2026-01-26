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
# Patterns / constants
# =========================================================

# Date token in OCR/text, allowing small OCR spacing noise
DATE_IN_TOKEN_RE = re.compile(r"(?P<d>\d{1,2})\s*/\s*(?P<m>\d{1,2})\s*/\s*(?P<y>\d{2,4})")

# Money token (OCR word) like:
#   1,234.56
#   1234.56
#   (1,234.56)
#   1,234.56-
#   1,234.56+
#   RM1,234.56
MONEY_TOKEN_RE = re.compile(
    r"^\(?\s*(?:RM\s*)?(?P<num>(?:\d{1,3}(?:,\d{3})*|\d+)?\.\d{2})\s*\)?(?P<trail_sign>[+-])?\s*[\.,;:|]*\s*$",
    re.I,
)

# For scanning in OCR text (line-based) - lenient to commas being missed
MONEY_IN_TEXT_RE = re.compile(r"\d[\d,]*\.\d{2}")

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


# =========================================================
# Small helpers
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

    # common OCR confusion
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
# OCR/text helpers for totals (FIXED)
# =========================================================
def _page_text_pdf_or_ocr(page: pdfplumber.page.Page, *, resolution: int = 130, crop: bool = False) -> str:
    """
    Return page text; if image-only, fallback to OCR.

    Performance notes:
    - For transaction pages, set crop=True to OCR only the table body (faster, fewer OCR errors).
    - For totals/headers, use crop=False (full page).
    """
    txt = (page.extract_text() or "").strip()
    if len(txt) >= 120:
        return txt
    if pytesseract is None:
        return txt

    try:
        if crop:
            # focus on the table body, avoid heavy headers/footers
            w, h = float(page.width), float(page.height)
            page = page.crop((0, 90, w, h - 55))
        img = page.to_image(resolution=resolution).original
        return pytesseract.image_to_string(img, config="--psm 6") or ""
    except Exception:
        return txt


def _parse_money_flexible(amount_str: str) -> Optional[float]:
    """
    Parse money token that may have:
      - commas or no commas (OCR often drops commas)
      - accidental leading/trailing punctuation
    """
    s = (amount_str or "").strip().strip(".,;:|")
    if not s:
        return None
    s = s.replace("O", "0").replace("o", "0")
    s = s.replace(",", "")
    if not re.fullmatch(r"\d+(?:\.\d{2})", s):
        return None
    try:
        return float(s)
    except Exception:
        return None


def _candidate_amounts_from_token(amount_str: str) -> List[float]:
    """
    Produce candidate parses for totals lines where a count digit may be concatenated
    into the amount (e.g., "4" + "80,932.36" OCRed as "480,932.36").

    Critical fix:
    - DO NOT strip prefix digits just because commas are missing.
      "400620.67" is a valid number and must be kept.
    """
    s_raw = (amount_str or "").strip()
    if not s_raw:
        return []

    # 1) direct parse (preferred)
    direct = _parse_money_flexible(s_raw)
    candidates: List[float] = []
    if direct is not None:
        candidates.append(float(direct))

    # 2) prefix-strip candidates ONLY if original token contains a comma
    if "," in s_raw:
        s = s_raw.strip().strip(".,;:|")
        for k in (1, 2, 3):
            if len(s) <= k:
                continue
            rem = s[k:]
            v = _parse_money_flexible(rem)
            if v is not None:
                candidates.append(float(v))

    # unique + stable order
    out: List[float] = []
    for v in candidates:
        if v not in out:
            out.append(v)
    return out


def _pick_last_money_token(line: str) -> Optional[str]:
    nums = MONEY_IN_TEXT_RE.findall(line or "")
    if not nums:
        return None
    return nums[-1].strip()


def _scan_lines_for_totals_candidates(text: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {
        "opening_balance": [],
        "total_debit": [],
        "total_credit": [],
        "ending_balance": [],
    }
    if not text:
        return out

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        up = line.upper()

        token = _pick_last_money_token(line)
        if not token:
            continue

        cands = _candidate_amounts_from_token(token)
        if not cands:
            continue

        # Total debit
        hit_debit = (
            ((("TOTAL" in up or "JUMLAH" in up) and "DEBIT" in up))
            or ("JUMLAH" in up and ("PENGELUARAN" in up or "KELUAR" in up))
        )
        if hit_debit:
            out["total_debit"].extend(cands)
            continue

        # Total credit
        hit_credit = (
            ((("TOTAL" in up or "JUMLAH" in up) and ("CREDIT" in up or "KREDIT" in up)))
            or ("JUMLAH" in up and ("PEMASUKAN" in up or "MASUK" in up))
        )
        if hit_credit:
            out["total_credit"].extend(cands)
            continue

        # Opening (B/F)
        hit_open = ("B/F" in up) or ("BROUGHT FORWARD" in up) or ("BAKI" in up and ("AWAL" in up or "MULA" in up))
        if hit_open:
            out["opening_balance"].extend(cands)
            continue

        # Ending (C/F)
        hit_end = (
            ("C/F" in up)
            or ("CARRIED FORWARD" in up)
            or ("ENDING BALANCE" in up)
            or ("CLOSING BALANCE" in up)
            or ("BAKI" in up and ("AKHIR" in up or "PENUTUP" in up or "TUTUP" in up))
        )
        if hit_end:
            out["ending_balance"].extend(cands)
            continue

    return out


def _choose_best_totals(cands: Dict[str, List[float]]) -> Dict[str, Optional[float]]:
    """
    Choose best values by enforcing accounting identity when possible:
      opening + total_credit - total_debit ~= ending
    """
    def uniq(xs: List[float]) -> List[float]:
        out=[]
        for x in xs:
            if x not in out:
                out.append(x)
        return out

    open_c = uniq(cands.get("opening_balance", []))
    end_c = uniq(cands.get("ending_balance", []))
    debit_c = uniq(cands.get("total_debit", []))
    credit_c = uniq(cands.get("total_credit", []))

    # Prefer "larger" opening/ending if multiple (stripping creates smaller artifacts)
    open_sorted = sorted(open_c, reverse=True)
    end_sorted = sorted(end_c, reverse=True)

    best = {"opening_balance": None, "total_debit": None, "total_credit": None, "ending_balance": None}

    tol = 0.05
    best_err = None
    for o in (open_sorted[:6] or [None]):
        for e in (end_sorted[:6] or [None]):
            for d in (sorted(debit_c, reverse=True)[:8] or [None]):
                for c in (sorted(credit_c, reverse=True)[:8] or [None]):
                    if o is None or e is None or d is None or c is None:
                        continue
                    err = abs((o + c - d) - e)
                    if err <= tol:
                        if best_err is None or err < best_err:
                            best_err = err
                            best = {"opening_balance": o, "total_debit": d, "total_credit": c, "ending_balance": e}

    if best["opening_balance"] is not None:
        return best

    # Fallbacks if identity cannot be solved (partial totals)
    best["opening_balance"] = open_sorted[0] if open_sorted else None
    best["ending_balance"] = end_sorted[0] if end_sorted else None
    best["total_debit"] = max(debit_c) if debit_c else None
    best["total_credit"] = max(credit_c) if credit_c else None
    return best


def extract_affin_statement_totals(pdf_input: Any, source_file: str = "") -> Dict[str, Any]:
    """
    Extract printed totals (ground truth) from statement:
      opening_balance, total_debit, total_credit, ending_balance

    Fixes:
    - OCR often drops commas (e.g., "400,620.67" -> "400620.67"). This must NOT be treated
      as a "count digit glued to amount".
    - If count digits are genuinely glued (rare), we keep both direct and stripped candidates
      and choose the combination that satisfies:
        opening + credit - debit = ending.
    """
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

        merged_candidates = {"opening_balance": [], "total_debit": [], "total_credit": [], "ending_balance": []}

        for i in idxs:
            text = _page_text_pdf_or_ocr(pdf.pages[i], resolution=140, crop=False).replace("\x0c", " ")
            found = _scan_lines_for_totals_candidates(text)
            for k in merged_candidates.keys():
                merged_candidates[k].extend(found.get(k, []))

        best = _choose_best_totals(merged_candidates)

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
# Transaction extraction
# =========================================================

_TX_LINE_RE = re.compile(
    r"^(?P<date>\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4})\s*\|\s*(?P<rest>.*)$"
)

def _parse_transactions_from_ocr_text(pdf: pdfplumber.PDF, source_file: str) -> List[Dict[str, Any]]:
    """
    Fast, robust OCR-text parser for Affin image statements.

    Strategy:
    - OCR the page to text (psm6), cropped to table body
    - Parse lines starting with "DD/MM/YY |"
    - Extract last money token as running balance
    - Compute debit/credit via balance delta (most reliable)
    """
    txs: List[Dict[str, Any]] = []
    prev_balance: Optional[float] = None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = _page_text_pdf_or_ocr(page, resolution=120, crop=True)
        if not text:
            continue

        lines = [_norm(l) for l in text.splitlines() if _norm(l)]
        for ln in lines:
            m = _TX_LINE_RE.match(ln)
            if not m:
                continue

            date_iso = _to_iso_date(m.group("date"))
            if not date_iso:
                continue

            rest = m.group("rest").strip()
            up = rest.upper()

            nums = MONEY_IN_TEXT_RE.findall(ln)
            if not nums:
                continue

            bal_token = nums[-1]
            bal = _parse_money_flexible(bal_token)
            if bal is None:
                continue

            # B/F row anchors prev_balance but is not a transaction
            if _is_bf_row(up) or "BALANCE BROUGHT" in up:
                prev_balance = float(bal)
                continue

            debit = credit = 0.0
            if prev_balance is not None:
                delta = round(float(bal) - float(prev_balance), 2)
                if delta > 0:
                    credit = delta
                elif delta < 0:
                    debit = -delta

            prev_balance = float(bal)

            desc = rest
            for tok in nums:
                desc = desc.replace(tok, " ").strip()
            desc = _norm(desc)

            txs.append(
                {
                    "date": date_iso,
                    "description": desc,
                    "debit": round(float(debit), 2),
                    "credit": round(float(credit), 2),
                    "balance": round(float(bal), 2),
                    "page": int(page_num),
                    "bank": "Affin Bank",
                    "source_file": source_file or "",
                }
            )

    return txs


# -----------------------------
# Word-based extraction (kept for text PDFs)
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


def _detect_columns(rows: List[Tuple[float, List[Dict[str, Any]]]]) -> Optional[Dict[str, float]]:
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
    return None


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
        return None, None, money_items[-1][1]

    debit_x = float(col.get("debit_x", -1))
    credit_x = float(col.get("credit_x", -1))
    balance_x = float(col.get("balance_x", -1))

    debit_vals: List[float] = []
    credit_vals: List[float] = []
    balance_vals: List[float] = []

    for xc, v in money_items:
        candidates = []
        if debit_x > 0:
            candidates.append(("debit", abs(xc - debit_x)))
        if credit_x > 0:
            candidates.append(("credit", abs(xc - credit_x)))
        if balance_x > 0:
            candidates.append(("balance", abs(xc - balance_x)))
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


def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    """
    Affin Bank parser.

    Fix:
    - For image-only statements, word-box OCR + column detection is unstable and causes
      wrong total debit/credit. We now switch to OCR-text parsing with balance-delta
      inference, which is significantly more robust.
    """
    bank_name = "Affin Bank"

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
        # If pdfplumber can extract words, use the word-based parser.
        # Otherwise, fall back to OCR-text mode.
        has_words = False
        try:
            for p in pdf.pages[:2]:
                if _words_from_pdf(p):
                    has_words = True
                    break
        except Exception:
            has_words = False

        if not has_words:
            return _parse_transactions_from_ocr_text(pdf, source_file)

        txs: List[Dict[str, Any]] = []
        for page_num, page in enumerate(pdf.pages, start=1):
            words = _words_from_pdf(page)
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

                if _looks_non_tx_row(up) and not _row_has_date(row_words):
                    i += 1
                    continue

                date_iso = None
                for w in row_words[:10]:
                    d = _to_iso_date(w["text"])
                    if d:
                        date_iso = d
                        break
                if not date_iso:
                    i += 1
                    continue

                # merge wrapped continuation lines
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

                if _is_bf_row(up):
                    i = k
                    continue

                desc_parts: List[str] = []
                for ww in block_words:
                    t = (ww.get("text") or "").strip().strip("|")
                    if not t:
                        continue
                    if _is_money_token(t):
                        continue
                    if _to_iso_date(t) == date_iso:
                        continue
                    desc_parts.append(t)

                description = _norm(" ".join(desc_parts))

                txs.append(
                    {
                        "date": date_iso,
                        "description": description,
                        "debit": round(float(debit or 0.0), 2),
                        "credit": round(float(credit or 0.0), 2),
                        "balance": round(float(balance), 2),
                        "page": int(page_num),
                        "bank": bank_name,
                        "source_file": source_file or "",
                        "_y": float(row_y),
                    }
                )
                i = k

        txs.sort(key=lambda x: (x.get("date", ""), int(x.get("page") or 0), float(x.get("_y") or 0.0)))
        for t in txs:
            t.pop("_y", None)
        return txs

    finally:
        if should_close:
            try:
                pdf.close()
            except Exception:
                pass
