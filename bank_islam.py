# bank_islam.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None

from PIL import ImageEnhance, ImageOps


# =========================================================
# BANK ISLAM – FORMAT 1 (TABLE-BASED)
# =========================================================
def parse_bank_islam_format1(pdf, source_file):
    transactions: List[Dict[str, Any]] = []

    def extract_amount(text):
        if not text:
            return None
        s = re.sub(r"\s+", "", str(text))
        m = re.search(r"(-?[\d,]+\.\d{2})", s)
        return float(m.group(1).replace(",", "")) if m else None

    for page_num, page in enumerate(pdf.pages, start=1):
        table = page.extract_table()
        if not table:
            continue

        for row in table:
            row = list(row) if row else []
            while len(row) < 12:
                row.append(None)

            (
                no,
                txn_date,
                customer_eft,
                txn_code,
                description,
                ref_no,
                branch,
                debit_raw,
                credit_raw,
                balance_raw,
                sender_recipient,
                payment_details,
            ) = row[:12]

            if not txn_date or not re.search(r"\d{2}/\d{2}/\d{4}", str(txn_date)):
                continue

            try:
                date = datetime.strptime(
                    re.search(r"\d{2}/\d{2}/\d{4}", str(txn_date)).group(), "%d/%m/%Y"
                ).date().isoformat()
            except Exception:
                continue

            debit = extract_amount(debit_raw) or 0.0
            credit = extract_amount(credit_raw) or 0.0
            balance = extract_amount(balance_raw) or 0.0

            # Recovery from description (kept as in your original)
            if debit == 0.0 and credit == 0.0:
                recovered = extract_amount(description)
                if recovered:
                    desc = str(description).upper()
                    if "CR" in desc or "CREDIT" in desc or "IN" in desc:
                        credit = recovered
                    else:
                        debit = recovered

            description_clean = " ".join(
                str(x).replace("\n", " ").strip()
                for x in [no, txn_code, description, sender_recipient, payment_details]
                if x and str(x).lower() != "nan"
            )

            transactions.append(
                {
                    "date": date,
                    "description": description_clean,
                    "debit": round(debit, 2),
                    "credit": round(credit, 2),
                    "balance": round(balance, 2),
                    "page": page_num,
                    "bank": "Bank Islam",
                    "source_file": source_file,
                    "format": "format1",
                }
            )

    return transactions


# =========================================================
# BANK ISLAM – FORMAT 2 (TEXT / STATEMENT-BASED)
# 100% BALANCE DELTA DR/CR LOGIC
# =========================================================
MONEY_RE = re.compile(r"\(?-?[\d,]+\.\d{2}\)?")
DATE_AT_START_RE = re.compile(r"^\s*(\d{1,2}/\d{1,2}/\d{2,4})\b")
BAL_BF_RE = re.compile(r"BAL\s+B/F", re.IGNORECASE)


def _to_float(val):
    if not val:
        return None
    neg = val.startswith("(") and val.endswith(")")
    val = val.strip("()").replace(",", "")
    try:
        num = float(val)
        return -num if neg else num
    except ValueError:
        return None


def _parse_date(d: str) -> Optional[str]:
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(d.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def parse_bank_islam_format2(pdf, source_file):
    transactions: List[Dict[str, Any]] = []
    prev_balance: Optional[float] = None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]

        for line in lines:
            upper = line.upper()

            # 1) Opening balance
            if BAL_BF_RE.search(upper):
                money = MONEY_RE.findall(line)
                if money:
                    prev_balance = _to_float(money[-1])
                continue

            # 2) Transaction lines (must start with date)
            m_date = DATE_AT_START_RE.match(line)
            if not m_date or prev_balance is None:
                continue

            date = _parse_date(m_date.group(1))
            if not date:
                continue

            money_raw = MONEY_RE.findall(line)
            money_vals = [_to_float(x) for x in money_raw if _to_float(x) is not None]
            if not money_vals:
                continue

            balance = money_vals[-1]

            # 3) Balance delta logic
            delta = round(balance - prev_balance, 2)
            credit = delta if delta > 0 else 0.0
            debit = abs(delta) if delta < 0 else 0.0
            prev_balance = balance

            # 4) Description
            desc = line[len(m_date.group(1)) :].strip()
            for tok in money_raw:
                desc = desc.replace(tok, "").strip()

            transactions.append(
                {
                    "date": date,
                    "description": desc,
                    "debit": round(debit, 2),
                    "credit": round(credit, 2),
                    "balance": round(balance, 2),
                    "page": page_num,
                    "bank": "Bank Islam",
                    "source_file": source_file,
                    "format": "format2_balance_delta",
                }
            )

    return transactions


# =========================================================
# FORMAT 3 – eSTATEMENT (BALANCE DELTA, DIFFERENT LAYOUT)
# =========================================================
def parse_bank_islam_format3(pdf, source_file):
    transactions: List[Dict[str, Any]] = []
    prev_balance: Optional[float] = None

    DATE_RE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4})")
    MONEY_RE3 = re.compile(r"(\d{1,3}(?:,\d{3})*\.\d{2})")
    BAL_BF_RE3 = re.compile(r"BAL\s+B/F", re.IGNORECASE)

    def to_float(x):
        return float(x.replace(",", ""))

    def parse_date(d):
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                return datetime.strptime(d, fmt).date().isoformat()
            except ValueError:
                continue
        return None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text(x_tolerance=1) or ""
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]

        for line in lines:
            # 1) Opening balance
            if BAL_BF_RE3.search(line):
                nums = MONEY_RE3.findall(line)
                if nums:
                    prev_balance = to_float(nums[-1])
                continue

            # 2) Extract only the line starting with a Date
            date_match = DATE_RE.match(line)
            if date_match and prev_balance is not None:
                raw_date = date_match.group(1)
                nums = MONEY_RE3.findall(line)

                # We expect at least a transaction amount and a balance on this line
                if len(nums) >= 2:
                    balance = to_float(nums[-1])

                    # Remove the date and all money values from this specific line
                    desc = line.replace(raw_date, "").strip()
                    for n in nums:
                        desc = desc.replace(n, "").strip()

                    delta = round(balance - prev_balance, 2)

                    transactions.append(
                        {
                            "date": parse_date(raw_date),
                            "description": desc,
                            "debit": abs(delta) if delta < 0 else 0.0,
                            "credit": delta if delta > 0 else 0.0,
                            "balance": balance,
                            "page": page_num,
                            "bank": "Bank Islam",
                            "source_file": source_file,
                            "format": "format3_estatement",
                        }
                    )

                    prev_balance = balance

    return transactions


# =========================================================
# FORMAT 4 – eSTATEMENT (BALANCE DELTA, DIFFERENT LAYOUT)
# FIXED: indentation + per-page processing
# =========================================================
def parse_bank_islam_format4(pdf, source_file):
    transactions: List[Dict[str, Any]] = []
    prev_balance: Optional[float] = None

    DATE_RE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4})")
    MONEY_RE4 = re.compile(r"(\d{1,3}(?:,\d{3})*\.\d{2})")
    BAL_BF_RE4 = re.compile(r"BAL\s+B/IF", re.IGNORECASE)

    def to_float(x):
        return float(x.replace(",", ""))

    def parse_date(d):
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                return datetime.strptime(d, fmt).date().isoformat()
            except ValueError:
                continue
        return None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text(x_tolerance=1) or ""
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]

        for line in lines:
            # 1) Capture Opening Balance (Note the extra 'I' in 'B/IF' from OCR)
            if BAL_BF_RE4.search(line):
                nums = MONEY_RE4.findall(line)
                if nums:
                    prev_balance = to_float(nums[-1])
                continue

            # 2) Extract Date-initiated lines
            date_match = DATE_RE.match(line)
            if date_match and prev_balance is not None:
                raw_date = date_match.group(1)
                nums = MONEY_RE4.findall(line)

                if nums:
                    current_balance = to_float(nums[-1])

                    delta = round(current_balance - prev_balance, 2)

                    # Clean description: Remove date and all detected money values
                    desc = line.replace(raw_date, "").strip()
                    for n in nums:
                        desc = desc.replace(n, "").strip()

                    transactions.append(
                        {
                            "date": parse_date(raw_date),
                            "description": desc,
                            "debit": abs(delta) if delta < 0 else 0.0,
                            "credit": delta if delta > 0 else 0.0,
                            "balance": current_balance,
                            "page": page_num,
                            "bank": "Bank Islam",
                            "source_file": source_file,
                            "format": "format4_normalized",
                        }
                    )

                    prev_balance = current_balance

    return transactions


# =========================================================
# OCR V2 (SCANNED PDFs) – MULTI-PASS + BALANCE-DELTA + VALIDATION
# (Only used if scanned + totals mismatch)
# =========================================================
_OCR_DATE_RE = re.compile(r"^\s*(\d{1,2}/\d{1,2}/\d{2,4})\b")
_OCR_MONEY_RE = re.compile(r"(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}")


def _ocr_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def _ocr_image(page, resolution: int = 400):
    img = page.to_image(resolution=resolution).original
    img = ImageOps.grayscale(img)
    img = ImageEnhance.Contrast(img).enhance(1.8)
    return img


def _ocr_text_page_multi(page) -> str:
    """
    Multi-pass OCR:
      - PSM 4 captures header/summary lines more reliably (BAL B/F, TOTALS)
      - PSM 6 captures transaction lines more reliably
    """
    if pytesseract is None:
        return ""
    img = _ocr_image(page, resolution=400)
    t4 = pytesseract.image_to_string(img, config="--psm 4") or ""
    t6 = pytesseract.image_to_string(img, config="--psm 6") or ""
    return t4 + "\n" + t6


def _extract_summary_totals_via_ocr(pdf) -> Tuple[Optional[float], Optional[float]]:
    """
    Find:
      TOTAL DEBIT  <count> <amount>
      TOTAL CREDIT <count> <amount>
    """
    if pytesseract is None or not getattr(pdf, "pages", None):
        return (None, None)

    text = _ocr_text_page_multi(pdf.pages[0])
    text_norm = re.sub(r"\s+", " ", text).upper()

    def find_total(label: str) -> Optional[float]:
        m = re.search(
            rf"{label}\s+\d+\s+((?:\d{{1,3}}(?:,\d{{3}})*)\.\d{{2}})",
            text_norm,
        )
        return _ocr_float(m.group(1)) if m else None

    return find_total("TOTAL DEBIT"), find_total("TOTAL CREDIT")


def _extract_opening_balance_via_ocr(pdf) -> Optional[float]:
    if pytesseract is None or not getattr(pdf, "pages", None):
        return None

    text = _ocr_text_page_multi(pdf.pages[0])
    text_norm = re.sub(r"\s+", " ", text).upper()

    m = re.search(
        r"BAL\s*B/F\s+((?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2})",
        text_norm,
    )
    return _ocr_float(m.group(1)) if m else None


def _date_sort_key(raw_date: str) -> datetime:
    d = raw_date.strip()
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    return datetime(2100, 1, 1)


def _collect_date_balances_from_ocr(pdf) -> List[Tuple[str, float, str, int]]:
    """
    Returns list of (raw_date, ending_balance, line_text, page_num).
    Dedupe by (raw_date, balance) across OCR passes.
    """
    rows: Dict[Tuple[str, float], Tuple[str, int]] = {}

    for page_num, page in enumerate(pdf.pages, start=1):
        text = _ocr_text_page_multi(page)
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]

        for line in lines:
            dm = _OCR_DATE_RE.match(line)
            if not dm:
                continue

            nums = _OCR_MONEY_RE.findall(line)
            if not nums:
                continue

            bal = _ocr_float(nums[-1])
            if bal is None:
                continue

            key = (dm.group(1), bal)
            rows[key] = (line, page_num)

    out: List[Tuple[str, float, str, int]] = []
    for (d, b), (line, page_num) in rows.items():
        out.append((d, b, line, page_num))

    out.sort(key=lambda x: _date_sort_key(x[0]))
    return out


def _recompute_totals_with_balance_adjustment(
    opening: float,
    date_bal_rows: List[Tuple[str, float, str, int]],
    adjust_index: Optional[int] = None,
    adjust_amount: float = 0.0,
) -> Tuple[float, float]:
    prev = opening
    total_debit = 0.0
    total_credit = 0.0

    for i, (_, bal, _, _) in enumerate(date_bal_rows):
        if adjust_index is not None and i == adjust_index:
            bal = round(bal + adjust_amount, 2)

        delta = round(bal - prev, 2)
        if delta > 0:
            total_credit += delta
        elif delta < 0:
            total_debit += abs(delta)
        prev = bal

    return round(total_debit, 2), round(total_credit, 2)


def parse_bank_islam_ocr_balance_delta(pdf, source_file) -> List[Dict[str, Any]]:
    """
    OCR fallback for scanned statements:
    - Multi-pass OCR
    - Use ending balance and compute debit/credit via balance delta
    - If statement totals exist, attempt tiny correction (±0.10/±0.20/±0.30) to fix OCR drift
    """
    if pytesseract is None or not getattr(pdf, "pages", None):
        return []

    opening = _extract_opening_balance_via_ocr(pdf)
    if opening is None:
        return []

    stmt_td, stmt_tc = _extract_summary_totals_via_ocr(pdf)
    rows = _collect_date_balances_from_ocr(pdf)
    if not rows:
        return []

    # Totals before correction
    td, tc = _recompute_totals_with_balance_adjustment(opening, rows)

    # Try correction if totals mismatch and statement provides totals
    if stmt_td is not None and stmt_tc is not None:
        if abs(td - stmt_td) > 0.01 or abs(tc - stmt_tc) > 0.01:
            best = None  # (score, idx, adj)
            for idx in range(len(rows)):
                for adj in (-0.10, 0.10, -0.20, 0.20, -0.30, 0.30):
                    td2, tc2 = _recompute_totals_with_balance_adjustment(opening, rows, idx, adj)
                    score = abs(td2 - stmt_td) + abs(tc2 - stmt_tc)
                    if best is None or score < best[0]:
                        best = (score, idx, adj)

            if best is not None and best[0] <= 0.02:
                _, idx, adj = best
                d, bal, line, page_num = rows[idx]
                rows[idx] = (d, round(bal + adj, 2), line, page_num)

    # Build transactions
    tx: List[Dict[str, Any]] = []
    prev = opening

    for raw_date, bal, line, page_num in rows:
        date_iso = _parse_date(raw_date)
        if not date_iso:
            prev = bal
            continue

        delta = round(bal - prev, 2)
        credit = delta if delta > 0 else 0.0
        debit = abs(delta) if delta < 0 else 0.0

        desc = line.replace(raw_date, "").strip()
        for n in _OCR_MONEY_RE.findall(line):
            desc = desc.replace(n, "").strip()

        tx.append(
            {
                "date": date_iso,
                "description": desc,
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(bal, 2),
                "page": page_num,
                "bank": "Bank Islam",
                "source_file": source_file,
                "format": "ocr_balance_delta_v2",
            }
        )

        prev = bal

    return tx


# =========================================================
# SCANNED DETECTION + SUM
# =========================================================
def _sum_tx(tx: List[Dict[str, Any]]) -> Tuple[float, float]:
    return (
        round(sum(t.get("debit", 0.0) or 0.0 for t in tx), 2),
        round(sum(t.get("credit", 0.0) or 0.0 for t in tx), 2),
    )


# -------------------------
# NEW (SAFE): detect CID-garbled / unreadable text extraction
# -------------------------
def _text_looks_garbled(txt: str) -> bool:
    """
    Some PDFs are not scanned, but pdfplumber extraction is CID-garbled, e.g. "(cid:123)".
    That content is effectively unparseable by regex-based logic, so we should treat it as
    "needs OCR fallback" without changing normal parsing behavior.
    """
    if not txt:
        return True

    up = txt.upper()

    # Strong signal: CID-coded extraction (font/encoding issue)
    if up.count("(CID:") >= 20:
        return True

    # Secondary signal: low alphanumeric ratio in a long string
    if len(txt) > 800:
        alnum = sum(ch.isalnum() for ch in txt)
        if (alnum / max(len(txt), 1)) < 0.15:
            return True

    return False


def _looks_like_scanned(source_file: str, pdf) -> bool:
    """
    Original behavior:
      - If pdfplumber can't extract meaningful text, it's probably scanned

    FIX (low risk):
      - Also treat CID-garbled extraction as "needs OCR" (not changing normal months)
      - Check first few pages (page 1 alone can be misleading)
    """
    try:
        pages = getattr(pdf, "pages", None) or []
        if not pages:
            return True

        # filename hint
        if "scan" in (source_file or "").lower() or "scanned" in (source_file or "").lower():
            return True

        texts = []
        for p in pages[:3]:
            try:
                texts.append(((p.extract_text() or "").strip()))
            except Exception:
                texts.append("")

        # If any page has decent, non-garbled text, do not treat as scanned
        for t in texts:
            if len(t) >= 120 and not _text_looks_garbled(t):
                return False

        # If all are short or garbled, treat as needing OCR
        if all((len(t) < 80) or _text_looks_garbled(t) for t in texts):
            return True

        return False
    except Exception:
        # Fail safe: allow OCR path
        return True


# =========================================================
# WRAPPER (FIXED ORDER + OCR OVERRIDE ONLY WHEN NEEDED)
# =========================================================
def parse_bank_islam(pdf, source_file):
    # 1) Keep existing behavior for normal (non-scanned) months
    tx = parse_bank_islam_format1(pdf, source_file)
    if not tx:
        tx = parse_bank_islam_format2(pdf, source_file)
    if not tx:
        tx = parse_bank_islam_format4(pdf, source_file)
    if not tx:
        tx = parse_bank_islam_format3(pdf, source_file)

    # 2) For scanned/garbled statements: validate using statement totals and override with OCR v2 if mismatch
    if _looks_like_scanned(source_file, pdf):
        stmt_td, stmt_tc = _extract_summary_totals_via_ocr(pdf)
        if stmt_td is not None and stmt_tc is not None:
            parsed_td, parsed_tc = _sum_tx(tx)
            if abs(parsed_td - stmt_td) > 0.01 or abs(parsed_tc - stmt_tc) > 0.01:
                tx_ocr = parse_bank_islam_ocr_balance_delta(pdf, source_file)
                if tx_ocr:
                    # If OCR yields totals closer to statement, use it
                    o_td, o_tc = _sum_tx(tx_ocr)
                    if abs(o_td - stmt_td) <= 0.01 and abs(o_tc - stmt_tc) <= 0.01:
                        return tx_ocr
                    # Otherwise still prefer OCR if it captured more rows (common for scanned PDFs)
                    if len(tx_ocr) > len(tx):
                        return tx_ocr

        # NEW (safe): if normal parse found nothing, try OCR anyway
        # This does NOT affect months that already parse, and only runs in the OCR-needed branch.
        if not tx:
            tx_ocr = parse_bank_islam_ocr_balance_delta(pdf, source_file)
            if tx_ocr:
                return tx_ocr

    return tx
