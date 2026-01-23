from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None


# =========================================================
# BANK ISLAM – FORMAT 1 (TABLE-BASED)
# =========================================================
def parse_bank_islam_format1(pdf, source_file):
    transactions = []

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
                no, txn_date, customer_eft, txn_code, description,
                ref_no, branch, debit_raw, credit_raw,
                balance_raw, sender_recipient, payment_details
            ) = row[:12]

            if not txn_date or not re.search(r"\d{2}/\d{2}/\d{4}", str(txn_date)):
                continue

            try:
                date = datetime.strptime(
                    re.search(r"\d{2}/\d{2}/\d{4}", str(txn_date)).group(),
                    "%d/%m/%Y"
                ).date().isoformat()
            except Exception:
                continue

            debit = extract_amount(debit_raw) or 0.0
            credit = extract_amount(credit_raw) or 0.0
            balance = extract_amount(balance_raw) or 0.0

            # Recovery from description
            if debit == 0.0 and credit == 0.0:
                recovered = extract_amount(description)
                if recovered:
                    desc = str(description).upper()
                    if "CR" in desc or "CREDIT" in desc or " IN " in f" {desc} ":
                        credit = recovered
                    else:
                        debit = recovered

            description_clean = " ".join(
                str(x).replace("\n", " ").strip()
                for x in [no, txn_code, description, sender_recipient, payment_details]
                if x and str(x).lower() != "nan"
            )

            transactions.append({
                "date": date,
                "description": description_clean,
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(balance, 2),
                "page": page_num,
                "bank": "Bank Islam",
                "source_file": source_file,
                "format": "format1"
            })

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


def _parse_date(d):
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(d.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def parse_bank_islam_format2(pdf, source_file):
    transactions = []
    prev_balance = None

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
            desc = line[len(m_date.group(1)):].strip()
            for tok in money_raw:
                desc = desc.replace(tok, "").strip()

            transactions.append({
                "date": date,
                "description": desc,
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(balance, 2),
                "page": page_num,
                "bank": "Bank Islam",
                "source_file": source_file,
                "format": "format2_balance_delta"
            })

    return transactions


# =========================================================
# FORMAT 3 – eSTATEMENT (BALANCE DELTA, DIFFERENT LAYOUT)
# =========================================================
def parse_bank_islam_format3(pdf, source_file):
    transactions = []
    prev_balance = None

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
            if BAL_BF_RE3.search(line):
                nums = MONEY_RE3.findall(line)
                if nums:
                    prev_balance = to_float(nums[-1])
                continue

            date_match = DATE_RE.match(line)
            if date_match and prev_balance is not None:
                raw_date = date_match.group(1)
                nums = MONEY_RE3.findall(line)

                if len(nums) >= 2:
                    balance = to_float(nums[-1])

                    desc = line.replace(raw_date, "").strip()
                    for n in nums:
                        desc = desc.replace(n, "").strip()

                    delta = round(balance - prev_balance, 2)

                    transactions.append({
                        "date": parse_date(raw_date),
                        "description": desc,
                        "debit": abs(delta) if delta < 0 else 0.0,
                        "credit": delta if delta > 0 else 0.0,
                        "balance": balance,
                        "page": page_num,
                        "bank": "Bank Islam",
                        "source_file": source_file,
                        "format": "format3_estatement"
                    })

                    prev_balance = balance

    return transactions


# =========================================================
# FORMAT 4 – eSTATEMENT (BALANCE DELTA, DIFFERENT LAYOUT)
# FIXED: indentation + per-page processing
# =========================================================
def parse_bank_islam_format4(pdf, source_file):
    transactions = []
    prev_balance = None

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
            if BAL_BF_RE4.search(line):
                nums = MONEY_RE4.findall(line)
                if nums:
                    prev_balance = to_float(nums[-1])
                continue

            date_match = DATE_RE.match(line)
            if date_match and prev_balance is not None:
                raw_date = date_match.group(1)
                nums = MONEY_RE4.findall(line)

                if nums:
                    current_balance = to_float(nums[-1])
                    delta = round(current_balance - prev_balance, 2)

                    desc = line.replace(raw_date, "").strip()
                    for n in nums:
                        desc = desc.replace(n, "").strip()

                    transactions.append({
                        "date": parse_date(raw_date),
                        "description": desc,
                        "debit": abs(delta) if delta < 0 else 0.0,
                        "credit": delta if delta > 0 else 0.0,
                        "balance": current_balance,
                        "page": page_num,
                        "bank": "Bank Islam",
                        "source_file": source_file,
                        "format": "format4_normalized"
                    })

                    prev_balance = current_balance

    return transactions


# =========================================================
# OCR PARSER (BALANCE-DELTA) FOR SCANNED STATEMENTS
# - Only used when needed (validation fails / scanned)
# =========================================================
_OCR_DATE_RE = re.compile(r"^\s*(\d{1,2}/\d{1,2}/\d{2,4})\b")
_OCR_MONEY_RE = re.compile(r"(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}")

def _ocr_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None

def _ocr_text_page(page: pdfplumber.page.Page, psm: str = "6") -> str:
    if pytesseract is None:
        return ""
    img = page.to_image(resolution=300).original
    return pytesseract.image_to_string(img, config=f"--psm {psm}") or ""

def parse_bank_islam_ocr_balance_delta(pdf, source_file) -> List[Dict[str, Any]]:
    """
    Robust for scanned statements:
    - Extract BAL B/F opening
    - Extract every date line and its ending balance (last money token)
    - Derive debit/credit by balance delta (fixes OCR misread amounts)
    """
    if pytesseract is None or not pdf.pages:
        return []

    tx: List[Dict[str, Any]] = []
    prev_balance: Optional[float] = None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = _ocr_text_page(page, psm="6")
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]

        # Opening balance (try to find BAL B/F line)
        if prev_balance is None:
            for line in lines:
                if re.search(r"\bBAL\b\s*B/F\b", line, re.I):
                    nums = _OCR_MONEY_RE.findall(line)
                    if nums:
                        prev_balance = _ocr_float(nums[-1])
                    break

        for line in lines:
            dm = _OCR_DATE_RE.match(line)
            if not dm:
                continue

            raw_date = dm.group(1)
            date_iso = _parse_date(raw_date)
            if not date_iso:
                continue

            nums = _OCR_MONEY_RE.findall(line)
            if not nums:
                continue

            balance = _ocr_float(nums[-1])
            if balance is None:
                continue

            debit = credit = 0.0
            if prev_balance is not None:
                delta = round(balance - prev_balance, 2)
                if delta > 0:
                    credit = delta
                elif delta < 0:
                    debit = abs(delta)

            desc = line.replace(raw_date, "").strip()
            for n in nums:
                desc = desc.replace(n, "").strip()

            tx.append({
                "date": date_iso,
                "description": desc,
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(balance, 2),
                "page": page_num,
                "bank": "Bank Islam",
                "source_file": source_file,
                "format": "ocr_balance_delta"
            })

            prev_balance = balance

    return tx


# =========================================================
# VALIDATION AGAINST STATEMENT SUMMARY (TOTAL DEBIT/CREDIT)
# Used to decide whether to re-parse scanned December
# =========================================================
def _extract_summary_totals_via_ocr(pdf) -> Tuple[Optional[float], Optional[float]]:
    """
    OCR the first page and try to find:
      TOTAL DEBIT <amount>
      TOTAL CREDIT <amount>
    Returns (total_debit, total_credit)
    """
    if pytesseract is None or not pdf.pages:
        return (None, None)

    text = _ocr_text_page(pdf.pages[0], psm="6")
    # Allow spaces/newlines between tokens
    text_norm = re.sub(r"\s+", " ", text).upper()

    def find_total(label: str) -> Optional[float]:
        # Examples: "TOTAL DEBIT 5 1,056,250.20"
        m = re.search(rf"{label}\s+\d+\s+((?:\d{{1,3}}(?:,\d{{3}})*)\.\d{{2}})", text_norm)
        if m:
            return float(m.group(1).replace(",", ""))
        return None

    td = find_total("TOTAL DEBIT")
    tc = find_total("TOTAL CREDIT")
    return td, tc


def _sum_tx(tx: List[Dict[str, Any]]) -> Tuple[float, float]:
    return (
        round(sum(t.get("debit", 0.0) or 0.0 for t in tx), 2),
        round(sum(t.get("credit", 0.0) or 0.0 for t in tx), 2),
    )


def _looks_like_scanned(source_file: str, pdf: pdfplumber.PDF) -> bool:
    # If pdfplumber text extraction is basically empty, it’s likely scanned
    try:
        sample = (pdf.pages[0].extract_text() or "").strip()
    except Exception:
        sample = ""
    if len(sample) < 20:
        return True
    return False


# =========================================================
# WRAPPER
# - Keeps existing behavior for other months
# - Fixes scanned December by validating and re-parsing only when needed
# =========================================================
def parse_bank_islam(pdf, source_file):
    # 1) Primary parsers (unchanged behavior for normal months)
    tx = parse_bank_islam_format1(pdf, source_file)
    if not tx:
        tx = parse_bank_islam_format2(pdf, source_file)
    if not tx:
        tx = parse_bank_islam_format4(pdf, source_file)
    if not tx:
        tx = parse_bank_islam_format3(pdf, source_file)

    # 2) Targeted fix: scanned statements (your Dec file is scanned)
    # Only override if:
    #   - looks scanned OR filename indicates scanned
    #   - and parsed totals do not match statement totals (from OCR)
    scanned_hint = ("scan" in source_file.lower()) or _looks_like_scanned(source_file, pdf)

    if scanned_hint:
        stmt_td, stmt_tc = _extract_summary_totals_via_ocr(pdf)
        if stmt_td is not None and stmt_tc is not None:
            parsed_td, parsed_tc = _sum_tx(tx)

            # If mismatch, re-parse with OCR balance-delta (fixes partial/incorrect OCR amounts)
            if abs(parsed_td - stmt_td) > 0.01 or abs(parsed_tc - stmt_tc) > 0.01:
                tx_ocr = parse_bank_islam_ocr_balance_delta(pdf, source_file)
                if tx_ocr:
                    # Re-check; accept if better (or exactly matches)
                    o_td, o_tc = _sum_tx(tx_ocr)
                    if (abs(o_td - stmt_td) <= 0.01 and abs(o_tc - stmt_tc) <= 0.01) or len(tx_ocr) > len(tx):
                        return tx_ocr

    return tx
