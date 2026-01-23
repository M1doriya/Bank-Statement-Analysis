from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

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
                    if "CR" in desc or "CREDIT" in desc or "IN" in desc:
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
    opening_balance = None
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
                    opening_balance = _to_float(money[-1])
                    prev_balance = opening_balance
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

    # Matches "9/10/23"
    DATE_RE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4})")
    # Matches currency patterns like "47,000.00"
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

            # 2) Line starting with date
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
# FIXED INDENTATION + PAGE LOOP
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
            # 1) Opening balance
            if BAL_BF_RE4.search(line):
                nums = MONEY_RE4.findall(line)
                if nums:
                    prev_balance = to_float(nums[-1])
                continue

            # 2) Date-initiated lines
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
# OCR FALLBACK – ONLY USED IF FORMAT 1–4 FAIL
# =========================================================
_OCR_DATE_RE = re.compile(r"^\s*(\d{1,2}/\d{1,2}/\d{2,4})\b")
_OCR_MONEY_RE = re.compile(r"(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}")


def _ocr_to_float(s: str) -> Optional[float]:
    if not s:
        return None
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def parse_bank_islam_ocr(pdf, source_file):
    """
    OCR fallback for scanned/image-based statements (e.g., Feb PDF).
    Runs ONLY when format1–4 return no transactions.
    Uses balance-delta method to infer debit/credit.
    """
    if pytesseract is None:
        return []
    if not pdf.pages:
        return []

    transactions: List[Dict[str, Any]] = []
    prev_balance: Optional[float] = None

    for page_num, page in enumerate(pdf.pages, start=1):
        # Render page image for OCR
        try:
            img = page.to_image(resolution=300).original
        except Exception:
            continue

        ocr_text = pytesseract.image_to_string(img, config="--psm 6") or ""
        lines = [re.sub(r"\s+", " ", l).strip() for l in ocr_text.splitlines() if l.strip()]

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

            balance = _ocr_to_float(nums[-1])
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

            transactions.append({
                "date": date_iso,
                "description": desc,
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(balance, 2),
                "page": page_num,
                "bank": "Bank Islam",
                "source_file": source_file,
                "format": "ocr_fallback"
            })

            prev_balance = balance

    return transactions


# =========================================================
# WRAPPER (FIXED ORDER + FIXED INDENTATION)
# =========================================================
def parse_bank_islam(pdf, source_file):
    # 1) Best: table-based digital statement
    tx = parse_bank_islam_format1(pdf, source_file)
    if tx:
        return tx

    # 2) Text-based statement with balance delta
    tx = parse_bank_islam_format2(pdf, source_file)
    if tx:
        return tx

    # 3) eStatement variants
    tx = parse_bank_islam_format4(pdf, source_file)
    if tx:
        return tx

    tx = parse_bank_islam_format3(pdf, source_file)
    if tx:
        return tx

    # 4) Last resort: OCR (for scanned/corrupted pages)
    return parse_bank_islam_ocr(pdf, source_file)
