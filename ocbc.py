# ocbc.py
# OCBC Bank (Malaysia) - Current Account statement parser
#
# Supports statements where transactions appear as text lines like:
#   Balance B/F 2.00
#   02 MAY 2023 DUITNOW(INST TRF) CR /IB 500.00 502.00
#   LF SERVICES SDN. BH
#   DESC:
#   REF: PV42/2023/LFS
#
# Also supports debit lines like:
#   08 MAY 2023 DUITNOW(INST TRF) DR /IB 17,371.00 202,582.00
#
# Output schema aligns with app.py/core_utils normalization:
#   date, description, debit, credit, balance, page, bank, source_file

from __future__ import annotations

import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from core_utils import normalize_text, safe_float


# --- Patterns ---
TX_START_RE = re.compile(
    r"^(?P<day>\d{2})\s+"
    r"(?P<mon>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+"
    r"(?P<year>\d{4})\s+"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)

BAL_BF_RE = re.compile(r"\bBalance\s+B/F\b\s+(?P<bal>-?[\d,]+\.\d{2})", re.IGNORECASE)

MONEY_RE = re.compile(r"^-?(\d{1,3}(?:,\d{3})*|\d+)\.\d{2}$")

STOP_LINES = (
    "TRANSACTION",
    "SUMMARY",
    "NO. OF WITHDRAWALS",
    "NO. OF DEPOSITS",
    "TOTAL WITHDRAWALS",
    "TOTAL DEPOSITS",
    "HOLD AMOUNT",
    "LATE LOCAL CHEQUE",
    "PAGE ",
    "STATEMENT OF CURRENT ACCOUNT",
    "PENYATA AKAUN SEMASA",
    "TRANSACTION DATE",
    "TARIKH TRANSAKSI",
    "TRANSACTION DESCRIPTION",
    "HURAIAN TRANSAKSI",
)

# classification hints
CREDIT_HINTS = (" CR ", "CR /IB", "CR INWARD", "CREDIT")
DEBIT_HINTS = (" DR ", "DR /IB", "DEBIT", "DUITNOW SC", "DEBIT AS ADVISED")


def _to_iso_date(day: str, mon: str, year: str) -> str:
    mon = mon.upper()
    month_map = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    mm = month_map.get(mon, "01")
    return f"{year}-{mm}-{day}"


def _extract_amount_and_balance_from_line(rest: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    From the 'rest' part of the first transaction line (after date),
    extract:
      - tx_amount (usually the first money token from the right that isn't the balance)
      - balance (last money token)
      - desc_text (rest with trailing numeric columns removed)
    """
    tokens = rest.split()
    money_idx = [i for i, t in enumerate(tokens) if MONEY_RE.match(t)]
    if len(money_idx) < 2:
        return None, None, rest

    # Last money token is balance
    bal_token = tokens[money_idx[-1]]
    balance = safe_float(bal_token)

    # The preceding money token is usually transaction amount
    amt_token = tokens[money_idx[-2]]
    tx_amount = safe_float(amt_token)

    # Remove trailing money tokens and anything after first trailing money block
    cut = money_idx[-2]  # keep description up to before amount
    desc_tokens = tokens[:cut]
    desc_text = " ".join(desc_tokens).strip()

    return tx_amount, balance, desc_text


def _is_noise_line(line: str) -> bool:
    up = line.upper().strip()
    if not up:
        return True
    return any(k in up for k in STOP_LINES)


def parse_transactions_ocbc(pdf_input: Any, source_file: str = "") -> List[Dict]:
    """
    Standard interface used by app.py:
      input: pdf bytes (preferred) OR file-like
      output: list of tx dicts with canonical keys
    """
    # open pdfplumber from bytes or file-like
    if isinstance(pdf_input, (bytes, bytearray)):
        pdf = pdfplumber.open(BytesIO(bytes(pdf_input)))
    else:
        pdf = pdfplumber.open(pdf_input)

    bank_name = "OCBC Bank"

    transactions: List[Dict] = []
    prev_balance: Optional[float] = None
    latest_date_iso: Optional[str] = None

    current_tx: Optional[Dict] = None

    try:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text:
                continue

            # find opening balance once
            if prev_balance is None:
                bf = BAL_BF_RE.search(text)
                if bf:
                    prev_balance = safe_float(bf.group("bal"))

            for raw_line in text.splitlines():
                line = normalize_text(raw_line)
                if not line:
                    continue

                # stop when summary starts
                if "TRANSACTION" in line.upper() and "SUMMARY" in line.upper():
                    current_tx = None
                    break

                m = TX_START_RE.match(line)
                if m:
                    # finalize previous tx (already appended when started)
                    day, mon, year = m.group("day"), m.group("mon"), m.group("year")
                    rest = m.group("rest")

                    date_iso = _to_iso_date(day, mon, year)
                    tx_amount, balance, desc_head = _extract_amount_and_balance_from_line(rest)

                    # If we can't find amount/balance, skip (rare)
                    if tx_amount is None or balance is None:
                        current_tx = None
                        continue

                    desc_upper = desc_head.upper()

                    debit = 0.0
                    credit = 0.0

                    # 1) keyword-driven classification
                    if any(h in f" {desc_upper} " for h in CREDIT_HINTS) and not any(h in f" {desc_upper} " for h in (" DR ", "DR /IB")):
                        credit = abs(tx_amount)
                    elif any(h in f" {desc_upper} " for h in DEBIT_HINTS):
                        debit = abs(tx_amount)
                    # 2) balance-delta fallback
                    elif prev_balance is not None:
                        delta = round(balance - prev_balance, 2)
                        # if the delta matches amount, classify
                        if abs(delta - tx_amount) <= 0.05:
                            credit = abs(tx_amount)
                        elif abs(delta + tx_amount) <= 0.05:
                            debit = abs(tx_amount)
                        else:
                            # if unclear, infer from sign of delta
                            if delta > 0:
                                credit = abs(delta)
                            elif delta < 0:
                                debit = abs(delta)

                    tx = {
                        "date": date_iso,
                        "description": desc_head,
                        "debit": round(float(debit), 2),
                        "credit": round(float(credit), 2),
                        "balance": round(float(balance), 2),
                        "page": page_idx,
                        "bank": bank_name,
                        "source_file": source_file,
                    }

                    transactions.append(tx)
                    current_tx = tx
                    prev_balance = balance
                    latest_date_iso = max(latest_date_iso, date_iso) if latest_date_iso else date_iso
                    continue

                # continuation lines
                if current_tx is not None and not _is_noise_line(line):
                    # Avoid accidentally appending numeric-only lines
                    if not MONEY_RE.match(line.replace(",", "")):
                        current_tx["description"] = normalize_text(current_tx["description"] + " " + line)

        return transactions

    finally:
        try:
            pdf.close()
        except Exception:
            pass
