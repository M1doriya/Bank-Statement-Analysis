from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from core_utils import normalize_date, normalize_text, safe_float


BANK_NAME = "MBSB Bank"
_DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
_AMOUNT_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}")
_DATE_PREFIX_RE = re.compile(r"^(?P<date>\d{2}/\d{2}/\d{4})(?:\s+(?P<rest>.*))?$")
_TIME_PREFIX_RE = re.compile(r"^(?P<time>\d{2}:\d{2}:\d{2})(?:\s+(?P<rest>.*))?$")


def _extract_header_meta(full_text: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    account_no: Optional[str] = None
    stmt_end_iso: Optional[str] = None
    prior_balance: Optional[float] = None

    m = re.search(r"Account\s+Number\s*/\s*Currency\s*:\s*([0-9]{8,20})\b", full_text, re.IGNORECASE)
    if m:
        account_no = m.group(1)

    m = re.search(r"Statement\s+Period\s*:\s*\d{2}/\d{2}/\d{4}\s*-\s*(\d{2}/\d{2}/\d{4})", full_text, re.IGNORECASE)
    if m:
        stmt_end_iso = normalize_date(m.group(1))

    m = re.search(r"Prior\s+Day\s+Balance\s*:\s*([-()\d,]+\.\d{2})", full_text, re.IGNORECASE)
    if m:
        prior_balance = safe_float(m.group(1))

    return account_no, stmt_end_iso, prior_balance


def _split_balance_tail(value: str) -> Tuple[Optional[float], Optional[str]]:
    token = normalize_text(value)
    if not token:
        return None, None

    merged = re.match(r"^(-?(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2})(\d{5})$", token)
    if merged:
        return safe_float(merged.group(1)), merged.group(2)

    if re.fullmatch(r"\d{5}", token):
        return None, token

    if _AMOUNT_RE.fullmatch(token):
        return safe_float(token), None

    return None, None


def _parse_row_payload(payload: str, prev_balance: Optional[float]) -> Tuple[str, float, float, float]:
    text = normalize_text(payload)
    if not text:
        return "", 0.0, 0.0, 0.0

    tokens = text.split()
    balance = 0.0
    branch_code: Optional[str] = None

    # Parse the right-most balance/branch code area first.
    if tokens:
        bal, branch = _split_balance_tail(tokens[-1])
        if bal is not None:
            balance = float(bal)
            branch_code = branch
            tokens = tokens[:-1]
        elif branch is not None:
            branch_code = branch
            tokens = tokens[:-1]
            if tokens and _AMOUNT_RE.fullmatch(tokens[-1]):
                balance = float(safe_float(tokens[-1]))
                tokens = tokens[:-1]

    if balance == 0.0:
        amounts = list(_AMOUNT_RE.finditer(text))
        if amounts:
            balance = float(safe_float(amounts[-1].group(0)))
            text = normalize_text(text[: amounts[-1].start()])
            tokens = text.split()

    # Gather amount candidates that could represent debit/credit amount.
    amount_values: List[float] = []
    amount_positions: List[int] = []
    for i, tok in enumerate(tokens):
        if _AMOUNT_RE.fullmatch(tok):
            amount_values.append(float(safe_float(tok)))
            amount_positions.append(i)

    debit = 0.0
    credit = 0.0
    if prev_balance is not None and balance != 0.0:
        delta = round(balance - prev_balance, 2)
        if delta > 0:
            credit = abs(delta)
        elif delta < 0:
            debit = abs(delta)

    if debit == 0.0 and credit == 0.0 and amount_values:
        txn_amt = abs(float(amount_values[-1]))
        lower = text.lower()
        if any(k in lower for k in ("profit", "hibah", "credit", "refund")):
            credit = txn_amt
        else:
            debit = txn_amt

    # Build description by removing trailing amount and structural tokens.
    if amount_positions:
        tokens = tokens[: amount_positions[0]]
    desc = normalize_text(" ".join(t for t in tokens if t.upper() != "NA"))
    if branch_code:
        desc = normalize_text(f"{desc} BRANCH {branch_code}")

    return desc, round(debit, 2), round(credit, 2), round(balance, 2)


def parse_transactions_mbsb(pdf: Any, source_file: str = "") -> List[Dict[str, Any]]:
    full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    account_no, statement_end_iso, prev_balance = _extract_header_meta(full_text)

    transactions: List[Dict[str, Any]] = []
    current_date: Optional[str] = None
    row_lines: List[str] = []

    def flush(page_num: int) -> None:
        nonlocal row_lines, current_date, prev_balance
        if not current_date:
            row_lines = []
            return
        payload = normalize_text(" ".join(row_lines))
        if not payload:
            row_lines = []
            return

        desc, debit, credit, balance = _parse_row_payload(payload, prev_balance)
        tx = {
            "date": normalize_date(current_date) or current_date,
            "description": desc or "(NO DESCRIPTION)",
            "debit": float(debit),
            "credit": float(credit),
            "balance": float(balance),
            "page": page_num,
            "bank": BANK_NAME,
            "source_file": source_file,
            "account_no": account_no,
        }
        transactions.append(tx)
        prev_balance = float(balance)
        row_lines = []

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        if not text:
            continue

        for raw_line in text.splitlines():
            line = normalize_text(raw_line)
            if not line:
                continue
            upper = line.upper()

            if upper.startswith("DISCLAIMER") or upper.startswith("PAGE "):
                flush(page_num)
                current_date = None
                continue

            if (
                "TRANSACTION DETAILS" in upper
                or "REPORT GENERATED ON" in upper
                or "ACCOUNT NUMBER /CURRENCY" in upper
                or "STATEMENT PERIOD" in upper
                or "BRANCH NAME" in upper
                or "PRIOR DAY BALANCE" in upper
                or "TRANSACTION RECIPIENT" in upper
                or "TRANSACTION REMARKS" in upper
            ):
                continue

            m_date = _DATE_PREFIX_RE.match(line)
            if m_date:
                flush(page_num)
                current_date = m_date.group("date")
                tail = normalize_text(m_date.group("rest") or "")
                if tail:
                    row_lines.append(tail)
                continue

            m_time = _TIME_PREFIX_RE.match(line)
            if m_time:
                tail = normalize_text(m_time.group("rest") or "")
                if current_date and tail:
                    row_lines.append(tail)
                continue

            if current_date:
                row_lines.append(line)

        flush(page_num)
        current_date = None

    if not transactions and statement_end_iso and prev_balance is not None:
        transactions.append(
            {
                "date": statement_end_iso,
                "description": "NO TRANSACTIONS (PRIOR DAY BALANCE)",
                "debit": 0.0,
                "credit": 0.0,
                "balance": round(float(prev_balance), 2),
                "page": None,
                "bank": BANK_NAME,
                "source_file": source_file,
                "account_no": account_no,
                "is_statement_balance": True,
            }
        )

    return transactions
