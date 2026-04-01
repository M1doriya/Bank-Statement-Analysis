from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from core_utils import normalize_text, safe_float


TX_START_RE = re.compile(
    r"^(?P<day>\d{1,2})\s+(?P<mon>[A-Za-z]{3})\s+(?P<rest>.+)$"
)
AMOUNT_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2}")
TIME_LINE_RE = re.compile(r"^\d{1,2}:\d{2}\s*(?:AM|PM)(?:\s+.*)?$", re.IGNORECASE)

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def _extract_year(text: str, source_file: str) -> int:
    m = re.search(r"\b(20\d{2})\b", text)
    if m:
        return int(m.group(1))

    m = re.search(r"(20\d{2})[-_](?:0?[1-9]|1[0-2])", source_file)
    if m:
        return int(m.group(1))

    return datetime.utcnow().year




def _extract_account_no(text: str) -> Optional[str]:
    m = re.search(r"Principal\s+Account\s*\(([^)]+)\)", text, flags=re.IGNORECASE)
    return normalize_text(m.group(1)) if m else None


def _extract_company_name(text: str) -> Optional[str]:
    m = re.search(
        r"here['’]?s\s+a\s+look\s+at\s+(.+?)(?:['’]s)?\s+performance\b",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    name = normalize_text(m.group(1))
    return name or None

def _to_iso(day: str, mon: str, year: int) -> Optional[str]:
    month = MONTHS.get(mon.upper())
    if not month:
        return None
    try:
        return datetime(year, month, int(day)).strftime("%Y-%m-%d")
    except Exception:
        return None


def parse_transactions_gx_bank(pdf: Any, source_file: str = "") -> List[Dict[str, Any]]:
    bank_name = "GX Bank"
    txs: List[Dict[str, Any]] = []

    full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    year = _extract_year(full_text, source_file)
    account_no = _extract_account_no(full_text)
    company_name = _extract_company_name(full_text)

    prev_balance: Optional[float] = None
    current_tx: Optional[Dict[str, Any]] = None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        if not text:
            continue

        for raw_line in text.splitlines():
            line = normalize_text(raw_line)
            if not line:
                continue

            up = line.upper()
            if (
                line.startswith("Total:")
                or "GX BANK BERHAD" in up
                or up.startswith("PAGE ")
                or "DATE TRANSACTION DETAILS" in up
                or "TARIKH BUTIR URUSNIAGA" in up
                or "STATEMENTS OF ACCOUNTS" in up
                or "MONEY IN" in up and "MONEY OUT" in up
            ):
                current_tx = None
                continue

            m = TX_START_RE.match(line)
            if m:
                day = m.group("day")
                mon = m.group("mon")
                rest = m.group("rest")
                date_iso = _to_iso(day, mon, year)
                if not date_iso:
                    continue

                amount_matches = list(AMOUNT_RE.finditer(rest))
                if not amount_matches:
                    current_tx = None
                    continue

                if len(amount_matches) >= 2:
                    tx_amount = safe_float(amount_matches[-2].group(0))
                    balance = safe_float(amount_matches[-1].group(0))
                    desc = normalize_text(rest[: amount_matches[-2].start()])
                else:
                    value = safe_float(amount_matches[-1].group(0))
                    desc = normalize_text(rest[: amount_matches[-1].start()])
                    if "opening balance" in desc.lower():
                        tx_amount = 0.0
                        balance = value
                    else:
                        tx_amount = value
                        balance = prev_balance if prev_balance is not None else value

                debit = credit = 0.0
                if prev_balance is not None:
                    delta = round(balance - prev_balance, 2)
                    if delta > 0:
                        credit = abs(delta)
                    elif delta < 0:
                        debit = abs(delta)
                    elif tx_amount > 0:
                        if any(k in desc.lower() for k in ("interest", "money in", "credit")):
                            credit = tx_amount
                prev_balance = balance

                current_tx = {
                    "date": date_iso,
                    "description": desc,
                    "debit": round(float(debit), 2),
                    "credit": round(float(credit), 2),
                    "balance": round(float(balance), 2),
                    "page": page_num,
                    "bank": bank_name,
                    "source_file": source_file,
                    "account_no": account_no,
                    "company_name": company_name,
                }
                txs.append(current_tx)
                continue

            if current_tx is None:
                continue

            if TIME_LINE_RE.match(line):
                line = re.sub(r"^\d{1,2}:\d{2}\s*(?:AM|PM)\s*", "", line, flags=re.IGNORECASE).strip()
                if not line:
                    continue
            if line.lower().startswith("note/"):
                current_tx = None
                continue

            current_tx["description"] = normalize_text(f"{current_tx['description']} {line}")

    return txs
