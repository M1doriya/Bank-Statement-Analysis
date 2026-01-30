# alliance_bank.py
# Alliance Bank Malaysia Berhad statement parser
#
# Key traits in the provided PDFs:
# - Transaction rows start with 6-digit date: DDMMYY (e.g. 010625)
# - "BEGINNING BALANCE" and "ENDING BALANCE" rows exist
# - Debit/Credit amount is inferred from running balance delta (balance-driven)
#
# Output schema (matches common pattern across this project):
# {
#   "date": "YYYY-MM-DD",
#   "description": "...",
#   "debit": float,
#   "credit": float,
#   "balance": float,
#   "bank": "Alliance Bank",
#   "page": int,
#   "source_file": str
# }

from __future__ import annotations

import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import pdfplumber


# -----------------------------
# Regex
# -----------------------------
_TX_START_RE = re.compile(r"^(?P<d>\d{2})(?P<m>\d{2})(?P<y>\d{2})\s+(?P<rest>.+)$")
_MONEY_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*\.\d{2}|\-?\d+\.\d{2}")

# Lines that should be ignored (headers/boilerplate)
_HEADER_SUBSTRS = (
    "STATEMENT OF ACCOUNT",
    "PENYATA AKAUN",
    "PAGE ",
    "HALAMAN ",
    "CURRENT A/C",
    "ACCOUNT NO",
    "NO. AKAUN",
    "CURRENCY",
    "MATAWANG",
    "PROTECTED BY PIDM",
    "DILINDUNGI",
    "CIF NO",
)

# Footer/stop markers: once we hit these, we stop appending to the current txn description
_STOP_MARKERS = (
    "THE ITEMS AND BALANCES SHOWN ABOVE WILL BE DEEMED CORRECT",
    "SEGALA BUTIRAN DAN BAKI AKAUN PENYATA DI ATAS DIANGGAP BETUL",
    "ALLIANCE BANK MALAYSIA BERHAD",
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\x00", " ")).strip()


def _is_noise(line: str) -> bool:
    up = _norm(line).upper()
    if not up:
        return True

    if any(k in up for k in _HEADER_SUBSTRS):
        return True

    # Column header line (varies slightly across pages)
    if (
        "TRANSACTION DETAILS" in up
        and "CHEQUE" in up
        and "DEBIT" in up
        and "CREDIT" in up
        and "BALANCE" in up
    ):
        return True

    # Another common header line
    if up.startswith("DATE TRANSACTION DETAILS") or up.startswith("TARIKH KETERANGAN"):
        return True

    return False


def _is_stop(line: str) -> bool:
    up = _norm(line).upper()
    return any(m in up for m in _STOP_MARKERS)


def _parse_money_tokens(text: str) -> List[float]:
    out: List[float] = []
    for m in _MONEY_RE.finditer(text):
        try:
            out.append(float(m.group().replace(",", "")))
        except Exception:
            continue
    return out


def _iso_from_ddmmyy(dd: str, mm: str, yy: str) -> str:
    # In your PDFs, YY is 25 for 2025
    y = 2000 + int(yy)
    return f"{y:04d}-{int(mm):02d}-{int(dd):02d}"


def _strip_trailing_amounts(s: str) -> str:
    """
    Remove trailing "... <amt> <bal> CR" from the *first line* of a txn
    so description doesn't permanently contain money tokens.
    """
    t = _norm(s)
    t = re.sub(r"\s+\b(CR|DR)\b\s*$", "", t, flags=re.I)
    t = re.sub(r"\s+-?\d[\d,]*\.\d{2}\s+-?\d[\d,]*\.\d{2}\s*$", "", t)
    t = re.sub(r"\s+-?\d[\d,]*\.\d{2}\s*$", "", t)
    return _norm(t)


def parse_transactions_alliance(
    pdf_input: Union[bytes, bytearray, str, Any],
    source_file: str = "",
) -> List[Dict[str, Any]]:
    """
    pdf_input can be:
      - bytes / bytearray
      - file path (str)
      - file-like object (e.g. Streamlit UploadedFile)
    """

    # -----------------------------
    # Load PDF
    # -----------------------------
    pdf: pdfplumber.PDF
    if isinstance(pdf_input, (bytes, bytearray)):
        pdf = pdfplumber.open(BytesIO(pdf_input))
    elif isinstance(pdf_input, str):
        pdf = pdfplumber.open(pdf_input)
        if not source_file:
            source_file = pdf_input
    else:
        # file-like (Streamlit UploadedFile etc.)
        data = pdf_input.read()
        pdf = pdfplumber.open(BytesIO(data))
        if not source_file:
            source_file = getattr(pdf_input, "name", "") or ""

    # -----------------------------
    # Extract raw grouped rows
    # -----------------------------
    raw_rows: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for page_no, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        for raw_line in text.splitlines():
            line = _norm(raw_line)
            if _is_noise(line):
                continue

            # Stop markers: finalize current txn and stop appending boilerplate
            if _is_stop(line):
                if current:
                    raw_rows.append(current)
                    current = None
                continue

            m = _TX_START_RE.match(line)
            if m:
                # finalize previous txn
                if current:
                    raw_rows.append(current)

                date_iso = _iso_from_ddmmyy(m.group("d"), m.group("m"), m.group("y"))
                rest = m.group("rest")

                vals = _parse_money_tokens(line)

                current = {
                    "date": date_iso,
                    "description_parts": [_strip_trailing_amounts(rest)],
                    "amount": None,   # transaction amount (unknown sign)
                    "balance": None,  # printed balance on row
                    "page": page_no,
                }

                if len(vals) >= 2:
                    current["amount"] = vals[-2]
                    current["balance"] = vals[-1]
                elif len(vals) == 1:
                    # BEGINNING/ENDING balance lines often only have one amount (balance)
                    current["balance"] = vals[-1]

                continue

            # Continuation line (belongs to current transaction)
            if not current:
                continue

            # Ignore a mid-table header fragment if it slips through
            up = line.upper()
            if "DATE" in up and "TRANSACTION" in up and "DETAILS" in up:
                continue

            current["description_parts"].append(line)

            # Sometimes the amount/balance tokens may appear on a subsequent line
            if current.get("balance") is None:
                vals = _parse_money_tokens(line)
                if len(vals) >= 2:
                    current["amount"] = vals[-2]
                    current["balance"] = vals[-1]
                elif len(vals) == 1:
                    current["balance"] = vals[-1]

    if current:
        raw_rows.append(current)

    pdf.close()

    # -----------------------------
    # Convert to final transactions (balance-driven debit/credit)
    # -----------------------------
    out: List[Dict[str, Any]] = []
    prev_balance: Optional[float] = None

    for r in raw_rows:
        desc = _norm(" ".join(r.get("description_parts") or []))
        desc_up = desc.upper()

        bal = r.get("balance")
        amt = r.get("amount")
        page_no = int(r.get("page") or 0)

        # BEGINNING BALANCE row
        if "BEGINNING BALANCE" in desc_up and isinstance(bal, (int, float)):
            prev_balance = float(bal)
            out.append(
                {
                    "date": r["date"],
                    "description": "BEGINNING BALANCE",
                    "debit": 0.0,
                    "credit": 0.0,
                    "balance": float(bal),
                    "bank": "Alliance Bank",
                    "page": page_no,
                    "source_file": source_file,
                }
            )
            continue

        # ENDING BALANCE row
        if "ENDING BALANCE" in desc_up and isinstance(bal, (int, float)):
            debit = credit = 0.0
            if isinstance(prev_balance, (int, float)):
                delta = round(float(bal) - float(prev_balance), 2)
                if delta >= 0:
                    credit = abs(delta)
                else:
                    debit = abs(delta)
            prev_balance = float(bal)

            out.append(
                {
                    "date": r["date"],
                    "description": "ENDING BALANCE",
                    "debit": float(debit),
                    "credit": float(credit),
                    "balance": float(bal),
                    "bank": "Alliance Bank",
                    "page": page_no,
                    "source_file": source_file,
                }
            )
            continue

        # Normal transactions
        debit = credit = 0.0

        if isinstance(prev_balance, (int, float)) and isinstance(bal, (int, float)):
            delta = round(float(bal) - float(prev_balance), 2)
            if delta >= 0:
                credit = abs(delta)
            else:
                debit = abs(delta)
        else:
            # Fallback if no beginning balance was found
            # (should rarely happen in your provided PDFs)
            if any(k in desc_up for k in ("CR ADVICE", "DUITNOW CR", "TRANSFER FROM", "LOCAL CHQ DEP", "CREDIT")):
                credit = float(amt or 0.0)
            else:
                debit = float(amt or 0.0)

        if isinstance(bal, (int, float)):
            prev_balance = float(bal)

        out.append(
            {
                "date": r["date"],
                "description": desc,
                "debit": float(debit),
                "credit": float(credit),
                "balance": float(bal) if isinstance(bal, (int, float)) else None,
                "bank": "Alliance Bank",
                "page": page_no,
                "source_file": source_file,
            }
        )

    return out
