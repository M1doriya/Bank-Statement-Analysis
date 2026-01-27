# cimb.py - Standalone CIMB Bank Parser
#
# Primary strategy:
#   1) Try table extraction (fast).
#   2) If table extraction fails / yields no transactions, fall back to robust text parsing.
#
# FIX (CIMB accuracy):
# - Extract statement footer totals (opening/closing/total debit/total credit) by scanning FULL PDF text.
# - Expose extract_cimb_statement_totals() for app.py monthly summary (prevents partial tx extraction from breaking totals).
# - Still appends a "CLOSING BALANCE / BAKI PENUTUP" row with is_statement_balance=True for ending-balance correctness.

import re
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

# ---------------------------------------------------------
# STATEMENT DATE (YEAR / MONTH)
# ---------------------------------------------------------

_STMT_DATE_RE = re.compile(
    r"(?:STATEMENT\s+DATE|TARIKH\s+PENYATA)\s*[:\s]+(\d{1,2})/(\d{1,2})/(\d{2,4})",
    re.IGNORECASE,
)


def extract_year_from_text(text: str) -> Optional[str]:
    """Extract year from statement header (Statement Date / Tarikh Penyata)."""
    if not text:
        return None
    m = _STMT_DATE_RE.search(text)
    if not m:
        return None
    yy_raw = m.group(3)
    return yy_raw if len(yy_raw) == 4 else str(2000 + int(yy_raw))


def extract_statement_month_from_text(text: str) -> Optional[str]:
    """Return statement month in YYYY-MM (based on Statement Date)."""
    if not text:
        return None
    m = _STMT_DATE_RE.search(text)
    if not m:
        return None
    _dd, mm, yy_raw = m.groups()
    yy = (2000 + int(yy_raw)) if len(yy_raw) == 2 else int(yy_raw)
    mm_i = int(mm)
    if 1 <= mm_i <= 12 and 2000 <= yy <= 2100:
        return f"{yy:04d}-{mm_i:02d}"
    return None


# ---------------------------------------------------------
# STATEMENT FOOTER: OPENING / CLOSING / TOTALS
# ---------------------------------------------------------

def extract_closing_balance_from_text(text: str) -> Optional[float]:
    """
    Extract:
      CLOSING BALANCE / BAKI PENUTUP 488,584.58
    Supports negative balances.
    """
    if not text:
        return None
    m = re.search(
        r"CLOSING\s+BALANCE\s*/\s*BAKI\s+PENUTUP\s+(-?[\d,]+\.\d{2})",
        text,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def extract_opening_balance_from_text(text: str) -> Optional[float]:
    """
    Extract:
      Opening Balance 410,883.69
    """
    if not text:
        return None
    m = re.search(r"\bOPENING\s+BALANCE\b\s+(-?[\d,]+\.\d{2})", text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def extract_total_withdrawal_deposit_from_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract totals printed in footer.

    Footer typically contains a final numeric line:
      302 35 2,058,960.56 2,136,661.45
    (counts + totals). We capture the LAST such line in the document.
    """
    if not text:
        return None, None

    rx = re.compile(
        r"\b(\d{1,4})\s+(\d{1,4})\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})\b"
    )
    matches = list(rx.finditer(text))
    if not matches:
        return None, None

    m = matches[-1]
    total_withdrawal = float(m.group(3).replace(",", ""))
    total_deposit = float(m.group(4).replace(",", ""))
    return total_withdrawal, total_deposit


def extract_cimb_statement_totals(pdf, source_filename: str = "") -> Dict[str, Any]:
    """
    Extract statement footer totals for monthly summary:
      opening_balance, ending_balance, total_debit (withdrawal), total_credit (deposit), statement_month.
    """
    try:
        all_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        all_text = ""

    total_debit, total_credit = extract_total_withdrawal_deposit_from_text(all_text)

    return {
        "source_file": source_filename,
        "statement_month": extract_statement_month_from_text(all_text),
        "opening_balance": extract_opening_balance_from_text(all_text),
        "ending_balance": extract_closing_balance_from_text(all_text),
        "total_debit": total_debit,
        "total_credit": total_credit,
    }


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def parse_float(value) -> float:
    """Converts '1,234.56' or '-1,234.56' to float. Returns 0.0 if invalid."""
    if value is None:
        return 0.0
    clean = str(value).replace("\n", "").replace(" ", "").replace(",", "")
    if clean == "":
        return 0.0
    if not re.match(r"^-?\d+(\.\d+)?$", clean):
        return 0.0
    try:
        return float(clean)
    except Exception:
        return 0.0


def clean_text(text) -> str:
    if not text:
        return ""
    return str(text).replace("\n", " ").strip()


def format_date(date_str: str, year: str) -> Optional[str]:
    """
    Format date string to YYYY-MM-DD.
    Handles DD/MM/YYYY and DD/MM.
    """
    if not date_str:
        return None

    date_str = clean_text(date_str)

    m = re.match(r"(\d{2})/(\d{2})/(\d{4})$", date_str)
    if m:
        dd, mm, yyyy = m.groups()
        return f"{yyyy}-{mm}-{dd}"

    m = re.match(r"(\d{2})/(\d{2})$", date_str)
    if m:
        dd, mm = m.groups()
        return f"{year}-{mm}-{dd}"

    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    return None


# ---------------------------------------------------------
# FALLBACK TEXT PARSER (only used if table parsing fails)
# ---------------------------------------------------------

_MONEY_TOKEN_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*\.\d{2}$")


def _extract_last_balance_token(line: str):
    """
    Finds the last money token on the line (e.g., balance), returns (balance_float, first_money_index).
    """
    toks = line.split()
    last_idx = None
    for i in range(len(toks) - 1, -1, -1):
        if _MONEY_TOKEN_RE.match(toks[i]):
            last_idx = i
            break
    if last_idx is None:
        return None, None

    bal = parse_float(toks[last_idx])

    first_money_idx = None
    for i, t in enumerate(toks):
        if t == "0" or _MONEY_TOKEN_RE.match(t):
            first_money_idx = i
            break

    return bal, first_money_idx


def _parse_transactions_cimb_text(
    pdf,
    source_filename: str,
    detected_year: str,
    bank_name: str = "CIMB Bank",
    closing_balance: Optional[float] = None,
):
    """
    Robust text-mode parser for CIMB statements.
    Uses balance delta to infer debit/credit.
    """
    transactions = []
    prev_balance = None
    latest_tx_date = None
    cur = None  # {"date":..., "parts":[...], "page":...}

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for ln in lines:
            up = ln.upper()

            if up.startswith("OPENING BALANCE"):
                bal, _ = _extract_last_balance_token(ln)
                if bal is not None:
                    prev_balance = bal
                continue

            if "CLOSING BALANCE" in up and "BAKI" in up:
                continue

            m = re.match(r"^(\d{2}/\d{2}/\d{4})\s+(.*)$", ln)
            if m:
                cur = {"date": m.group(1), "parts": [m.group(2)], "page": page_num}

                bal, first_money_idx = _extract_last_balance_token(ln)
                if bal is not None:
                    toks = ln.split()
                    desc = " ".join(toks[1:first_money_idx]) if first_money_idx is not None else " ".join(toks[1:])
                    date_iso = format_date(cur["date"], detected_year)
                    if date_iso:
                        debit = credit = 0.0
                        if prev_balance is not None:
                            delta = round(bal - prev_balance, 2)
                            if delta > 0:
                                credit = delta
                            elif delta < 0:
                                debit = -delta

                        transactions.append(
                            {
                                "date": date_iso,
                                "description": clean_text(desc),
                                "debit": round(debit, 2),
                                "credit": round(credit, 2),
                                "balance": round(bal, 2),
                                "page": page_num,
                                "source_file": source_filename,
                                "bank": bank_name,
                            }
                        )

                        prev_balance = bal
                        if latest_tx_date is None or date_iso > latest_tx_date:
                            latest_tx_date = date_iso

                    cur = None
                continue

            if cur is not None:
                bal, first_money_idx = _extract_last_balance_token(ln)
                if bal is not None:
                    toks = ln.split()
                    cur["parts"].append(" ".join(toks[:first_money_idx]) if first_money_idx is not None else ln)

                    date_iso = format_date(cur["date"], detected_year)
                    if date_iso:
                        debit = credit = 0.0
                        if prev_balance is not None:
                            delta = round(bal - prev_balance, 2)
                            if delta > 0:
                                credit = delta
                            elif delta < 0:
                                debit = -delta

                        transactions.append(
                            {
                                "date": date_iso,
                                "description": clean_text(" ".join(cur["parts"])),
                                "debit": round(debit, 2),
                                "credit": round(credit, 2),
                                "balance": round(bal, 2),
                                "page": cur["page"],
                                "source_file": source_filename,
                                "bank": bank_name,
                            }
                        )

                        prev_balance = bal
                        if latest_tx_date is None or date_iso > latest_tx_date:
                            latest_tx_date = date_iso

                    cur = None
                else:
                    cur["parts"].append(ln)

    if closing_balance is None:
        try:
            all_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception:
            all_text = ""
        closing_balance = extract_closing_balance_from_text(all_text)

    if closing_balance is not None:
        cb_date = latest_tx_date or f"{detected_year}-01-01"
        transactions.append(
            {
                "date": cb_date,
                "description": "CLOSING BALANCE / BAKI PENUTUP",
                "debit": 0.0,
                "credit": 0.0,
                "balance": round(float(closing_balance), 2),
                "page": None,
                "source_file": source_filename,
                "bank": bank_name,
                "is_statement_balance": True,
            }
        )

    return transactions


# ---------------------------------------------------------
# MAIN PARSER
# ---------------------------------------------------------

def parse_transactions_cimb(pdf, source_filename: str = ""):
    """
    CIMB parser:
    - First: attempt extract_table()
    - Fallback: text parsing for multi-line table layouts
    - Always attempts statement footer closing balance from FULL document and appends it.
    """
    transactions = []
    detected_year = None
    bank_name = "CIMB Bank"

    # quick bank branding + year
    for page in pdf.pages[:2]:
        text = page.extract_text() or ""
        if "CIMB ISLAMIC BANK" in text.upper():
            bank_name = "CIMB Islamic Bank"
        if not detected_year:
            detected_year = extract_year_from_text(text)
        if detected_year:
            break

    if not detected_year:
        detected_year = str(datetime.now().year)

    # FIX: scan FULL document for closing balance (not just first pages)
    try:
        all_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception:
        all_text = ""
    closing_balance = extract_closing_balance_from_text(all_text)

    latest_tx_date = None

    # primary: table extraction
    for page_num, page in enumerate(pdf.pages, start=1):
        table = page.extract_table()
        if not table:
            continue

        for row in table:
            if not row or len(row) < 6:
                continue

            first_col = str(row[0]).lower() if row[0] else ""
            if "date" in first_col or "tarikh" in first_col:
                continue

            desc_text = str(row[1]).lower() if row[1] else ""
            if "opening balance" in desc_text:
                continue

            if not row[5]:
                continue

            debit_val = parse_float(row[3])   # Withdrawal
            credit_val = parse_float(row[4])  # Deposit

            if debit_val == 0.0 and credit_val == 0.0:
                continue

            date_formatted = format_date(row[0], detected_year)
            if not date_formatted:
                continue

            if latest_tx_date is None or date_formatted > latest_tx_date:
                latest_tx_date = date_formatted

            transactions.append(
                {
                    "date": date_formatted,
                    "description": clean_text(row[1]),
                    "ref_no": clean_text(row[2]),
                    "debit": debit_val,
                    "credit": credit_val,
                    "balance": parse_float(row[5]),
                    "page": page_num,
                    "source_file": source_filename,
                    "bank": bank_name,
                }
            )

    # fallback if table parse produced nothing
    if not transactions:
        return _parse_transactions_cimb_text(
            pdf,
            source_filename=source_filename,
            detected_year=detected_year,
            bank_name=bank_name,
            closing_balance=closing_balance,
        )

    # append statement closing balance row (prevents wrong ending_balance)
    if closing_balance is not None:
        cb_date = latest_tx_date or f"{detected_year}-01-01"
        transactions.append(
            {
                "date": cb_date,
                "description": "CLOSING BALANCE / BAKI PENUTUP",
                "ref_no": "",
                "debit": 0.0,
                "credit": 0.0,
                "balance": float(closing_balance),
                "page": None,
                "source_file": source_filename,
                "bank": bank_name,
                "is_statement_balance": True,
            }
        )

    return transactions
