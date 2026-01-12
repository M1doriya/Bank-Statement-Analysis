# cimb.py - Standalone CIMB Bank Parser
# Fix: Closing balance row now gets a real date (latest transaction date),
# so Streamlit monthly summary won't drop it.

import re
from datetime import datetime

# ---------------------------------------------------------
# YEAR EXTRACTION
# ---------------------------------------------------------

def extract_year_from_text(text):
    """
    Extract year from CIMB statement.
    Handles 4-digit and 2-digit year formats.
    """
    if not text:
        return None

    match = re.search(
        r'(?:STATEMENT DATE|TARIKH PENYATA)\s*[:\s]+\d{1,2}/\d{1,2}/(\d{2,4})',
        text,
        re.IGNORECASE
    )
    if match:
        y = match.group(1)
        return y if len(y) == 4 else str(2000 + int(y))

    return None


# ---------------------------------------------------------
# CLOSING BALANCE EXTRACTION (layout regex)
# ---------------------------------------------------------

def extract_closing_balance_from_text(text):
    """
    Extract:
    CLOSING BALANCE / BAKI PENUTUP 51.79
    """
    if not text:
        return None

    match = re.search(
        r'CLOSING\s+BALANCE\s*/\s*BAKI\s+PENUTUP\s+([\d,]+\.\d{2})',
        text,
        re.IGNORECASE
    )
    if match:
        return float(match.group(1).replace(",", ""))

    return None


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def parse_float(value):
    """Converts string '1,234.56' to float 1234.56. Returns 0.0 if invalid."""
    if not value:
        return 0.0
    clean = str(value).replace("\n", "").replace(" ", "").replace(",", "")
    if not re.match(r'^-?\d+(\.\d+)?$', clean):
        return 0.0
    return float(clean)


def clean_text(text):
    """Removes excess newlines from descriptions."""
    if not text:
        return ""
    return text.replace("\n", " ").strip()


def format_date(date_str, year):
    """
    Format date string to YYYY-MM-DD.
    Handles DD/MM/YYYY and DD/MM.
    """
    if not date_str:
        return None

    date_str = clean_text(date_str)

    # DD/MM/YYYY
    m = re.match(r'(\d{2})/(\d{2})/(\d{4})$', date_str)
    if m:
        dd, mm, yyyy = m.groups()
        return f"{yyyy}-{mm}-{dd}"

    # DD/MM
    m = re.match(r'(\d{2})/(\d{2})$', date_str)
    if m:
        dd, mm = m.groups()
        return f"{year}-{mm}-{dd}"

    # Already YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    return None


# ---------------------------------------------------------
# MAIN PARSER
# ---------------------------------------------------------

def parse_transactions_cimb(pdf, source_filename=""):
    """
    CIMB parser:
    - Transactions from table
    - Closing balance from layout regex
    - Closing balance row gets date = latest transaction date (so app.py keeps it)
    """
    transactions = []
    detected_year = None
    closing_balance = None

    # ---- Pass 1: detect year + closing balance from layout text ----
    for page in pdf.pages:
        text = page.extract_text() or ""

        if not detected_year:
            detected_year = extract_year_from_text(text)

        if closing_balance is None:
            closing_balance = extract_closing_balance_from_text(text)

        if detected_year and closing_balance is not None:
            break

    if not detected_year:
        detected_year = str(datetime.now().year)

    latest_tx_date = None  # YYYY-MM-DD string

    # ---- Pass 2: parse transaction table ----
    for page_num, page in enumerate(pdf.pages, start=1):
        table = page.extract_table()
        if not table:
            continue

        for row in table:
            # CIMB Structure: [Date, Desc, Ref, Withdrawal, Deposit, Balance]
            if not row or len(row) < 6:
                continue

            # Skip headers
            first_col = str(row[0]).lower() if row[0] else ""
            if "date" in first_col or "tarikh" in first_col:
                continue

            desc_text = str(row[1]).lower() if row[1] else ""
            if "opening balance" in desc_text:
                # (Optional) keep opening balance row if you want
                continue

            if not row[5]:
                continue

            debit_val = parse_float(row[3])   # Withdrawal
            credit_val = parse_float(row[4])  # Deposit

            # skip spill rows without amounts
            if debit_val == 0.0 and credit_val == 0.0:
                continue

            date_formatted = format_date(row[0], detected_year)
            if not date_formatted:
                continue

            # track latest transaction date
            if latest_tx_date is None or date_formatted > latest_tx_date:
                latest_tx_date = date_formatted

            tx = {
                "date": date_formatted,
                "description": clean_text(row[1]),
                "ref_no": clean_text(row[2]),
                "debit": debit_val,
                "credit": credit_val,
                "balance": parse_float(row[5]),
                "page": page_num,
                "source_file": source_filename,
                "bank": "CIMB Bank"
            }
            transactions.append(tx)

    # ---- Append closing balance row with a REAL DATE so app.py won't drop it ----
    if closing_balance is not None:
        # Use latest transaction date so it falls into the correct month in monthly summary
        cb_date = latest_tx_date or f"{detected_year}-01-01"

        transactions.append({
            "date": cb_date,
            "description": "CLOSING BALANCE / BAKI PENUTUP",
            "ref_no": "",
            "debit": 0.0,
            "credit": 0.0,
            "balance": closing_balance,
            "page": None,
            "source_file": source_filename,
            "bank": "CIMB Bank",
            "is_statement_balance": True
        })

    return transactions
