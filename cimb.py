# cimb.py - Standalone CIMB Bank Parser (FIXED)
#
# Fixes implemented:
# 1) Ending balance accuracy:
#    - Closing balance marker row is forced to sort LAST in app.py monthly summary
#      by setting page to a very large number and __row_order to a very large number.
#
# 2) July (and similar months) total debit/credit accuracy:
#    - Table extraction may return partial rows (especially on multi-line CIMB layouts),
#      causing missing transactions and wrong totals.
#    - We now extract statement totals (Total Withdrawals / Total Deposits) from the
#      statement footer and validate computed totals.
#    - If table result mismatches statement totals beyond tolerance, we fall back to
#      robust text parsing (balance-delta driven).
#
# Existing performance is preserved:
# - We still try table extraction first.
# - We only run fallback when table extraction is empty OR fails validation.

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
    CLOSING BALANCE / BAKI PENUTUP -2,971,129.25
    Supports negative balances.
    """
    if not text:
        return None

    match = re.search(
        r'CLOSING\s+BALANCE\s*/\s*BAKI\s+PENUTUP\s+(-?[\d,]+\.\d{2})',
        text,
        re.IGNORECASE
    )
    if match:
        return float(match.group(1).replace(",", ""))

    return None


# ---------------------------------------------------------
# STATEMENT TOTALS EXTRACTION (footer)
# ---------------------------------------------------------

def extract_statement_totals_from_text(text):
    """
    Extract statement footer totals:
      - TOTAL WITHDRAWALS / JUMLAH PENGELUARAN
      - TOTAL DEPOSITS / JUMLAH DEPOSIT

    Returns (total_withdrawals, total_deposits) as floats or (None, None).
    """
    if not text:
        return (None, None)

    # Normalize whitespace to improve regex stability
    norm = re.sub(r"\s+", " ", text.upper())

    # Common CIMB footer wording:
    # "TOTAL WITHDRAWALS Jumlah Pengeluaran ... <amount>"
    # "TOTAL DEPOSITS Jumlah Deposit ... <amount>"
    #
    # Sometimes both numbers appear on the same footer row, but usually labels exist.
    money = r"(-?[\d,]+\.\d{2})"

    td = None
    tc = None

    m = re.search(r"TOTAL\s+WITHDRAWALS.*?" + money, norm)
    if m:
        td = float(m.group(1).replace(",", ""))

    m = re.search(r"TOTAL\s+DEPOSITS.*?" + money, norm)
    if m:
        tc = float(m.group(1).replace(",", ""))

    return (td, tc)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def parse_float(value):
    """Converts string '1,234.56' or '-1,234.56' to float. Returns 0.0 if invalid."""
    if value is None:
        return 0.0
    clean = str(value).replace("\n", "").replace(" ", "").replace(",", "")
    if clean == "":
        return 0.0
    if not re.match(r'^-?\d+(\.\d+)?$', clean):
        return 0.0
    try:
        return float(clean)
    except Exception:
        return 0.0


def clean_text(text):
    """Removes excess newlines from descriptions."""
    if not text:
        return ""
    return str(text).replace("\n", " ").strip()


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
# FALLBACK TEXT PARSER (balance-delta based)
# ---------------------------------------------------------

_MONEY_TOKEN_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*\.\d{2}$")

def _extract_last_balance_token(line):
    """
    Finds the last money token on the line (e.g., balance), returns (balance_float, first_money_index).
    Also returns first_money_index so we can strip trailing numeric columns from description.
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


def _parse_transactions_cimb_text(pdf, source_filename, detected_year, bank_name="CIMB Bank", closing_balance=None):
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

            # Detect opening balance more flexibly (not only startswith)
            if "OPENING BALANCE" in up:
                bal, _ = _extract_last_balance_token(ln)
                if bal is not None:
                    prev_balance = bal
                continue

            # Ignore closing balance line here (we append once at end)
            if "CLOSING BALANCE" in up and "BAKI" in up:
                continue

            # Start of a transaction row (CIMB usually prints full date DD/MM/YYYY)
            m = re.match(r"^(\d{2}/\d{2}/\d{4})\s+(.*)$", ln)
            if m:
                cur = {"date": m.group(1), "parts": [m.group(2)], "page": page_num}

                # Sometimes the same line already includes balance
                bal, first_money_idx = _extract_last_balance_token(ln)
                if bal is not None:
                    toks = ln.split()
                    if first_money_idx is not None:
                        desc = " ".join(toks[1:first_money_idx])
                    else:
                        desc = " ".join(toks[1:])

                    date_iso = format_date(cur["date"], detected_year)
                    if date_iso:
                        debit = credit = 0.0
                        if prev_balance is not None:
                            delta = round(bal - prev_balance, 2)
                            if delta > 0:
                                credit = delta
                            elif delta < 0:
                                debit = -delta

                        transactions.append({
                            "date": date_iso,
                            "description": clean_text(desc),
                            "debit": round(debit, 2),
                            "credit": round(credit, 2),
                            "balance": round(bal, 2),
                            "page": page_num,
                            "__row_order": len(transactions),
                            "source_file": source_filename,
                            "bank": bank_name
                        })

                        prev_balance = bal
                        if latest_tx_date is None or date_iso > latest_tx_date:
                            latest_tx_date = date_iso

                    cur = None
                continue

            # Continuation lines for multi-line description until we find a balance token
            if cur is not None:
                bal, first_money_idx = _extract_last_balance_token(ln)
                if bal is not None:
                    toks = ln.split()
                    if first_money_idx is not None:
                        cur["parts"].append(" ".join(toks[:first_money_idx]))
                    else:
                        cur["parts"].append(ln)

                    date_iso = format_date(cur["date"], detected_year)
                    if date_iso:
                        debit = credit = 0.0
                        if prev_balance is not None:
                            delta = round(bal - prev_balance, 2)
                            if delta > 0:
                                credit = delta
                            elif delta < 0:
                                debit = -delta

                        transactions.append({
                            "date": date_iso,
                            "description": clean_text(" ".join(cur["parts"])),
                            "debit": round(debit, 2),
                            "credit": round(credit, 2),
                            "balance": round(bal, 2),
                            "page": cur["page"],
                            "__row_order": len(transactions),
                            "source_file": source_filename,
                            "bank": bank_name
                        })

                        prev_balance = bal
                        if latest_tx_date is None or date_iso > latest_tx_date:
                            latest_tx_date = date_iso

                    cur = None
                else:
                    cur["parts"].append(ln)

    # If caller didn't provide closing balance, try full text
    if closing_balance is None:
        all_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        closing_balance = extract_closing_balance_from_text(all_text)

    # Append closing balance marker that sorts LAST in app.py
    if closing_balance is not None:
        cb_date = latest_tx_date or f"{detected_year}-01-01"
        BIG = 10**9
        transactions.append({
            "date": cb_date,
            "description": "CLOSING BALANCE / BAKI PENUTUP",
            "debit": 0.0,
            "credit": 0.0,
            "balance": round(float(closing_balance), 2),
            "page": BIG,                # ensure last after sorting by (date, page, __row_order)
            "__row_order": BIG,
            "source_file": source_filename,
            "bank": bank_name,
            "is_statement_balance": True
        })

    return transactions


# ---------------------------------------------------------
# MAIN PARSER
# ---------------------------------------------------------

def parse_transactions_cimb(pdf, source_filename=""):
    """
    CIMB parser:
    - First: attempt extract_table() (fast, but sometimes partial).
    - Validate against statement totals (footer).
    - If mismatch (or no rows), fall back to robust text parsing (balance-delta).
    """
    transactions = []
    detected_year = None
    closing_balance = None
    bank_name = "CIMB Bank"

    # ---- Pass 1: detect bank + year + closing balance ----
    for page in pdf.pages[:2]:
        text = page.extract_text() or ""

        if "CIMB ISLAMIC BANK" in text.upper():
            bank_name = "CIMB Islamic Bank"

        if not detected_year:
            detected_year = extract_year_from_text(text)

        if closing_balance is None:
            closing_balance = extract_closing_balance_from_text(text)

        if detected_year and closing_balance is not None:
            break

    if not detected_year:
        detected_year = str(datetime.now().year)

    latest_tx_date = None  # YYYY-MM-DD

    # ---- Pass 2: primary parse via table extraction ----
    for page_num, page in enumerate(pdf.pages, start=1):
        table = page.extract_table()
        if not table:
            continue

        for row in table:
            # CIMB Structure: [Date, Desc, Ref, Withdrawal, Deposit, Balance]
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

            # skip spill rows without amounts
            if debit_val == 0.0 and credit_val == 0.0:
                continue

            date_formatted = format_date(row[0], detected_year)
            if not date_formatted:
                continue

            if latest_tx_date is None or date_formatted > latest_tx_date:
                latest_tx_date = date_formatted

            tx = {
                "date": date_formatted,
                "description": clean_text(row[1]),
                "ref_no": clean_text(row[2]),
                "debit": round(debit_val, 2),
                "credit": round(credit_val, 2),
                "balance": round(parse_float(row[5]), 2),
                "page": page_num,
                "__row_order": len(transactions),
                "source_file": source_filename,
                "bank": bank_name
            }
            transactions.append(tx)

    # ---- Extract statement totals from full document (used for validation) ----
    all_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    stmt_total_debit, stmt_total_credit = extract_statement_totals_from_text(all_text)

    def _totals_diff(tx_list):
        td = round(sum(t.get("debit", 0.0) or 0.0 for t in tx_list), 2)
        tc = round(sum(t.get("credit", 0.0) or 0.0 for t in tx_list), 2)
        diff = 0.0
        parts = 0
        if stmt_total_debit is not None:
            diff += abs(td - float(stmt_total_debit))
            parts += 1
        if stmt_total_credit is not None:
            diff += abs(tc - float(stmt_total_credit))
            parts += 1
        return diff, td, tc, parts

    # ---- Decide whether to trust table extraction ----
    use_fallback = False

    if not transactions:
        use_fallback = True
    else:
        # Validate only if statement totals exist (otherwise keep existing behavior)
        diff, td, tc, parts = _totals_diff(transactions)
        if parts > 0:
            # tolerance: a few cents
            if diff > 0.05:
                use_fallback = True

        # Also validate closing balance if we have it:
        # if the last balance doesn't equal statement closing balance, likely missing rows.
        if (not use_fallback) and (closing_balance is not None):
            last_bal = None
            # pick max date then max page then max __row_order
            try:
                last_tx = sorted(
                    transactions,
                    key=lambda x: (x.get("date") or "", int(x.get("page") or 0), int(x.get("__row_order") or 0))
                )[-1]
                last_bal = float(last_tx.get("balance")) if last_tx.get("balance") is not None else None
            except Exception:
                last_bal = None

            if last_bal is not None and abs(last_bal - float(closing_balance)) > 0.01:
                use_fallback = True

    if use_fallback:
        return _parse_transactions_cimb_text(
            pdf,
            source_filename=source_filename,
            detected_year=detected_year,
            bank_name=bank_name,
            closing_balance=closing_balance
        )

    # ---- Append closing balance marker that sorts LAST in app.py ----
    if closing_balance is None:
        closing_balance = extract_closing_balance_from_text(all_text)

    if closing_balance is not None:
        cb_date = latest_tx_date or f"{detected_year}-01-01"
        BIG = 10**9
        transactions.append({
            "date": cb_date,
            "description": "CLOSING BALANCE / BAKI PENUTUP",
            "ref_no": "",
            "debit": 0.0,
            "credit": 0.0,
            "balance": round(float(closing_balance), 2),
            "page": BIG,
            "__row_order": BIG,
            "source_file": source_filename,
            "bank": bank_name,
            "is_statement_balance": True
        })

    return transactions
