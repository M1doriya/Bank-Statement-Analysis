# cimb.py - Standalone CIMB Bank Parser
#
# Strategy:
#   1) Parse using pdfplumber.extract_table()
#   2) Also parse using robust text parsing fallback
#   3) If statement footer totals (Total Withdrawal/Deposits) exist, choose the parse
#      whose summed totals best match the footer totals.
#   4) Always attempt to extract closing balance from the FULL document text.
#
# Notes:
# - Without OCR, scanned/image-only PDFs cannot be parsed reliably.

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
# CLOSING BALANCE EXTRACTION
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


def extract_statement_totals_from_text(text):
    """Extract statement-level Total Withdrawal/Total Deposits if present.

    Common CIMB footer looks like:
        No of Withdrawal No of Deposits Total Withdrawal Total Deposits
        320 51 5,511,545.17 6,609,360.94

    Returns (total_withdrawal, total_deposits) as floats, or (None, None).
    """
    if not text:
        return (None, None)

    up = text.upper()
    if "TOTAL WITHDRAWAL" not in up or "TOTAL DEPOSITS" not in up:
        return (None, None)

    idx = up.rfind("TOTAL WITHDRAWAL")
    window = text[idx: idx + 700] if idx != -1 else text

    # Try to match: count count amount amount
    m = re.search(r"\b\d{1,6}\s+\d{1,6}\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\b", window)
    if m:
        return (parse_float(m.group(1)), parse_float(m.group(2)))

    # Fallback: last two money tokens in window
    money = re.findall(r"[\d,]+\.\d{2}", window)
    if len(money) >= 2:
        return (parse_float(money[-2]), parse_float(money[-1]))

    return (None, None)


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
# FALLBACK TEXT PARSER
# ---------------------------------------------------------

_MONEY_TOKEN_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*\.\d{2}$")


def _extract_amounts_and_balance(line):
    """Return (amount_tokens, balance, first_money_idx)."""
    toks = line.split()
    money_idx = [i for i, t in enumerate(toks) if _MONEY_TOKEN_RE.match(t)]
    if not money_idx:
        return [], None, None

    last_idx = money_idx[-1]
    balance = parse_float(toks[last_idx])
    amount_tokens = [toks[i] for i in money_idx[:-1]]

    first_money_idx = None
    for i, t in enumerate(toks):
        if t == "0" or _MONEY_TOKEN_RE.match(t):
            first_money_idx = i
            break

    return amount_tokens, balance, first_money_idx


def _parse_transactions_cimb_text(pdf, source_filename, detected_year, bank_name="CIMB Bank", closing_balance=None):
    """
    Text-mode parser:
    - Detects transactions by date at line start (DD/MM/YYYY)
    - Captures trailing money tokens and balance
    """
    raw_rows = []
    opening_balance = None
    latest_tx_date = None
    seq = 0
    cur = None  # {"date":..., "parts":[...], "page":...}

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for ln in lines:
            up = ln.upper()

            if up.startswith("OPENING BALANCE"):
                _, bal, _ = _extract_amounts_and_balance(ln)
                if bal is not None:
                    opening_balance = bal
                continue

            if "CLOSING BALANCE" in up and "BAKI" in up:
                continue

            m = re.match(r"^(\d{2}/\d{2}/\d{4})\s+(.*)$", ln)
            if m:
                cur = {"date": m.group(1), "parts": [m.group(2)], "page": page_num}

                amount_tokens, bal, first_money_idx = _extract_amounts_and_balance(ln)
                if bal is not None:
                    toks = ln.split()
                    desc = " ".join(toks[1:first_money_idx]) if first_money_idx is not None else " ".join(toks[1:])
                    date_iso = format_date(cur["date"], detected_year)
                    if date_iso:
                        seq += 1
                        raw_rows.append({
                            "date": date_iso,
                            "description": clean_text(desc),
                            "balance": round(bal, 2),
                            "amount_tokens": amount_tokens,
                            "page": page_num,
                            "_seq": seq,
                        })
                        if latest_tx_date is None or date_iso > latest_tx_date:
                            latest_tx_date = date_iso
                    cur = None
                continue

            if cur is not None:
                amount_tokens, bal, first_money_idx = _extract_amounts_and_balance(ln)
                if bal is not None:
                    toks = ln.split()
                    cur["parts"].append(" ".join(toks[:first_money_idx]) if first_money_idx is not None else ln)

                    date_iso = format_date(cur["date"], detected_year)
                    if date_iso:
                        seq += 1
                        raw_rows.append({
                            "date": date_iso,
                            "description": clean_text(" ".join(cur["parts"])),
                            "balance": round(bal, 2),
                            "amount_tokens": amount_tokens,
                            "page": cur["page"],
                            "_seq": seq,
                        })
                        if latest_tx_date is None or date_iso > latest_tx_date:
                            latest_tx_date = date_iso
                    cur = None
                else:
                    cur["parts"].append(ln)

    # Build transactions; best-effort debit/credit:
    # If intermediate rows are missing, gross totals may still be understated (no OCR can fix that).
    raw_rows = _dedupe_transactions(raw_rows)
    raw_rows_chrono = list(reversed(raw_rows))

    transactions = []
    prev_balance = opening_balance

    for r in raw_rows_chrono:
        bal = parse_float(r.get("balance"))
        debit = credit = 0.0
        amount_tokens = r.get("amount_tokens") or []
        amt = parse_float(amount_tokens[-1]) if amount_tokens else 0.0

        if prev_balance is not None:
            delta = round(bal - prev_balance, 2)
            if delta > 0:
                credit = round(amt if amt > 0 else delta, 2)
            elif delta < 0:
                debit = round(amt if amt > 0 else -delta, 2)

        transactions.append({
            "date": r.get("date"),
            "description": r.get("description"),
            "debit": round(debit, 2),
            "credit": round(credit, 2),
            "balance": round(bal, 2),
            "page": r.get("page"),
            "source_file": source_filename,
            "bank": bank_name,
        })
        prev_balance = bal

    if closing_balance is None:
        all_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        closing_balance = extract_closing_balance_from_text(all_text)

    if closing_balance is not None:
        cb_date = latest_tx_date or (transactions[-1]["date"] if transactions else f"{detected_year}-01-01")
        transactions.append({
            "date": cb_date,
            "description": "CLOSING BALANCE / BAKI PENUTUP",
            "debit": 0.0,
            "credit": 0.0,
            "balance": round(float(closing_balance), 2),
            "page": None,
            "source_file": source_filename,
            "bank": bank_name,
            "is_statement_balance": True
        })

    return _dedupe_transactions(transactions)


# ---------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------

def _dedupe_transactions(transactions):
    """De-duplicate rows that pdfplumber/text parsing can repeat."""
    if not transactions:
        return []

    # Exact de-dupe
    seen_exact = set()
    stage = []
    for t in transactions:
        key = (
            t.get("date"),
            clean_text(t.get("description")),
            clean_text(t.get("ref_no", "")),
            round(parse_float(t.get("debit")), 2),
            round(parse_float(t.get("credit")), 2),
            None if t.get("balance") is None else round(parse_float(t.get("balance")), 2),
        )
        if key in seen_exact:
            continue
        seen_exact.add(key)
        stage.append(t)

    # Soft de-dupe on (date, desc, balance)
    best_by_soft = {}
    order = []
    for t in stage:
        soft_key = (
            t.get("date"),
            clean_text(t.get("description")),
            None if t.get("balance") is None else round(parse_float(t.get("balance")), 2),
        )
        score = round(parse_float(t.get("debit")) + parse_float(t.get("credit")), 2)
        if soft_key not in best_by_soft:
            best_by_soft[soft_key] = (score, t)
            order.append(soft_key)
        else:
            prev_score, _ = best_by_soft[soft_key]
            if score > prev_score:
                best_by_soft[soft_key] = (score, t)

    return [best_by_soft[k][1] for k in order]


def _table_parse_quality_bad(transactions, tolerance=0.05):
    """Heuristic: if too many rows don't reconcile with balance deltas, table extraction is unreliable."""
    rows = [t for t in transactions if t.get("balance") is not None]
    if len(rows) < 15:
        return False

    mismatches = 0
    checks = 0
    prev_balance = None
    for t in rows:
        bal = parse_float(t.get("balance"))
        if prev_balance is None:
            prev_balance = bal
            continue

        debit = parse_float(t.get("debit"))
        credit = parse_float(t.get("credit"))

        if debit == 0.0 and credit == 0.0:
            prev_balance = bal
            continue

        expected = round(prev_balance - debit + credit, 2)
        checks += 1
        if abs(expected - round(bal, 2)) > tolerance:
            mismatches += 1

        prev_balance = bal

    if checks == 0:
        return False

    return (mismatches / checks) > 0.08


# ---------------------------------------------------------
# MAIN PARSER
# ---------------------------------------------------------

def parse_transactions_cimb(pdf, source_filename=""):
    """
    CIMB parser:
    - Parse via table extraction
    - Also parse via robust text mode
    - If footer totals exist, choose the parse whose totals best match
    - Always try to extract closing balance from FULL document text
    """
    transactions = []
    detected_year = None
    closing_balance = None
    bank_name = "CIMB Bank"

    # Detect year / branding quickly
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

    # Full doc scan (critical for closing balance + footer totals)
    full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    stmt_total_debit, stmt_total_credit = extract_statement_totals_from_text(full_text)

    latest_tx_date = None

    # ---- Table extraction ----
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

            date_formatted = format_date(row[0], detected_year)
            if not date_formatted:
                continue

            if latest_tx_date is None or date_formatted > latest_tx_date:
                latest_tx_date = date_formatted

            debit_val = parse_float(row[3])
            credit_val = parse_float(row[4])

            transactions.append({
                "date": date_formatted,
                "description": clean_text(row[1]),
                "ref_no": clean_text(row[2]),
                "debit": debit_val,
                "credit": credit_val,
                "balance": parse_float(row[5]),
                "page": page_num,
                "source_file": source_filename,
                "bank": bank_name,
            })

    transactions = _dedupe_transactions(transactions)

    # ---- Text fallback parse ----
    alt_transactions = _parse_transactions_cimb_text(
        pdf,
        source_filename=source_filename,
        detected_year=detected_year,
        bank_name=bank_name,
        closing_balance=closing_balance,
    )

    def _sum_totals(txs):
        td = tc = 0.0
        for t in txs:
            if t.get("is_statement_balance"):
                continue
            td += parse_float(t.get("debit"))
            tc += parse_float(t.get("credit"))
        return round(td, 2), round(tc, 2)

    # ---- Choose parser output ----
    if transactions and (stmt_total_debit is not None and stmt_total_credit is not None):
        td_a, tc_a = _sum_totals(transactions)
        td_b, tc_b = _sum_totals(alt_transactions)

        diff_a = abs(td_a - stmt_total_debit) + abs(tc_a - stmt_total_credit)
        diff_b = abs(td_b - stmt_total_debit) + abs(tc_b - stmt_total_credit)

        chosen = transactions if diff_a <= diff_b else alt_transactions
    elif transactions:
        chosen = alt_transactions if _table_parse_quality_bad(transactions) else transactions
    else:
        chosen = alt_transactions

    transactions = chosen

    # Closing balance: full doc scan
    if closing_balance is None:
        closing_balance = extract_closing_balance_from_text(full_text)

    # Append closing balance only if not already present
    has_statement_balance = any(t.get("is_statement_balance") for t in transactions)
    if closing_balance is not None and not has_statement_balance:
        cb_date = latest_tx_date or f"{detected_year}-01-01"
        transactions.append({
            "date": cb_date,
            "description": "CLOSING BALANCE / BAKI PENUTUP",
            "ref_no": "",
            "debit": 0.0,
            "credit": 0.0,
            "balance": float(closing_balance),
            "page": None,
            "source_file": source_filename,
            "bank": bank_name,
            "is_statement_balance": True
        })

    return transactions
