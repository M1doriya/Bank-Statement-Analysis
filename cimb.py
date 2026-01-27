# cimb.py - Standalone CIMB Bank Parser
#
# Primary strategy:
#   1) Try table extraction (existing behavior).
#   2) If table extraction fails / yields no transactions, fall back to robust text parsing.
#
# Why fallback is needed:
# Some CIMB/CIMB Islamic statements render the transaction table with multi-line descriptions
# (e.g., JOMPAY, AUTOPAY DR refs, etc.). pdfplumber.extract_table() often fails on these,
# but the text is still extractable via page.extract_text().
#
# The fallback parser:
# - Detects transactions by date at line start (DD/MM/YYYY)
# - Collects multi-line descriptions until it finds a line containing the ending balance
# - Computes debit/credit using balance delta vs previous balance (robust for negative balances)
#
# Existing behavior is preserved because fallback only triggers when table parsing yields no rows.

import re
from datetime import datetime

# ---------------------------------------------------------
# NOTE ON ACCURACY
# ---------------------------------------------------------
# The two main root-causes of inaccurate monthly totals / ending balance in CIMB PDFs are:
#   1) Closing balance is typically printed on the LAST page, but earlier logic only searched
#      the first 2 pages for the closing balance.
#   2) Some PDFs (especially OCR'd or regenerated statements) contain duplicated transaction rows.
#      If not removed, monthly debit/credit totals (notably July in your dataset) become inflated.
#
# This file fixes both issues while keeping the original fast table-extraction path.

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
# DEDUPLICATION
# ---------------------------------------------------------

def _dedupe_transactions(transactions):
    """Remove duplicated rows produced by OCR / regenerated PDFs.

    We only drop a row if it is an exact duplicate of a previously seen row *including balance*.
    Using balance in the signature avoids removing legitimate repeated payments that share the
    same amount/description but occur at different points in the ledger.
    """
    if not transactions:
        return transactions

    seen = set()
    out = []
    for tx in transactions:
        # Be defensive: tx may come from different code paths.
        date = tx.get("date")
        ref_no = tx.get("ref_no") or ""
        desc = (tx.get("description") or "").strip()
        debit = round(float(tx.get("debit") or 0.0), 2)
        credit = round(float(tx.get("credit") or 0.0), 2)
        bal = round(float(tx.get("balance") or 0.0), 2)
        bank = tx.get("bank") or ""

        # If ref is missing, include description to prevent over-deduping.
        key = (date, ref_no, desc if not ref_no else "", debit, credit, bal, bank)
        if key in seen:
            continue
        seen.add(key)
        out.append(tx)
    return out


def _scan_statement_meta(pdf, max_first_pages=2, max_last_pages=3):
    """Extract (year, closing_balance, bank_name) with a cheap scan.

    Closing balance is frequently on the last page; scanning only the first pages is insufficient.
    To keep performance stable, we scan:
      - the first `max_first_pages`
      - the last `max_last_pages` (if distinct)
    """
    bank_name = "CIMB Bank"
    detected_year = None
    closing_balance = None

    pages = list(pdf.pages)
    if not pages:
        return detected_year, closing_balance, bank_name

    idxs = list(range(min(max_first_pages, len(pages))))
    tail = list(range(max(len(pages) - max_last_pages, 0), len(pages)))
    for i in tail:
        if i not in idxs:
            idxs.append(i)

    for i in idxs:
        text = pages[i].extract_text() or ""
        up = text.upper()
        if "CIMB ISLAMIC BANK" in up:
            bank_name = "CIMB Islamic Bank"
        if not detected_year:
            detected_year = extract_year_from_text(text)
        if closing_balance is None:
            closing_balance = extract_closing_balance_from_text(text)
        if detected_year and closing_balance is not None and bank_name:
            break

    return detected_year, closing_balance, bank_name


# ---------------------------------------------------------
# FALLBACK TEXT PARSER (only used if table parsing fails)
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
        # treat "0" sometimes used for tax column; it is not money token but appears as numeric column
        if t == "0" or _MONEY_TOKEN_RE.match(t):
            first_money_idx = i
            break

    return bal, first_money_idx


def _parse_transactions_cimb_text(pdf, source_filename, detected_year, bank_name="CIMB Bank", closing_balance=None):
    """
    Robust text-mode parser for CIMB Islamic / Current Account-i style statements.
    Uses balance delta to infer debit/credit.
    """
    transactions = []

    # Track opening/previous balance
    prev_balance = None
    latest_tx_date = None

    cur = None  # {"date":..., "parts":[...], "page":...}

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for ln in lines:
            up = ln.upper()

            # detect opening balance
            if up.startswith("OPENING BALANCE"):
                bal, _ = _extract_last_balance_token(ln)
                if bal is not None:
                    prev_balance = bal
                continue

            # ignore closing balance line here (we append it once at end)
            if "CLOSING BALANCE" in up and "BAKI" in up:
                continue

            # Start of a transaction row
            m = re.match(r"^(\d{2}/\d{2}/\d{4})\s+(.*)$", ln)
            if m:
                # If we had a dangling tx without a balance line, drop it safely
                cur = {"date": m.group(1), "parts": [m.group(2)], "page": page_num}

                # sometimes the same line already includes balance
                bal, first_money_idx = _extract_last_balance_token(ln)
                if bal is not None:
                    toks = ln.split()
                    # strip date token + numeric columns from desc
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
                        else:
                            # fallback classification if opening balance missing
                            desc_up = desc.upper()
                            if " CR" in f" {desc_up} " or "CREDIT" in desc_up:
                                credit = 0.0
                            else:
                                debit = 0.0

                        transactions.append({
                            "date": date_iso,
                            "description": clean_text(desc),
                            "debit": round(debit, 2),
                            "credit": round(credit, 2),
                            "balance": round(bal, 2),
                            "page": page_num,
                            "source_file": source_filename,
                            "bank": bank_name
                        })

                        prev_balance = bal
                        if latest_tx_date is None or date_iso > latest_tx_date:
                            latest_tx_date = date_iso

                    cur = None

                continue

            # Continuation lines (multi-line description) until we find a balance line
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
                            "source_file": source_filename,
                            "bank": bank_name
                        })

                        prev_balance = bal
                        if latest_tx_date is None or date_iso > latest_tx_date:
                            latest_tx_date = date_iso

                    cur = None
                else:
                    cur["parts"].append(ln)

    # If caller didn't provide closing balance, try to find it from whole document text
    if closing_balance is None:
        all_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        closing_balance = extract_closing_balance_from_text(all_text)

    if closing_balance is not None:
        cb_date = latest_tx_date or f"{detected_year}-01-01"
        transactions.append({
            "date": cb_date,
            "description": "CLOSING BALANCE / BAKI PENUTUP",
            "debit": 0.0,
            "credit": 0.0,
            "balance": round(float(closing_balance), 2),
            "page": None,
            "source_file": source_filename,
            "bank": bank_name,
            "is_statement_balance": True,
            "__row_order": 10**12
        })

    # final de-duplication (OCR/regenerated PDFs can duplicate whole rows)
    return _dedupe_transactions(transactions)


# ---------------------------------------------------------
# MAIN PARSER
# ---------------------------------------------------------

def parse_transactions_cimb(pdf, source_filename=""):
    """
    CIMB parser:
    - First: attempt extract_table() (existing behavior).
    - Fallback: text parsing for multi-line table layouts (CIMB Islamic / Current Account-i, etc.)
    """
    transactions = []

    # ---- Pass 1: detect year + closing balance + bank branding ----
    detected_year, closing_balance, bank_name = _scan_statement_meta(pdf)

    if not detected_year:
        detected_year = str(datetime.now().year)

    latest_tx_date = None  # YYYY-MM-DD string

    # ---- Pass 2: primary parse via table extraction (existing behavior) ----
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
                "bank": bank_name
            }
            transactions.append(tx)

    # ---- De-duplicate table-extracted rows (fixes inflated totals on some PDFs) ----
    transactions = _dedupe_transactions(transactions)

    # ---- If table extraction found nothing, use fallback text parser ----
    if not transactions:
        return _parse_transactions_cimb_text(
            pdf,
            source_filename=source_filename,
            detected_year=detected_year,
            bank_name=bank_name,
            closing_balance=closing_balance
        )

    # ---- Append closing balance row with a REAL DATE so app.py won't drop it ----
    if closing_balance is not None:
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
            "is_statement_balance": True,
            # give downstream sort a stable hint to keep this last on the date
            "__row_order": 10**12
        })

    return _dedupe_transactions(transactions)
