import re
from datetime import datetime
import pdfplumber


def parse_ambank(pdf, filename):
    """
    Parse AmBank statement and extract all transactions.
    """
    transactions = []

    # Extract statement metadata
    statement_info = extract_statement_info(pdf)

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if not text:
            continue

        page_transactions = extract_transactions_from_page(
            text, page_num, filename, statement_info
        )
        transactions.extend(page_transactions)

    # Sort chronologically
    transactions = sort_transactions(transactions)

    # Infer debit / credit using balance delta
    infer_debit_credit_from_balance(
        transactions,
        statement_info.get('opening_balance')
    )

    return transactions


# ---------------------------------------------------------------------
# STATEMENT INFO
# ---------------------------------------------------------------------

def extract_statement_info(pdf):
    """Extract account number, statement period, and opening balance."""
    info = {
        'account_number': None,
        'statement_period': None,
        'currency': 'MYR',
        'opening_balance': None
    }

    try:
        first_page = pdf.pages[0].extract_text()

        # Account number
        acc = re.search(r'ACCOUNT NO\..*?:\s*(\d+)', first_page)
        if acc:
            info['account_number'] = acc.group(1)

        # Statement date
        period = re.search(
            r'STATEMENT DATE.*?:\s*(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})',
            first_page
        )
        if period:
            info['statement_period'] = f"{period.group(1)} to {period.group(2)}"

        # ✅ OPENING BALANCE (verified against your PDF)
        opening = re.search(
            r'OPENING BALANCE\s*/\s*BAKI PEMBUKAAN\s+([\d,]+\.\d{2})',
            first_page,
            re.IGNORECASE
        )
        if opening:
            info['opening_balance'] = float(
                opening.group(1).replace(',', '')
            )

    except Exception as e:
        print(f"Warning extracting statement info: {e}")

    return info


# ---------------------------------------------------------------------
# PAGE PARSING
# ---------------------------------------------------------------------

def extract_transactions_from_page(text, page_num, filename, statement_info):
    transactions = []
    lines = text.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if should_skip_line(line):
            i += 1
            continue

        tx, consumed = parse_transaction_block(
            lines[i:], page_num, filename, statement_info
        )

        if tx:
            transactions.append(tx)
            i += consumed
        else:
            i += 1

    return transactions


def should_skip_line(line):
    skip_patterns = [
        r'^ACCOUNT NO\.',
        r'^STATEMENT DATE',
        r'^CURRENCY',
        r'^PAGE',
        r'^ACCOUNT STATEMENT',
        r'^PENYATA AKAUN',
        r'^Protected by PIDM',
        r'^Dilindungi oleh PIDM',
        r'^OPENING BALANCE',
        r'^TOTAL DEBITS',
        r'^TOTAL CREDITS',
        r'^CLOSING BALANCE',
        r'^DATE$',
        r'^TARIKH$',
        r'^CHEQUE NO',
        r'^TRANSACTION$',
        r'^DEBIT$',
        r'^CREDIT$',
        r'^BALANCE$',
        r'^Baki Bawa Ke Hadapan',
        r'^\s*$'
    ]

    return any(re.match(p, line, re.IGNORECASE) for p in skip_patterns)


# ---------------------------------------------------------------------
# TRANSACTION PARSING
# ---------------------------------------------------------------------

def parse_transaction_block(lines, page_num, filename, statement_info):
    if not lines:
        return None, 0

    first_line = lines[0].strip()

    match = re.match(
        r'^(\d{2})(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(.+)$',
        first_line
    )

    if not match:
        return None, 1

    day, month, rest = match.groups()

    tx = parse_transaction_details(
        day, month, rest, page_num, filename, statement_info
    )

    return (tx, 1) if tx else (None, 1)


def parse_transaction_details(day, month, rest, page_num, filename, statement_info):
    # Year extraction
    year = "2024"
    if statement_info.get('statement_period'):
        y = re.search(r'/(\d{4})', statement_info['statement_period'])
        if y:
            year = y.group(1)

    month_map = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }

    date_str = f"{year}-{month_map[month]}-{day}"

    # -----------------------------------------------------------------
    # ✅ BALANCE PARSING (supports DR/CR suffix at end of line)
    # Example: "11,538.71DR" or "11,538.71 DR"
    # DR => negative balance (overdraft)
    # CR/None => positive balance
    # -----------------------------------------------------------------
    balance = None
    suffix = None

    balance_match = re.search(r'([\d,]+\.\d{2})\s*(DR|CR)?\s*$', rest, re.IGNORECASE)
    if balance_match:
        bal_str = balance_match.group(1)
        suffix = (balance_match.group(2) or "").upper()

        balance = float(bal_str.replace(',', ''))
        if suffix == "DR":
            balance = -balance  # ✅ overdraft

        # remove the trailing balance chunk from rest before extracting amounts/description
        rest_wo_balance = rest[:balance_match.start()].strip()
    else:
        rest_wo_balance = rest.strip()

    # Monetary values in the remaining part (typically debit/credit amounts)
    numbers = re.findall(r'[\d,]+\.\d{2}', rest_wo_balance)

    description = rest_wo_balance
    for n in numbers:
        description = description.replace(n, '')

    cheque = re.search(r'\b(\d{6})\b', description)
    cheque_no = cheque.group(1) if cheque else None

    if cheque_no:
        description = re.sub(r'\b\d{6}\b', '', description)

    description = re.sub(r'\s+', ' ', description)
    description = re.sub(r',+\s*,+', ',', description)
    description = re.sub(r',\s*$', '', description)
    description = description.strip()

    tx = {
        'date': date_str,
        'description': description,
        'debit': 0.0,
        'credit': 0.0,
        'balance': round(balance, 2) if balance is not None else None,
        'page': page_num,
        'bank': 'AmBank Islamic',
        'source_file': filename
    }

    if cheque_no:
        tx['cheque_no'] = cheque_no

    return tx



# ---------------------------------------------------------------------
# POST-PROCESSING
# ---------------------------------------------------------------------

def infer_debit_credit_from_balance(transactions, opening_balance):
    prev_balance = opening_balance

    for tx in transactions:
        curr_balance = tx.get('balance')

        if prev_balance is not None and curr_balance is not None:
            diff = round(curr_balance - prev_balance, 2)

            if diff > 0:
                tx['credit'] = diff
                tx['debit'] = 0.0
            elif diff < 0:
                tx['debit'] = abs(diff)
                tx['credit'] = 0.0

        prev_balance = curr_balance


def sort_transactions(transactions):
    try:
        for tx in transactions:
            tx['_dt'] = datetime.strptime(tx['date'], '%Y-%m-%d')

        transactions.sort(key=lambda x: x['_dt'])

        for tx in transactions:
            del tx['_dt']
    except Exception as e:
        print(f"Warning sorting transactions: {e}")

    return transactions
