import re
from datetime import datetime

# =========================================================
# BANK ISLAM – FORMAT 1 (TABLE-BASED)
# =========================================================
def parse_bank_islam_format1(pdf, source_file):
    transactions = []

    def extract_amount(text):
        if not text:
            return None
        s = re.sub(r"\s+", "", str(text))
        m = re.search(r"(-?[\d,]+\.\d{2})", s)
        return float(m.group(1).replace(",", "")) if m else None

    for page_num, page in enumerate(pdf.pages, start=1):
        table = page.extract_table()
        if not table:
            continue

        for row in table:
            row = list(row) if row else []
            while len(row) < 12:
                row.append(None)

            (
                no, txn_date, customer_eft, txn_code, description,
                ref_no, branch, debit_raw, credit_raw,
                balance_raw, sender_recipient, payment_details
            ) = row[:12]

            if not txn_date or not re.search(r"\d{2}/\d{2}/\d{4}", str(txn_date)):
                continue

            try:
                date = datetime.strptime(
                    re.search(r"\d{2}/\d{2}/\d{4}", str(txn_date)).group(),
                    "%d/%m/%Y"
                ).date().isoformat()
            except Exception:
                continue

            debit = extract_amount(debit_raw) or 0.0
            credit = extract_amount(credit_raw) or 0.0
            balance = extract_amount(balance_raw) or 0.0

            # Recovery from description
            if debit == 0.0 and credit == 0.0:
                recovered = extract_amount(description)
                if recovered:
                    desc = str(description).upper()
                    if "CR" in desc or "CREDIT" in desc or "IN" in desc:
                        credit = recovered
                    else:
                        debit = recovered

            description_clean = " ".join(
                str(x).replace("\n", " ").strip()
                for x in [no, txn_code, description, sender_recipient, payment_details]
                if x and str(x).lower() != "nan"
            )

            transactions.append({
                "date": date,
                "description": description_clean,
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(balance, 2),
                "page": page_num,
                "bank": "Bank Islam",
                "source_file": source_file,
                "format": "format1"
            })

    return transactions


# =========================================================
# BANK ISLAM – FORMAT 2 (TEXT / STATEMENT-BASED)
# 100% BALANCE DELTA DR/CR LOGIC
# =========================================================

MONEY_RE = re.compile(r"\(?-?[\d,]+\.\d{2}\)?")
DATE_AT_START_RE = re.compile(r"^\s*(\d{1,2}/\d{1,2}/\d{2,4})\b")
BAL_BF_RE = re.compile(r"BAL\s+B/F", re.IGNORECASE)

def _to_float(val):
    if not val:
        return None
    neg = val.startswith("(") and val.endswith(")")
    val = val.strip("()").replace(",", "")
    try:
        num = float(val)
        return -num if neg else num
    except ValueError:
        return None

def _parse_date(d):
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(d.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None

def parse_bank_islam_format2(pdf, source_file):
    transactions = []
    opening_balance = None
    prev_balance = None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]

        for line in lines:
            upper = line.upper()

            # 1) Opening balance
            if BAL_BF_RE.search(upper):
                money = MONEY_RE.findall(line)
                if money:
                    opening_balance = _to_float(money[-1])
                    prev_balance = opening_balance
                continue

            # 2) Transaction lines (must start with date)
            m_date = DATE_AT_START_RE.match(line)
            if not m_date or prev_balance is None:
                continue

            date = _parse_date(m_date.group(1))
            if not date:
                continue

            money_raw = MONEY_RE.findall(line)
            money_vals = [_to_float(x) for x in money_raw if _to_float(x) is not None]
            if not money_vals:
                continue

            balance = money_vals[-1]

            # 3) Balance delta logic
            delta = round(balance - prev_balance, 2)
            credit = delta if delta > 0 else 0.0
            debit = abs(delta) if delta < 0 else 0.0
            prev_balance = balance

            # 4) Description
            desc = line[len(m_date.group(1)):].strip()
            for tok in money_raw:
                desc = desc.replace(tok, "").strip()

            transactions.append({
                "date": date,
                "description": desc,
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(balance, 2),
                "page": page_num,
                "bank": "Bank Islam",
                "source_file": source_file,
                "format": "format2_balance_delta"
            })

    return transactions


# =========================================================
# FORMAT 3 – eSTATEMENT (BALANCE DELTA, DIFFERENT LAYOUT)
# =========================================================

def parse_bank_islam_format3(pdf, source_file):
    import re
    from datetime import datetime

    transactions = []
    prev_balance = None

    # Matches "9/10/23" 
    DATE_RE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4})")
    # Matches currency patterns like "47,000.00" 
    MONEY_RE = re.compile(r"(\d{1,3}(?:,\d{3})*\.\d{2})")
    BAL_BF_RE = re.compile(r"BAL\s+B/F", re.IGNORECASE)

    def to_float(x):
        return float(x.replace(",", ""))

    def parse_date(d):
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                return datetime.strptime(d, fmt).date().isoformat()
            except ValueError:
                continue
        return None

    for page_num, page in enumerate(pdf.pages, start=1):
        # Extracting text with tolerance=1 as requested to ensure word separation
        text = page.extract_text(x_tolerance=1) or ""
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]

        for line in lines:
            # 1. Capture Opening Balance 
            if BAL_BF_RE.search(line):
                nums = MONEY_RE.findall(line)
                if nums:
                    prev_balance = to_float(nums[-1])
                continue

            # 2. Extract only the line starting with a Date 
            date_match = DATE_RE.match(line)
            if date_match and prev_balance is not None:
                raw_date = date_match.group(1)
                nums = MONEY_RE.findall(line)
                
                # We expect at least a transaction amount and a balance on this line 
                if len(nums) >= 2:
                    balance = to_float(nums[-1])
                    
                    # Create the first-line-only description
                    # Remove the date and all money values from this specific line 
                    desc = line.replace(raw_date, "").strip()
                    for n in nums:
                        desc = desc.replace(n, "").strip()
                    
                    delta = round(balance - prev_balance, 2)
                    
                    transactions.append({
                        "date": parse_date(raw_date),
                        "description": desc, # Only captures text from this line 
                        "debit": abs(delta) if delta < 0 else 0.0,
                        "credit": delta if delta > 0 else 0.0,
                        "balance": balance,
                        "page": page_num,
                        "bank": "Bank Islam",
                        "source_file": source_file
                    })
                    
                    prev_balance = balance

    return transactions


# =========================================================
# FORMAT 4 – eSTATEMENT (BALANCE DELTA, DIFFERENT LAYOUT)
# =========================================================


def parse_bank_islam_format4(pdf, source_file):
    import re
    from datetime import datetime

    transactions = []
    prev_balance = None

    # Matches "26/02/25" or "28/02/25"
    DATE_RE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4})")
    # Matches currency patterns like "1,000,000.00"
    MONEY_RE = re.compile(r"(\d{1,3}(?:,\d{3})*\.\d{2})")
    BAL_BF_RE = re.compile(r"BAL\s+B/IF", re.IGNORECASE)

    def to_float(x):
        return float(x.replace(",", ""))

    def parse_date(d):
        for fmt in ("%d/%m/%y", "%d/%m/%Y"):
            try:
                return datetime.strptime(d, fmt).date().isoformat()
            except ValueError:
                continue
        return None

    for page_num, page in enumerate(pdf.pages, start=1):
        # Using x_tolerance=1 as established for your normalized text
        text = page.extract_text(x_tolerance=1) or ""
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]

    for line in lines:
        # 1. Capture Opening Balance (Note the extra 'I' in 'B/IF' from your OCR)
        if BAL_BF_RE.search(line):
            nums = MONEY_RE.findall(line)
            if nums:
                prev_balance = to_float(nums[-1])
            continue

        # 2. Extract Date-initiated lines
        date_match = DATE_RE.match(line)
        if date_match and prev_balance is not None:
            raw_date = date_match.group(1)
            nums = MONEY_RE.findall(line)
            
            if nums:
                # The last number is ALWAYS the Balance
                current_balance = to_float(nums[-1])
                
                # The first number (if there are multiple) is the Transaction Amount
                # If only one number exists on the line, it's just the balance (rare but possible)
                tx_amount = to_float(nums[0]) if len(nums) >= 2 else 0.0
                
                # Calculate delta to verify if it's Debit or Credit
                delta = round(current_balance - prev_balance, 2)
                
                # Clean description: Remove date and all detected money values
                desc = line.replace(raw_date, "").strip()
                for n in nums:
                    desc = desc.replace(n, "").strip()

                transactions.append({
                    "date": parse_date(raw_date),
                    "description": desc,
                    "debit": abs(delta) if delta < 0 else 0.0,
                    "credit": delta if delta > 0 else 0.0,
                    "balance": current_balance,
                    "page": page_num,
                    "bank": "Bank Islam",
                    "source_file": source_file,
                    "format": "format4_normalized"
                })
                
                # Update anchor for next row
                prev_balance = current_balance

    return transactions


# =========================================================
# WRAPPER
# =========================================================
def parse_bank_islam(pdf, source_file):
    tx = parse_bank_islam_format1(pdf, source_file)
    if tx:
        return tx

    tx = parse_bank_islam_format2(pdf, source_file)
    if tx:
        return tx

        tx = parse_bank_islam_format4(pdf, source_file)
    if tx:
        return tx

    return parse_bank_islam_format3(pdf, source_file)
