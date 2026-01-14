import re
from datetime import datetime


# =========================================================
# MAIN ENTRY (USED BY app.py)
# =========================================================

def parse_hong_leong(pdf, filename):
    """
    Robust HLIB parser.

    Key change vs old version:
    - Prefer the statement's Balance column value (if found) instead of purely calculating
      running balance from credit/debit.
    - This prevents false "overdraft" detection when a credit/debit is misclassified.
    """
    transactions = []

    opening_balance = extract_opening_balance(pdf)
    running_balance = opening_balance

    for page_num, page in enumerate(pdf.pages, start=1):
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        if not words:
            continue

        rows = group_words_by_row(words, tolerance=3)

        # Detect Deposit/Withdrawal/Balance column x positions (per page)
        col_x = detect_amount_columns(rows)

        i = 0
        while i < len(rows):
            row = rows[i]

            date = extract_date(row)
            if not date:
                i += 1
                continue

            if is_total_row(row):
                i += 1
                continue

            desc_tokens = []
            amount_tokens = []

            # Current row
            desc_tokens.extend(extract_desc_tokens(row))
            amount_tokens.extend(extract_amount_tokens(row, col_x))

            # Continuation rows until next date
            j = i + 1
            while j < len(rows) and extract_date(rows[j]) is None:
                if is_total_row(rows[j]):
                    break
                desc_tokens.extend(extract_desc_tokens(rows[j]))
                amount_tokens.extend(extract_amount_tokens(rows[j], col_x))
                j += 1

            # Classify amounts into credit/debit based on nearest column x
            credit, debit = classify_amounts_by_columns(amount_tokens, col_x)

            # Extract statement balance from Balance column (preferred)
            stmt_balance = extract_statement_balance(amount_tokens, col_x)

            if credit == 0.0 and debit == 0.0 and stmt_balance is None:
                i = j
                continue

            # If statement balance exists, trust it and sync running_balance to it.
            if stmt_balance is not None:
                running_balance = stmt_balance
                balance_out = stmt_balance
            else:
                # Fallback: compute (less reliable)
                running_balance = round(running_balance + credit - debit, 2)
                balance_out = running_balance

            transactions.append({
                "date": date,
                "description": clean_description(desc_tokens),
                "debit": debit,
                "credit": credit,
                "balance": balance_out,  # ✅ now matches statement when available
                "page": page_num,
                "bank": "Hong Leong Islamic Bank",
                "source_file": filename
            })

            i = j

    return transactions


# =========================================================
# OPENING BALANCE
# =========================================================

def extract_opening_balance(pdf):
    text = pdf.pages[0].extract_text() or ""
    # Allow flexible whitespace/newlines
    m = re.search(
        r"Balance\s+from\s+previous\s+statement\s*([\d,]+\.\d{2})",
        text,
        re.IGNORECASE | re.DOTALL
    )
    if not m:
        raise ValueError("Opening balance not found")
    return float(m.group(1).replace(",", ""))


# =========================================================
# ROW GROUPING (Y AXIS)
# =========================================================

def group_words_by_row(words, tolerance=3):
    rows = []
    for w in words:
        placed = False
        for row in rows:
            if abs(row[0]["top"] - w["top"]) <= tolerance:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])

    for row in rows:
        row.sort(key=lambda x: x["x0"])
    return rows


# =========================================================
# COLUMN DETECTION (Deposit / Withdrawal / Balance)
# =========================================================

def detect_amount_columns(rows):
    """
    Detect column anchors from the header row.

    Supports both English and Malay headers:
    - Deposit / Withdrawal / Balance
    - Simpanan / Pengeluaran / Baki
    """
    deposit_x = withdrawal_x = balance_x = None

    for row in rows:
        joined = " ".join(w["text"].strip().lower() for w in row)

        english_hit = ("deposit" in joined and "withdrawal" in joined and "balance" in joined)
        malay_hit = ("simpanan" in joined and "pengeluaran" in joined and "baki" in joined)

        if english_hit or malay_hit:
            for w in row:
                t = w["text"].strip().lower()

                # English labels
                if t == "deposit":
                    deposit_x = w["x0"]
                elif t == "withdrawal":
                    withdrawal_x = w["x0"]
                elif t == "balance":
                    balance_x = w["x0"]

                # Malay labels
                elif t == "simpanan" and deposit_x is None:
                    deposit_x = w["x0"]
                elif t == "pengeluaran" and withdrawal_x is None:
                    withdrawal_x = w["x0"]
                elif t == "baki" and balance_x is None:
                    balance_x = w["x0"]

            # If we got at least balance_x and one of deposit/withdrawal, good enough
            if balance_x is not None and (deposit_x is not None or withdrawal_x is not None):
                break

    # Sensible fallbacks if header text isn't captured on some pages
    # (These are approximate and will still work because we use "nearest column" logic.)
    if deposit_x is None:
        deposit_x = 320.0
    if withdrawal_x is None:
        withdrawal_x = 410.0
    if balance_x is None:
        balance_x = 520.0

    return {
        "deposit_x": float(deposit_x),
        "withdrawal_x": float(withdrawal_x),
        "balance_x": float(balance_x),
    }


# =========================================================
# DATE DETECTION
# =========================================================

def extract_date(row):
    for w in row:
        if re.fullmatch(r"\d{2}-\d{2}-\d{4}", w["text"]):
            return datetime.strptime(w["text"], "%d-%m-%Y").strftime("%Y-%m-%d")
    return None


# =========================================================
# TOKEN EXTRACTION
# =========================================================

def extract_amount_tokens(row, col_x):
    """
    Only treat money-looking tokens as amounts if they are positioned
    in the right-side amount columns area (near Deposit/Withdrawal/Balance).

    This prevents description numerics like '382.99 PLUS 1500' from being treated as amounts.
    """
    out = []
    # allow small drift to the left of Deposit column
    min_amount_x = col_x["deposit_x"] - 25

    for w in row:
        t = w["text"].strip()
        if re.fullmatch(r"[\d,]+\.\d{2}", t):
            # FILTER BY X: ignore numeric tokens inside description area
            if w["x0"] >= min_amount_x:
                out.append({"x": w["x0"], "value": float(t.replace(",", ""))})

    return out


def extract_desc_tokens(row):
    out = []
    for w in row:
        t = w["text"].strip()
        if not t:
            continue
        if re.fullmatch(r"\d{2}-\d{2}-\d{4}", t):
            continue
        if is_noise(t):
            continue
        out.append(t)
    return out


# =========================================================
# AMOUNT CLASSIFICATION USING COLUMN X (ignore balance column)
# =========================================================

def classify_amounts_by_columns(amount_words, col_x):
    credit = 0.0
    debit = 0.0

    dep = col_x["deposit_x"]
    wdr = col_x["withdrawal_x"]
    bal = col_x["balance_x"]

    for a in amount_words:
        x = a["x"]
        val = a["value"]

        dist_dep = abs(x - dep)
        dist_wdr = abs(x - wdr)
        dist_bal = abs(x - bal)

        if dist_dep <= dist_wdr and dist_dep <= dist_bal:
            credit += val
        elif dist_wdr <= dist_dep and dist_wdr <= dist_bal:
            debit += val
        else:
            # balance column -> ignore here
            pass

    return round(credit, 2), round(debit, 2)


# =========================================================
# STATEMENT BALANCE EXTRACTION (NEW)
# =========================================================

def extract_statement_balance(amount_words, col_x, tol=45):
    """
    Try to extract the balance shown on the statement for this transaction block.

    Strategy:
    - Choose numeric token closest to the Balance column x (within tolerance).
    - If multiple, pick the closest; tie-break by rightmost.
    """
    bal_x = col_x["balance_x"]
    candidates = [a for a in amount_words if abs(a["x"] - bal_x) <= tol]
    if not candidates:
        return None

    candidates.sort(key=lambda a: (abs(a["x"] - bal_x), -a["x"]))
    return round(candidates[0]["value"], 2)


# =========================================================
# FILTERS / CLEANUP
# =========================================================

def is_total_row(row):
    text = " ".join(w["text"] for w in row)
    return bool(re.search(
        r"Total Withdrawals|Total Deposits|Closing Balance|Important Notices",
        text,
        re.IGNORECASE
    ))


def is_noise(text):
    return bool(re.search(
        r"Protected by PIDM|Dilindungi oleh PIDM|Hong Leong Islamic Bank|hlisb\.com\.my|Menara Hong Leong|CURRENT ACCOUNT",
        text,
        re.IGNORECASE
    ))


def clean_description(parts):
    s = " ".join(parts)
    s = re.sub(r"\s+", " ", s).strip()
    return s
