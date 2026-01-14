import re
from datetime import datetime


# =========================================================
# Regex
# =========================================================

# Dates: 26-09-2025
DATE_TOKEN_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")

# Money tokens may appear as:
#   1,234.56
#   1,234.56-
#   1,234.56+
MONEY_TOKEN_RE = re.compile(r"^(?P<num>[\d,]+\.\d{2})(?P<sign>[+-])?$")

# Optional: detect explicit OD mention in PDF text.
OD_KEYWORDS_RE = re.compile(
    r"\b(overdraft|od\s+facility|od\s+limit|overdrawn|excess\s+limit|interest\s+on\s+overdraft)\b",
    re.I
)


# =========================================================
# MAIN ENTRY (USED BY app.py)
# =========================================================

def parse_hong_leong(pdf, filename):
    """
    Minimal-change fix:
      - Keep existing debit/credit extraction (to avoid deviating from current totals/rows).
      - ALSO extract statement Balance column token when available and anchor running_balance to it.
      - Prevent false negative lowest-balance when Balance extraction is missing:
          * If no OD keywords in statement and computed running balance goes negative,
            carry forward last known statement balance (or last running_balance) instead of going negative.
    """
    transactions = []

    opening_balance = extract_opening_balance(pdf)
    running_balance = opening_balance

    # Track last statement-anchored balance (more reliable than computed)
    last_stmt_balance = None

    # If the PDF explicitly mentions OD, allow negative balances to surface
    overdraft_possible = pdf_mentions_overdraft(pdf)

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

            # Existing behavior: classify credit/debit from amount tokens
            credit, debit = classify_amounts_by_columns(amount_tokens, col_x)

            # NEW: extract statement balance from balance column token if present
            stmt_balance = extract_statement_balance(amount_tokens, col_x)

            # If nothing found, skip
            if credit == 0.0 and debit == 0.0 and stmt_balance is None:
                i = j
                continue

            # Compute balance as before
            computed_balance = round(running_balance + credit - debit, 2)

            # Anchor to statement balance if available (prevents drift and false OD)
            if stmt_balance is not None:
                running_balance = float(stmt_balance)
                last_stmt_balance = float(stmt_balance)
            else:
                # No statement balance captured on this row:
                # If computed would go negative but statement has no OD, do NOT surface fake negative.
                if (not overdraft_possible) and computed_balance < 0:
                    # Prefer last known statement balance if we have it; otherwise keep previous running_balance.
                    if last_stmt_balance is not None:
                        running_balance = float(last_stmt_balance)
                    else:
                        # keep running_balance unchanged
                        running_balance = float(running_balance)
                else:
                    running_balance = float(computed_balance)

            transactions.append({
                "date": date,
                "description": clean_description(desc_tokens),
                "debit": debit,
                "credit": credit,
                "balance": round(float(running_balance), 2),  # now anchored when possible
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
    m = re.search(
        r"Balance from previous statement\s+([\d,]+\.\d{2})",
        text,
        re.IGNORECASE
    )
    if not m:
        raise ValueError("Opening balance not found")
    return float(m.group(1).replace(",", ""))


def pdf_mentions_overdraft(pdf) -> bool:
    for p in pdf.pages:
        t = (p.extract_text() or "")
        if OD_KEYWORDS_RE.search(t):
            return True
    return False


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
    deposit_x = withdrawal_x = balance_x = None

    for row in rows:
        joined = " ".join(w["text"].strip().lower() for w in row)

        # header typically contains all three labels
        if "deposit" in joined and "withdrawal" in joined and "balance" in joined:
            for w in row:
                t = w["text"].strip().lower()
                if t == "deposit":
                    deposit_x = w["x0"]
                elif t == "withdrawal":
                    withdrawal_x = w["x0"]
                elif t == "balance":
                    balance_x = w["x0"]
            break

    # sensible fallbacks if header text isn't captured on some pages
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
        if DATE_TOKEN_RE.fullmatch(w["text"]):
            return datetime.strptime(w["text"], "%d-%m-%Y").strftime("%Y-%m-%d")
    return None


# =========================================================
# TOKEN EXTRACTION
# =========================================================

def parse_money_token(s: str) -> float:
    """
    Parse '1,234.56', '1,234.56-' or '1,234.56+'.
    Returns signed float.
    """
    m = MONEY_TOKEN_RE.match(s)
    if not m:
        raise ValueError("Not a money token")
    num = float(m.group("num").replace(",", ""))
    sign = m.group("sign")
    if sign == "-":
        return -num
    return num


def extract_amount_tokens(row, col_x):
    """
    Only treat money-looking tokens as amounts if they are positioned
    in the right-side amount columns area (near Deposit/Withdrawal/Balance).
    """
    out = []
    min_amount_x = col_x["deposit_x"] - 25

    for w in row:
        t = w["text"].strip()
        if MONEY_TOKEN_RE.match(t):
            if w["x0"] >= min_amount_x:
                val = parse_money_token(t)
                out.append({"x": w["x0"], "value": val})

    return out


def extract_desc_tokens(row):
    out = []
    for w in row:
        t = w["text"].strip()
        if not t:
            continue
        if DATE_TOKEN_RE.fullmatch(t):
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
        # deposit/withdrawal amounts should be positive magnitudes
        val = abs(a["value"])

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
# NEW: STATEMENT BALANCE EXTRACTION
# =========================================================

def extract_statement_balance(amount_words, col_x, tol=70.0):
    """
    Extract the statement printed Balance token from amount_words using x proximity.
    Returns float (can be negative if token has '-' sign), or None if not found.
    """
    bal_x = col_x["balance_x"]
    dep_x = col_x["deposit_x"]
    wdr_x = col_x["withdrawal_x"]

    candidates = []
    for a in amount_words:
        x = float(a["x"])
        v = float(a["value"])

        # Must be plausibly in balance column region
        if abs(x - bal_x) <= tol or x >= (bal_x - 10):
            # Also require it is closer to balance than deposit/withdrawal
            if abs(x - bal_x) <= abs(x - dep_x) and abs(x - bal_x) <= abs(x - wdr_x):
                candidates.append((abs(x - bal_x), -x, v))

    if not candidates:
        return None

    # Closest to balance; tie-breaker: rightmost
    candidates.sort()
    return round(float(candidates[0][2]), 2)


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
