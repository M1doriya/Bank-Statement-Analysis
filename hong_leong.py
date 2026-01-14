import re
from datetime import datetime


# =========================================================
# MAIN ENTRY (USED BY app.py)
# =========================================================

def parse_hong_leong(pdf, filename):
    """
    Hong Leong Islamic Bank (Current Account-i) parser.

    FIX (IMPORTANT):
    - Uses the statement's BALANCE column as source-of-truth whenever possible.
    - Infers debit/credit from balance delta (current_balance - prev_balance).
    - Avoids misclassifying the BALANCE column number as a debit/credit amount,
      which previously caused false negative balances / false OD detection.

    Output schema:
      {
        "date": "YYYY-MM-DD",
        "description": str,
        "debit": float,
        "credit": float,
        "balance": float,   # statement balance when available, else calculated
        "page": int,
        "bank": "Hong Leong Islamic Bank",
        "source_file": filename
      }
    """
    transactions = []

    opening_balance = extract_opening_balance(pdf)
    prev_balance = opening_balance

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
            amt_tokens = []       # deposit/withdrawal candidates (excluding balance)
            bal_tokens = []       # balance candidates

            # Current row
            desc_tokens.extend(extract_desc_tokens(row, col_x))
            a, b = extract_amount_and_balance_tokens(row, col_x)
            amt_tokens.extend(a)
            bal_tokens.extend(b)

            # Continuation rows until next date
            j = i + 1
            while j < len(rows) and extract_date(rows[j]) is None:
                if is_total_row(rows[j]):
                    break
                desc_tokens.extend(extract_desc_tokens(rows[j], col_x))
                a2, b2 = extract_amount_and_balance_tokens(rows[j], col_x)
                amt_tokens.extend(a2)
                bal_tokens.extend(b2)
                j += 1

            # Prefer statement balance token (balance-delta method)
            current_balance = pick_best_balance(bal_tokens)

            if current_balance is not None:
                delta = round(current_balance - prev_balance, 2)
                credit = delta if delta > 0 else 0.0
                debit = abs(delta) if delta < 0 else 0.0

                # If delta is 0, usually it's not a real transaction row; skip quietly.
                if credit == 0.0 and debit == 0.0:
                    i = j
                    prev_balance = current_balance
                    continue

                tx_balance = current_balance
                prev_balance = current_balance
            else:
                # Fallback: classify tokens by column (only deposit/withdrawal area)
                credit, debit = classify_amounts_by_columns(amt_tokens, col_x)
                if credit == 0.0 and debit == 0.0:
                    i = j
                    continue
                tx_balance = round(prev_balance + credit - debit, 2)
                prev_balance = tx_balance

            transactions.append({
                "date": date,
                "description": clean_description(desc_tokens),
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(tx_balance, 2),
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
        if re.fullmatch(r"\d{2}-\d{2}-\d{4}", w["text"]):
            return datetime.strptime(w["text"], "%d-%m-%Y").strftime("%Y-%m-%d")
    return None


# =========================================================
# ROW SKIP (TOTALS / SUMMARY)
# =========================================================

def is_total_row(row):
    joined = " ".join((w.get("text") or "").strip().lower() for w in row)
    # End-of-table summary lines
    keys = (
        "total withdrawals",
        "jumlah pengeluaran",
        "total deposits",
        "jumlah simpanan",
        "closing balance",
        "baki akhir",
        "important notices",
        "notis penting",
    )
    return any(k in joined for k in keys)


# =========================================================
# TOKEN EXTRACTION (DESCRIPTION, AMOUNTS, BALANCE)
# =========================================================

_MONEY_TOKEN_RE = re.compile(r"^\d{1,3}(?:,\d{3})*\.\d{2}$")

def extract_desc_tokens(row, col_x):
    """
    Keep only left-side tokens (description area), excluding the date token.
    """
    out = []
    cutoff = col_x["deposit_x"] - 10  # anything left of deposit column is considered description
    for w in row:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        if re.fullmatch(r"\d{2}-\d{2}-\d{4}", t):
            continue
        if w["x0"] < cutoff:
            out.append(t)
    return out


def extract_amount_and_balance_tokens(row, col_x):
    """
    Split money-looking tokens into:
      - amt_tokens: deposit/withdrawal candidates (left of balance column)
      - bal_tokens: balance candidates (in/near the balance column)
    """
    amt_tokens = []
    bal_tokens = []

    # allow small drift
    min_amount_x = col_x["deposit_x"] - 25
    bal_cut = col_x["balance_x"] - 20  # balance column area starts here

    for w in row:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        if not _MONEY_TOKEN_RE.match(t):
            continue
        x = float(w["x0"])

        # ignore money-looking tokens in the description area
        if x < min_amount_x:
            continue

        if x >= bal_cut:
            bal_tokens.append({"text": t, "x0": x})
        else:
            amt_tokens.append({"text": t, "x0": x})

    return amt_tokens, bal_tokens


def pick_best_balance(bal_tokens):
    """
    Choose the most likely BALANCE number from a transaction block.
    Usually the right-most (largest x0) money token in the balance area.
    """
    if not bal_tokens:
        return None
    best = max(bal_tokens, key=lambda z: z["x0"])
    try:
        return float(best["text"].replace(",", ""))
    except Exception:
        return None


# =========================================================
# AMOUNT CLASSIFICATION (FALLBACK ONLY)
# =========================================================

def classify_amounts_by_columns(amount_tokens, col_x):
    """
    Fallback classification when balance token is missing:
    - decide credit vs debit by proximity to deposit_x vs withdrawal_x
    """
    if not amount_tokens:
        return 0.0, 0.0

    deposit_x = col_x["deposit_x"]
    withdrawal_x = col_x["withdrawal_x"]

    credit = 0.0
    debit = 0.0

    for tok in amount_tokens:
        val = safe_float(tok["text"])
        if val is None:
            continue
        x0 = tok["x0"]

        # nearest-column heuristic
        if abs(x0 - deposit_x) <= abs(x0 - withdrawal_x):
            credit += val
        else:
            debit += val

    return round(credit, 2), round(debit, 2)


# =========================================================
# HELPERS
# =========================================================

def safe_float(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None


def clean_description(tokens):
    s = " ".join(tokens)
    s = re.sub(r"\s+", " ", s).strip()
    return s
