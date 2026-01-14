import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# =========================================================
# MAIN ENTRY (USED BY app.py)
# =========================================================

def parse_hong_leong(pdf, filename):
    """
    Robust parser for Hong Leong Islamic Bank CURRENT ACCOUNT-i statements.

    Key improvements vs old version:
    - Uses per-page money-x clustering to locate Deposit/Withdrawal/Balance columns.
    - Extracts statement balance from Balance column when present (trusted).
    - Uses balance delta as sanity check to avoid impossible debit=credit cases.
    - No OCR / no pytesseract dependency.
    """
    transactions = []

    opening_balance = extract_opening_balance(pdf)
    running_balance = opening_balance

    for page_num, page in enumerate(pdf.pages, start=1):
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        if not words:
            continue

        rows = group_words_by_row(words, tolerance=2.5)

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

            desc_tokens: List[str] = []
            amount_tokens: List[Dict[str, float]] = []

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

            credit, debit, extracted_balance = classify_amounts_and_balance(amount_tokens, col_x)

            # If we have an extracted statement balance, use it to infer amounts when needed
            if extracted_balance is not None:
                delta = round(extracted_balance - running_balance, 2)

                # Case A: No usable deposit/withdrawal detected -> infer from delta
                if credit == 0.0 and debit == 0.0:
                    if delta > 0:
                        credit = delta
                    elif delta < 0:
                        debit = -delta

                # Case B: Conflicting / suspicious classification -> trust delta if it explains the movement
                # (This avoids "debit==credit" and "balance used as debit" artifacts.)
                else:
                    # Expected delta from classified amounts
                    implied_delta = round(credit - debit, 2)

                    # If mismatch is large, rebuild using delta (single-sided)
                    if abs(implied_delta - delta) > 0.01:
                        if delta > 0:
                            credit = delta
                            debit = 0.0
                        elif delta < 0:
                            debit = -delta
                            credit = 0.0
                        else:
                            credit = 0.0
                            debit = 0.0

            # If still no movement, skip
            if credit == 0.0 and debit == 0.0:
                i = j
                continue

            # Update running balance:
            # - If statement balance present, trust it
            # - Else compute from amounts
            if extracted_balance is not None:
                running_balance = round(extracted_balance, 2)
            else:
                running_balance = round(running_balance + credit - debit, 2)

            transactions.append({
                "date": date,
                "description": clean_description(desc_tokens),
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(running_balance, 2),
                "page": page_num,
                "bank": "Hong Leong Islamic Bank",
                "source_file": filename
            })

            i = j

    return transactions


# =========================================================
# OPENING BALANCE
# =========================================================

def extract_opening_balance(pdf) -> float:
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

def group_words_by_row(words, tolerance=2.5):
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

_MONEY_RE = re.compile(r"^[\d,]+\.\d{2}$")

def detect_amount_columns(rows) -> Dict[str, float]:
    """
    Robustly detect deposit / withdrawal / balance columns.

    First attempt: use header labels if present.
    Fallback: cluster money token x-positions and take the 3 rightmost clusters:
      left -> deposit, mid -> withdrawal, right -> balance
    """
    deposit_x = withdrawal_x = balance_x = None

    # 1) Try header-based detection
    for row in rows:
        joined = " ".join(w["text"].strip().lower() for w in row)
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

    if deposit_x is not None and withdrawal_x is not None and balance_x is not None:
        return {
            "deposit_x": float(deposit_x),
            "withdrawal_x": float(withdrawal_x),
            "balance_x": float(balance_x),
        }

    # 2) Fallback: x-cluster money tokens (page layout can shift!)
    x_positions: List[float] = []
    for row in rows:
        if is_total_row(row):
            continue
        for w in row:
            t = w["text"].strip()
            if _MONEY_RE.fullmatch(t) and w["x0"] >= 250:  # ignore description area
                x_positions.append(float(w["x0"]))

    # If not enough data, last resort to sensible defaults (but wider)
    if len(x_positions) < 10:
        return {
            "deposit_x": 320.0,
            "withdrawal_x": 410.0,
            "balance_x": 520.0,
        }

    # Simple 1D k-means (k=3) on x positions to find three columns
    centroids = _kmeans_1d(x_positions, k=3, iters=12)
    centroids.sort()

    # The statement columns are the 3 rightmost money columns.
    # Sometimes there can be extra numeric columns; to be safe, take the 3 largest centroids.
    # (If centroids already 3, this is a no-op.)
    dep, wdr, bal = centroids[-3], centroids[-2], centroids[-1]

    return {"deposit_x": dep, "withdrawal_x": wdr, "balance_x": bal}


def _kmeans_1d(values: List[float], k: int = 3, iters: int = 10) -> List[float]:
    v = sorted(values)
    # init centroids from quantiles
    def q(p: float) -> float:
        idx = int(p * (len(v) - 1))
        return float(v[idx])

    centroids = [q(0.2), q(0.5), q(0.8)][:k]

    for _ in range(iters):
        buckets = [[] for _ in range(k)]
        for x in v:
            j = min(range(k), key=lambda i: abs(x - centroids[i]))
            buckets[j].append(x)
        for i in range(k):
            if buckets[i]:
                centroids[i] = float(sum(buckets[i]) / len(buckets[i]))
    return centroids


# =========================================================
# DATE DETECTION
# =========================================================

def extract_date(row) -> Optional[str]:
    for w in row:
        if re.fullmatch(r"\d{2}-\d{2}-\d{4}", w["text"]):
            return datetime.strptime(w["text"], "%d-%m-%Y").strftime("%Y-%m-%d")
    return None


# =========================================================
# TOKEN EXTRACTION
# =========================================================

def extract_amount_tokens(row, col_x: Dict[str, float]) -> List[Dict[str, float]]:
    """
    Only take money tokens that appear in the amount columns region.
    """
    out: List[Dict[str, float]] = []
    min_x = min(col_x["deposit_x"], col_x["withdrawal_x"], col_x["balance_x"]) - 35.0

    for w in row:
        t = w["text"].strip()
        if _MONEY_RE.fullmatch(t) and w["x0"] >= min_x:
            out.append({"x": float(w["x0"]), "value": float(t.replace(",", ""))})

    return out


def extract_desc_tokens(row) -> List[str]:
    out: List[str] = []
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
# AMOUNT CLASSIFICATION + BALANCE EXTRACTION
# =========================================================

def classify_amounts_and_balance(
    amount_words: List[Dict[str, float]],
    col_x: Dict[str, float]
) -> Tuple[float, float, Optional[float]]:
    """
    Uses column boundaries (midpoints) rather than nearest-distance.
    This is much less sensitive to small layout shifts.

    Returns: (credit, debit, extracted_balance)
    """
    dep = col_x["deposit_x"]
    wdr = col_x["withdrawal_x"]
    bal = col_x["balance_x"]

    # boundaries between columns
    b1 = (dep + wdr) / 2.0
    b2 = (wdr + bal) / 2.0

    credit = 0.0
    debit = 0.0
    balance_candidates: List[Tuple[float, float]] = []  # (x, value)

    for a in amount_words:
        x = a["x"]
        val = a["value"]
        if x < b1:
            credit += val
        elif x < b2:
            debit += val
        else:
            balance_candidates.append((x, val))

    extracted_balance = None
    if balance_candidates:
        # pick the rightmost balance token (most likely actual balance)
        balance_candidates.sort(key=lambda t: t[0])
        extracted_balance = balance_candidates[-1][1]

    return round(credit, 2), round(debit, 2), extracted_balance


# =========================================================
# FILTERS / CLEANUP
# =========================================================

def is_total_row(row) -> bool:
    text = " ".join(w["text"] for w in row)
    return bool(re.search(
        r"Total Withdrawals|Total Deposits|Closing Balance|Important Notices",
        text,
        re.IGNORECASE
    ))


def is_noise(text: str) -> bool:
    return bool(re.search(
        r"Protected by PIDM|Dilindungi oleh PIDM|Hong Leong Islamic Bank|hlisb\.com\.my|Menara Hong Leong|CURRENT ACCOUNT",
        text,
        re.IGNORECASE
    ))


def clean_description(parts: List[str]) -> str:
    s = " ".join(parts)
    s = re.sub(r"\s+", " ", s).strip()
    return s
