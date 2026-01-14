import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# =========================================================
# Regex
# =========================================================
DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
MONEY_RE = re.compile(r"^\d{1,3}(?:,\d{3})*\.\d{2}$")
MONEY_FINDALL_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}")


# =========================================================
# MAIN ENTRY (USED BY app.py)
# =========================================================

def parse_hong_leong(pdf, filename: str) -> List[Dict]:
    """
    Hong Leong Islamic Bank (HLIB) Current Account-i parser.

    Key fix:
    - Uses statement BALANCE column (extracted via Balance-column crop) as source of truth.
    - Computes debit/credit from balance delta.
    - Falls back to deposit/withdrawal classification only if balance cannot be extracted.
    - Never emits "transactions" with debit=0 and credit=0 (prevents header/date lines).

    This eliminates false negative balances (and therefore false OD) on months like September.
    """
    transactions: List[Dict] = []

    opening_balance = extract_opening_balance(pdf)
    prev_balance: Optional[float] = opening_balance

    for page_num, page in enumerate(pdf.pages, start=1):
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        if not words:
            continue

        rows = group_words_by_row(words, tolerance=3)

        # Detect Deposit/Withdrawal/Balance column x positions per page
        col_x = detect_amount_columns(rows)

        i = 0
        while i < len(rows):
            row = rows[i]

            date_iso = extract_date(row)
            if not date_iso:
                i += 1
                continue

            if is_total_row(row):
                i += 1
                continue

            # Build a "transaction block": the dated row + continuation rows until next date
            desc_tokens: List[str] = []
            amt_tokens: List[Dict] = []  # money tokens in deposit/withdrawal area (not balance)
            y_top, y_bottom = row_y_span(row)

            desc_tokens.extend(extract_desc_tokens(row, col_x))
            amt_tokens.extend(extract_amount_tokens(row, col_x))
            j = i + 1

            while j < len(rows) and extract_date(rows[j]) is None:
                if is_total_row(rows[j]):
                    break
                desc_tokens.extend(extract_desc_tokens(rows[j], col_x))
                amt_tokens.extend(extract_amount_tokens(rows[j], col_x))
                y2_top, y2_bottom = row_y_span(rows[j])
                y_top = min(y_top, y2_top)
                y_bottom = max(y_bottom, y2_bottom)
                j += 1

            description = clean_description(desc_tokens)

            # Skip obvious header/address date lines (they have a date but no amounts)
            # Example: "Date / Tarikh : 25-09-2025"
            if is_header_date_line(description) and not amt_tokens:
                i = j
                continue

            # 1) Prefer: extract BALANCE from a cropped region in the Balance column across this block
            stmt_balance = extract_balance_from_crop(page, y_top, y_bottom, col_x)

            # If crop-based balance failed, try word-based last resort (rare)
            if stmt_balance is None:
                stmt_balance = extract_balance_from_words(page, y_top, y_bottom, col_x)

            credit = 0.0
            debit = 0.0
            balance_out: Optional[float] = None

            if stmt_balance is not None and prev_balance is not None:
                # Balance-delta method
                delta = round(stmt_balance - prev_balance, 2)
                if delta > 0:
                    credit = delta
                elif delta < 0:
                    debit = abs(delta)
                else:
                    # Usually not a real transaction row; update prev balance defensively and skip
                    prev_balance = stmt_balance
                    i = j
                    continue

                balance_out = stmt_balance
                prev_balance = stmt_balance

            else:
                # 2) Fallback: deposit/withdrawal classification (less reliable)
                credit, debit = classify_amounts_by_columns(amt_tokens, col_x)

                # If no amounts, skip
                if credit == 0.0 and debit == 0.0:
                    i = j
                    continue

                # Compute running balance only as a fallback
                if prev_balance is None:
                    prev_balance = 0.0
                prev_balance = round(prev_balance + credit - debit, 2)
                balance_out = prev_balance

            # Never output a transaction with both 0
            if credit == 0.0 and debit == 0.0:
                i = j
                continue

            transactions.append({
                "date": date_iso,
                "description": description,
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(float(balance_out), 2) if balance_out is not None else None,
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
# ROW GROUPING / GEOMETRY
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


def row_y_span(row) -> Tuple[float, float]:
    tops = [w["top"] for w in row if "top" in w]
    bottoms = [w["bottom"] for w in row if "bottom" in w]
    if not tops or not bottoms:
        return 0.0, 0.0
    return float(min(tops)), float(max(bottoms))


# =========================================================
# COLUMN DETECTION (Deposit / Withdrawal / Balance)
# =========================================================

def detect_amount_columns(rows) -> Dict[str, float]:
    deposit_x = withdrawal_x = balance_x = None

    for row in rows:
        joined = " ".join((w.get("text") or "").strip().lower() for w in row)

        # English header row
        if "deposit" in joined and "withdrawal" in joined and "balance" in joined:
            for w in row:
                t = (w.get("text") or "").strip().lower()
                if t == "deposit":
                    deposit_x = w["x0"]
                elif t == "withdrawal":
                    withdrawal_x = w["x0"]
                elif t == "balance":
                    balance_x = w["x0"]
            break

        # Malay header row
        if "simpanan" in joined and "pengeluaran" in joined and "baki" in joined:
            for w in row:
                t = (w.get("text") or "").strip().lower()
                if t == "simpanan":
                    deposit_x = deposit_x or w["x0"]
                elif t == "pengeluaran":
                    withdrawal_x = withdrawal_x or w["x0"]
                elif t == "baki":
                    balance_x = balance_x or w["x0"]
            break

    # Fallbacks if header isn't captured on this page
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
# DATE / FILTERS
# =========================================================

def extract_date(row) -> Optional[str]:
    for w in row:
        t = (w.get("text") or "").strip()
        if DATE_RE.fullmatch(t):
            return datetime.strptime(t, "%d-%m-%Y").strftime("%Y-%m-%d")
    return None


def is_total_row(row) -> bool:
    text = " ".join((w.get("text") or "") for w in row).lower()
    keys = (
        "total withdrawals",
        "total deposits",
        "closing balance",
        "important notices",
        "jumlah pengeluaran",
        "jumlah simpanan",
        "baki akhir",
        "notis penting",
    )
    return any(k in text for k in keys)


def is_header_date_line(description: str) -> bool:
    s = (description or "").lower()
    return ("date" in s and "tarikh" in s and ":" in s)


# =========================================================
# DESCRIPTION / AMOUNTS EXTRACTION
# =========================================================

def is_noise_token(text: str) -> bool:
    return bool(re.search(
        r"Protected by PIDM|Dilindungi oleh PIDM|Hong Leong Islamic Bank|hlisb\.com\.my|Menara Hong Leong|CURRENT ACCOUNT|AKAUN SEMASA",
        text,
        re.IGNORECASE
    ))


def extract_desc_tokens(row, col_x) -> List[str]:
    """
    Only take tokens left of the Deposit column (description area).
    This prevents numeric columns leaking into description.
    """
    out = []
    cutoff = col_x["deposit_x"] - 10
    for w in row:
        t = (w.get("text") or "").strip()
        if not t:
            continue
        if DATE_RE.fullmatch(t):
            continue
        if is_noise_token(t):
            continue
        if float(w["x0"]) < cutoff:
            out.append(t)
    return out


def extract_amount_tokens(row, col_x) -> List[Dict]:
    """
    Extract money tokens that are *not* in the Balance column area.
    This is only used as a fallback when balance extraction fails.
    """
    out = []
    min_amount_x = col_x["deposit_x"] - 25
    balance_guard_x = col_x["balance_x"] - 30  # anything to the right is likely balance column

    for w in row:
        t = (w.get("text") or "").strip()
        if not MONEY_RE.fullmatch(t):
            continue

        x0 = float(w["x0"])
        if x0 < min_amount_x:
            continue

        # exclude balance column numbers
        if x0 >= balance_guard_x:
            continue

        out.append({"x": x0, "value": float(t.replace(",", ""))})

    return out


def clean_description(parts: List[str]) -> str:
    s = " ".join(parts)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================================================
# BALANCE EXTRACTION (AUTHORITATIVE)
# =========================================================

def extract_balance_from_crop(page, y_top: float, y_bottom: float, col_x: Dict[str, float]) -> Optional[float]:
    """
    Most reliable method on these HLIB PDFs:
    crop the Balance column region for this transaction block and regex the last money value.
    """
    try:
        bal_x = float(col_x["balance_x"])
        x0 = max(bal_x - 20.0, 0.0)
        x1 = float(getattr(page, "width", 600.0))
        top = max(y_top - 1.0, 0.0)
        bottom = min(y_bottom + 1.0, float(getattr(page, "height", y_bottom + 10.0)))

        region = page.crop((x0, top, x1, bottom))
        text = region.extract_text(x_tolerance=1) or ""
        amounts = MONEY_FINDALL_RE.findall(text)
        if not amounts:
            return None
        return float(amounts[-1].replace(",", ""))
    except Exception:
        return None


def extract_balance_from_words(page, y_top: float, y_bottom: float, col_x: Dict[str, float]) -> Optional[float]:
    """
    Last-resort fallback: use word tokens near the Balance column within the block y-range.
    """
    try:
        bal_x = float(col_x["balance_x"])
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        # Keep money tokens within y-range and near balance column
        candidates = []
        for w in words:
            t = (w.get("text") or "").strip()
            if not MONEY_RE.fullmatch(t):
                continue
            if float(w.get("top", 0.0)) < y_top - 1.0 or float(w.get("bottom", 0.0)) > y_bottom + 1.0:
                continue
            x0 = float(w.get("x0", 0.0))
            if abs(x0 - bal_x) <= 70.0:
                candidates.append((abs(x0 - bal_x), -x0, t))
        if not candidates:
            return None
        candidates.sort()
        best = candidates[0][2]
        return float(best.replace(",", ""))
    except Exception:
        return None


# =========================================================
# FALLBACK AMOUNT CLASSIFICATION
# =========================================================

def classify_amounts_by_columns(amount_words: List[Dict], col_x: Dict[str, float]) -> Tuple[float, float]:
    """
    Fallback classification if balance extraction fails.
    Uses proximity to deposit_x vs withdrawal_x.
    """
    credit = 0.0
    debit = 0.0

    dep = float(col_x["deposit_x"])
    wdr = float(col_x["withdrawal_x"])

    for a in amount_words:
        x = float(a["x"])
        val = float(a["value"])
        if abs(x - dep) <= abs(x - wdr):
            credit += val
        else:
            debit += val

    return round(credit, 2), round(debit, 2)
