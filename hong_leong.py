import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pytesseract
from PIL import Image, ImageEnhance, ImageOps


# =========================================================
# Regex
# =========================================================
DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}")
HEADER_DATE_LINE_RE = re.compile(r"\bDate\s*/\s*Tarikh\s*:\s*\d{2}-\d{2}-\d{4}\b", re.I)
TOTAL_ROW_RE = re.compile(
    r"(Total\s+Deposits|Total\s+Withdrawals|Closing\s+Balance|Important\s+Notices|"
    r"Jumlah\s+Simpanan|Jumlah\s+Pengeluaran|Baki\s+Akhir|Notis\s+Penting)",
    re.I,
)
OD_KEYWORDS_RE = re.compile(
    r"\b(overdraft|od\s+facility|od\s+limit|overdrawn|excess\s+limit|interest\s+on\s+overdraft)\b",
    re.I,
)


# =========================================================
# Public entrypoint used by your app
# =========================================================

def parse_hong_leong(pdf, filename: str) -> List[Dict]:
    """
    Hong Leong Islamic Bank (HLIB) Current Account-i statement parser.

    Objectives:
      1) Accurate debit/credit totals (avoid "hongLeong3.json" inflation).
      2) Avoid false OD / false negative lowest-balance when Balance column extraction fails
         (the failure mode in hongLeong.json / hongLeong2.json).
      3) Keep output schema stable.

    Strategy:
      - Extract transaction blocks by date row + continuation rows.
      - Extract Deposit/Withdrawal amounts by cropping their columns and regexing amounts.
      - Try to extract statement BALANCE from Balance column crop (text first, then OCR).
      - Maintain two balances:
          * computed_balance: opening + credits - debits (always updated)
          * reliable_balance: last extracted statement balance (only updated when found)
      - Output "balance" uses:
          * extracted statement balance when available
          * otherwise the last reliable balance (carry-forward) to prevent false OD
            (because computed_balance can drift when some amounts are missed/garbled).

    If the statement truly had OD and printed a negative balance, this code will still show it
    when it can extract the Balance column. The false-OD problem occurs when Balance column
    is not extractable and computed_balance goes negative due to parsing imperfections.
    """

    transactions: List[Dict] = []

    opening_balance = extract_opening_balance(pdf)
    computed_balance = float(opening_balance)

    # Last balance we trust from the statement's Balance column
    reliable_balance: Optional[float] = None

    # If the statement explicitly mentions OD, we allow negative computed balances to surface
    # (because OD might be real). Otherwise, we treat negative computed balances as unreliable.
    overdraft_possible = pdf_mentions_overdraft(pdf)

    for page_num, page in enumerate(pdf.pages, start=1):
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        if not words:
            continue

        rows = group_words_by_row(words, tolerance=3)
        col_x = detect_amount_columns(rows)

        i = 0
        while i < len(rows):
            row = rows[i]
            row_txt = join_row_text(row)

            # Skip totals/summary lines early
            if TOTAL_ROW_RE.search(row_txt):
                i += 1
                continue

            date_iso = extract_date(row)
            if not date_iso:
                i += 1
                continue

            # Skip header "Date / Tarikh : DD-MM-YYYY" lines (not transactions)
            if HEADER_DATE_LINE_RE.search(row_txt):
                i += 1
                continue

            # Build a transaction block: date row + continuation rows until next date row
            y_top, y_bot = row_y_span(row)

            j = i + 1
            while j < len(rows):
                next_txt = join_row_text(rows[j])
                if extract_date(rows[j]) is not None:
                    break
                if TOTAL_ROW_RE.search(next_txt):
                    break
                y2_top, y2_bot = row_y_span(rows[j])
                y_top = min(y_top, y2_top)
                y_bot = max(y_bot, y2_bot)
                j += 1

            # Extract deposit/withdrawal from column crops (text -> OCR fallback)
            deposit_amt = extract_column_amount(page, y_top, y_bot, col_x, which="deposit")
            withdrawal_amt = extract_column_amount(page, y_top, y_bot, col_x, which="withdrawal")

            # If no movement, skip
            if deposit_amt == 0.0 and withdrawal_amt == 0.0:
                i = j
                continue

            # Update computed balance always
            computed_balance = round(computed_balance + deposit_amt - withdrawal_amt, 2)

            # Try to extract statement balance for this block (text -> OCR)
            stmt_balance = extract_statement_balance(page, y_top, y_bot, col_x)

            # Decide what to output as "balance"
            if stmt_balance is not None:
                # Anchor both balances to statement balance (prevents drift)
                reliable_balance = float(stmt_balance)
                computed_balance = float(stmt_balance)
                out_balance = float(stmt_balance)
            else:
                # No statement balance available:
                # - if we already have a reliable balance, carry it forward (prevents false OD)
                # - else fall back to computed balance (early pages before any anchor)
                if reliable_balance is not None:
                    out_balance = float(reliable_balance)
                else:
                    out_balance = float(computed_balance)

                # If OD is NOT indicated/possible, do not surface negative computed drift as if real
                if (not overdraft_possible) and out_balance < 0 and reliable_balance is not None:
                    out_balance = float(reliable_balance)

            transactions.append(
                {
                    "date": date_iso,
                    "description": extract_description_text(rows, i, j, col_x),
                    "debit": round(float(withdrawal_amt), 2),
                    "credit": round(float(deposit_amt), 2),
                    "balance": round(float(out_balance), 2),
                    "page": int(page_num),
                    "bank": "Hong Leong Islamic Bank",
                    "source_file": filename,
                }
            )

            i = j

    return transactions


# =========================================================
# Opening balance / OD keyword scan
# =========================================================

def extract_opening_balance(pdf) -> float:
    text = pdf.pages[0].extract_text() or ""
    m = re.search(r"Balance\s+from\s+previous\s+statement\s+([\d,]+\.\d{2})", text, re.I)
    if not m:
        raise ValueError("Opening balance not found")
    return float(m.group(1).replace(",", ""))


def pdf_mentions_overdraft(pdf) -> bool:
    # If statement explicitly mentions OD, allow negative to pass through as potentially real
    for p in pdf.pages:
        t = (p.extract_text() or "")
        if OD_KEYWORDS_RE.search(t):
            return True
    return False


# =========================================================
# Row grouping / geometry
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
    rows.sort(key=lambda r: r[0]["top"])
    return rows


def row_y_span(row) -> Tuple[float, float]:
    tops = [w["top"] for w in row]
    bottoms = [w["bottom"] for w in row]
    return float(min(tops)), float(max(bottoms))


def join_row_text(row) -> str:
    return " ".join((w.get("text") or "").strip() for w in row if (w.get("text") or "").strip())


# =========================================================
# Column detection
# =========================================================

def detect_amount_columns(rows) -> Dict[str, float]:
    deposit_x = withdrawal_x = balance_x = None

    for row in rows:
        joined = " ".join((w.get("text") or "").strip().lower() for w in row)
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

        if "simpanan" in joined and "pengeluaran" in joined and "baki" in joined:
            for w in row:
                t = (w.get("text") or "").strip().lower()
                if t == "simpanan" and deposit_x is None:
                    deposit_x = w["x0"]
                elif t == "pengeluaran" and withdrawal_x is None:
                    withdrawal_x = w["x0"]
                elif t == "baki" and balance_x is None:
                    balance_x = w["x0"]
            break

    # fallback anchors (stable for your HLIB PDFs)
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
# Date / Description
# =========================================================

def extract_date(row) -> Optional[str]:
    for w in row:
        t = (w.get("text") or "").strip()
        if DATE_RE.fullmatch(t):
            return datetime.strptime(t, "%d-%m-%Y").date().isoformat()
    return None


def extract_description_text(rows, i: int, j: int, col_x: Dict[str, float]) -> str:
    """
    Build description from tokens left of deposit column across the block [i, j).
    This avoids numeric pollution and avoids the 'inflated totals' issue.
    """
    cutoff = col_x["deposit_x"] - 10
    parts: List[str] = []

    for k in range(i, j):
        for w in rows[k]:
            t = (w.get("text") or "").strip()
            if not t:
                continue
            if DATE_RE.fullmatch(t):
                continue
            # ignore obvious noise
            if is_noise_token(t):
                continue
            # take only left-side tokens (description area)
            if float(w["x0"]) < cutoff:
                parts.append(t)

    s = " ".join(parts)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_noise_token(text: str) -> bool:
    return bool(
        re.search(
            r"(Protected by PIDM|Dilindungi oleh PIDM|Hong Leong Islamic Bank|hlisb\.com\.my|Menara Hong Leong|CURRENT ACCOUNT|AKAUN SEMASA|Page No)",
            text,
            re.I,
        )
    )


# =========================================================
# Column crop extraction (text -> OCR fallback)
# =========================================================

def extract_column_amount(page, y_top: float, y_bot: float, col_x: Dict[str, float], which: str) -> float:
    """
    Extract numeric amount from Deposit or Withdrawal column by cropping.
    Returns sum of all amounts found in that column for the block (usually 0 or 1 amount).
    """
    dep = col_x["deposit_x"]
    wdr = col_x["withdrawal_x"]
    bal = col_x["balance_x"]

    if which == "deposit":
        x0, x1 = dep - 80, wdr - 15
    elif which == "withdrawal":
        x0, x1 = wdr - 80, bal - 15
    else:
        raise ValueError("which must be 'deposit' or 'withdrawal'")

    nums = extract_amounts_from_bbox(page, (x0, y_top - 1, x1, y_bot + 1))
    if nums:
        return round(sum(nums), 2)

    # OCR fallback if text layer fails
    nums = extract_amounts_from_bbox_ocr(page, (x0, y_top - 1, x1, y_bot + 1))
    if nums:
        return round(sum(nums), 2)

    return 0.0


def extract_statement_balance(page, y_top: float, y_bot: float, col_x: Dict[str, float]) -> Optional[float]:
    """
    Extract statement balance from Balance column crop.
    Returns the last (most right-aligned) amount in the balance column region when available.

    Many HLIB PDFs do not expose Balance column reliably in the text layer for all rows.
    We try text first, then OCR.
    """
    bal = col_x["balance_x"]
    x0, x1 = bal - 80, page.width

    nums = extract_amounts_from_bbox(page, (x0, y_top - 1, x1, y_bot + 1))
    if not nums:
        nums = extract_amounts_from_bbox_ocr(page, (x0, y_top - 1, x1, y_bot + 1))

    if not nums:
        return None

    # Heuristic: if there are multiple numbers (e.g., "800.00 12,132.58"), the last is balance
    candidate = float(nums[-1])

    # Guard: if only one number and it equals a typical debit/credit in that row, balance may be missing
    # (We let caller decide; here we just return the candidate we found.)
    return round(candidate, 2)


def extract_amounts_from_bbox(page, bbox) -> List[float]:
    """
    Extract all money amounts from a cropped bbox using text layer.
    """
    try:
        region = page.crop(bbox)
        txt = region.extract_text(x_tolerance=1) or ""
        matches = MONEY_RE.findall(txt)
        return [float(m.replace(",", "")) for m in matches] if matches else []
    except Exception:
        return []


def extract_amounts_from_bbox_ocr(page, bbox, dpi=350) -> List[float]:
    """
    OCR fallback, tightly scoped to a bbox. Used sparingly to avoid performance issues.
    """
    try:
        region = page.crop(bbox)
        im = region.to_image(resolution=dpi).original
        im = preprocess_for_ocr(im)

        text = pytesseract.image_to_string(
            im,
            config="--psm 6 -c tessedit_char_whitelist=0123456789.,-",
        )

        matches = MONEY_RE.findall(text)
        if not matches:
            return []

        return [float(m.replace(",", "")) for m in matches]
    except Exception:
        return []


def preprocess_for_ocr(im: Image.Image) -> Image.Image:
    """
    Conservative image preprocessing for OCR on numeric tables.
    """
    im = im.convert("L")
    im = ImageOps.autocontrast(im)
    im = ImageEnhance.Contrast(im).enhance(2.2)
    return im
