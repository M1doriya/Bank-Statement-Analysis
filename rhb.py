import re
import fitz  # PyMuPDF
import pdfplumber
from datetime import datetime
from io import BytesIO


# ======================================================
# Helper: read PDF bytes safely (Streamlit / file / path)
# ======================================================
def _read_pdf_bytes(pdf_input):
    # Case 1: raw bytes
    if isinstance(pdf_input, (bytes, bytearray)):
        return bytes(pdf_input)

    # Case 2: Streamlit UploadedFile
    if hasattr(pdf_input, "getvalue"):
        data = pdf_input.getvalue()
        if data:
            return data

    # Case 3: file-like object
    if hasattr(pdf_input, "read"):
        try:
            pdf_input.seek(0)
        except Exception:
            pass
        data = pdf_input.read()
        if data:
            return data

    # Case 4: file path
    if isinstance(pdf_input, str):
        with open(pdf_input, "rb") as f:
            return f.read()

    raise ValueError("Unable to read PDF bytes")


# ======================================================
# 1️⃣ RHB ISLAMIC — TEXT BASED
# ======================================================
def _parse_rhb_islamic_text(pdf_bytes, source_filename):
    transactions = []
    previous_balance = None

    balance_re = re.compile(r"(-?\d{1,3}(?:,\d{3})*\.\d{2})\s*$")
    date_re = re.compile(r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)")

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        header = pdf.pages[0].extract_text() or ""
        period_match = re.search(r"Statement Period.*?(\d{2})", header, re.IGNORECASE)
        if not period_match:
            return []

        year = int("20" + period_match.group(1))

        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                bal_match = balance_re.search(line)
                date_match = date_re.search(line)
                if not bal_match or not date_match:
                    continue

                balance = float(bal_match.group(1).replace(",", ""))

                if re.search(r"\bB/F\b|\bC/F\b", line):
                    previous_balance = balance
                    continue

                if previous_balance is None:
                    previous_balance = balance
                    continue

                day, month = date_match.groups()
                date_iso = datetime.strptime(
                    f"{day} {month} {year}", "%d %b %Y"
                ).strftime("%Y-%m-%d")

                delta = balance - previous_balance
                debit = abs(delta) if delta < 0 else 0.0
                credit = delta if delta > 0 else 0.0

                desc = re.sub(balance_re, "", line)
                desc = desc.replace(date_match.group(0), "")
                desc = re.sub(r"\s+", " ", desc).strip()

                transactions.append({
                    "date": date_iso,
                    "description": desc,
                    "debit": round(debit, 2),
                    "credit": round(credit, 2),
                    "balance": round(balance, 2),
                    "page": page_index + 1,
                    "bank": "RHB Islamic Bank",
                    "source_file": source_filename
                })

                previous_balance = balance

    return transactions


# ======================================================
# 2️⃣ RHB CONVENTIONAL — TEXT BASED
# ======================================================

def _parse_rhb_conventional_text(pdf_bytes, source_filename):
    transactions = []
    previous_balance = None

    balance_re = re.compile(r"(-?\d{1,3}(?:,\d{3})*\.\d{2})\s*$")
    date_re = re.compile(r"(\d{2})(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)")

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        header = pdf.pages[0].extract_text() or ""
        ym = re.search(r"[A-Za-z]{3}(\d{2})", header)
        if not ym:
            return []

        year = int("20" + ym.group(1))

        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue

            for line in text.split("\n"):
                bal = balance_re.search(line)
                date = date_re.search(line)
                if not bal or not date:
                    continue

                balance = float(bal.group(1).replace(",", ""))

                if previous_balance is None:
                    previous_balance = balance
                    continue

                day, month = date.groups()
                date_iso = datetime.strptime(
                    f"{day}{month}{year}", "%d%b%Y"
                ).strftime("%Y-%m-%d")

                delta = balance - previous_balance
                debit = abs(delta) if delta < 0 else 0.0
                credit = delta if delta > 0 else 0.0

                desc = re.sub(balance_re, "", line)
                desc = desc.replace(date.group(0), "")
                desc = re.sub(r"\s+", " ", desc).strip()

                transactions.append({
                    "date": date_iso,
                    "description": desc,
                    "debit": round(debit, 2),
                    "credit": round(credit, 2),
                    "balance": round(balance, 2),
                    "page": page_index + 1,
                    "bank": "RHB Bank",
                    "source_file": source_filename
                })

                previous_balance = balance

    return transactions


# ======================================================
# 3️⃣ RHB REFLEX — LAYOUT BASED (FIXED VERSION)
# ======================================================
def _parse_rhb_reflex_layout(pdf_bytes, source_filename):
    import re
    import fitz
    import pdfplumber
    from io import BytesIO
    from datetime import datetime
    
    transactions = []
    
    DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
    # Updated MONEY_RE to optionally capture +/- signs
    MONEY_RE = re.compile(r"(?:\d{1,3}(?:,\d{3})*|\d)?\.\d{2}[+-]?")
    
    def norm_date(text):
        return datetime.strptime(text, "%d-%m-%Y").strftime("%Y-%m-%d")
    
    # ==================================================
    # 1️⃣ Extract OPENING BALANCE first (CRITICAL) - FIXED
    # ==================================================
    def extract_opening_balance():
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if "Beginning Balance" in text:
                    # NEW: Handle both positive and negative balances
                    # Matches: "251,613.85", "251,613.85+", or "845,425.30-"
                    m = re.search(r"([\d,]+\.\d{2})([+-])?", text)
                    if m:
                        amount = float(m.group(1).replace(",", ""))
                        # If there's a minus sign, make it negative
                        if m.group(2) == "-":
                            amount = -amount
                        # If plus sign or no sign, keep positive
                        return amount
        return None
    
    previous_balance = extract_opening_balance()
    
    # ==================================================
    # 2️⃣ Parse TRANSACTIONS using layout - FIXED
    # ==================================================
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    for page_index, page in enumerate(doc):
        words = page.get_text("words")
        
        rows = [{
            "x": w[0],
            "y": round(w[1], 1),
            "text": w[4].strip()
        } for w in words if w[4].strip()]
        
        rows.sort(key=lambda r: (r["y"], r["x"]))
        used_y = set()
        
        for r in rows:
            if not DATE_RE.match(r["text"]):
                continue
            
            y = r["y"]
            if y in used_y:
                continue
            
            line = [w for w in rows if abs(w["y"] - y) <= 1.5]
            line.sort(key=lambda w: w["x"])
            
            money = [w for w in line if MONEY_RE.match(w["text"])]
            if len(money) < 2:
                continue
            
            bal_word = money[-1]
            
            # ------------------------------
            # Balance (FIXED: handles both + and -)
            # ------------------------------
            bal_text = bal_word["text"].replace(",", "")
            
            # Check for negative (overdraft)
            is_negative = bal_text.endswith("-")
            # Check for positive (some statements mark with +)
            is_positive = bal_text.endswith("+")
            
            # Remove all signs and convert to float
            bal_val = float(bal_text.replace("-", "").replace("+", ""))
            
            # Apply sign
            if is_negative:
                bal_val = -bal_val
            # If is_positive or no sign, keep positive (default)
            
            # ------------------------------
            # DR / CR by BALANCE MOVEMENT
            # ------------------------------
            debit = credit = 0.0
            if previous_balance is not None:
                delta = bal_val - previous_balance
                if delta < 0:
                    debit = abs(delta)
                elif delta > 0:
                    credit = delta
            
            # ------------------------------
            # Description
            # ------------------------------
            description = [
                w["text"] for w in line
                if w not in money
                and not DATE_RE.match(w["text"])
                and not w["text"].isdigit()
            ]
            
            transactions.append({
                "date": norm_date(r["text"]),
                "description": " ".join(description)[:200],
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(bal_val, 2),
                "page": page_index + 1,
                "bank": "RHB Bank",
                "source_file": source_filename
            })
            
            previous_balance = bal_val
            used_y.add(y)
    
    doc.close()
    return transactions


# ==================================================
# WHAT WAS FIXED:
# ==================================================
"""
1. extract_opening_balance():
   OLD: m = re.search(r"([\d,]+\.\d{2})-", text)
        Only matched negative balances like "845,425.30-"
   
   NEW: m = re.search(r"([\d,]+\.\d{2})([+-])?", text)
        Matches: "251,613.85", "251,613.85+", "845,425.30-"
        Checks group(2) for sign and applies it

2. MONEY_RE pattern:
   OLD: r"(?:\d{1,3}(?:,\d{3})*|\d)?\.\d{2}"
        Didn't include +/- in pattern
   
   NEW: r"(?:\d{1,3}(?:,\d{3})*|\d)?\.\d{2}[+-]?"
        Now optionally matches trailing +/- signs

3. Balance parsing:
   OLD: bal_val = float(bal_word["text"].replace(",", "").replace("-", ""))
        if bal_word["text"].endswith("-"):
            bal_val = -bal_val
        Only checked for dash (-)
   
   NEW: Checks for both "-" and "+"
        - If ends with "-": negative (overdraft)
        - If ends with "+": positive (explicit)
        - If no sign: positive (default)

RESULT: Now works for BOTH overdraft and regular accounts!
"""

def parse_transactions_rhb(pdf_input, source_filename):
    pdf_bytes = _read_pdf_bytes(pdf_input)

    for parser in (
        _parse_rhb_islamic_text,
        _parse_rhb_conventional_text,
        _parse_rhb_reflex_layout,
    ):
        try:
            tx = parser(pdf_bytes, source_filename)
            if tx:
                return tx
        except Exception:
            continue

    return []

