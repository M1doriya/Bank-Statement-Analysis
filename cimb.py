# cimb.py - CIMB Bank Parser (fixed ordering + full-doc closing balance)
#
# Key fixes:
# 1) CIMB statements list latest transaction as #1 (reverse chronological).
#    We MUST reorder extracted rows to chronological (oldest->newest) for any
#    running-balance/delta logic and for consistent downstream processing.
#    For same-date rows, we reverse within the date too.
# 2) Closing balance "CLOSING BALANCE / BAKI PENUTUP" is usually near the END of PDF.
#    We scan the FULL document text, not just first pages.
# 3) Extract statement totals (Total Withdrawal/Total Deposits) from footer text
#    for reference / future use.

import re
from datetime import datetime


# -----------------------------
# Regex helpers
# -----------------------------

_MONEY_TOKEN_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*\.\d{2}$")

_CIMB_STMT_DATE_RE = re.compile(
    r"(?:STATEMENT\s+DATE|TARIKH\s+PENYATA)\s*[:\s]+\d{1,2}/\d{1,2}/(\d{2,4})",
    re.IGNORECASE
)

_CIMB_CLOSING_RE = re.compile(
    r"CLOSING\s+BALANCE\s*/\s*BAKI\s+PENUTUP\s+(-?[\d,]+\.\d{2})",
    re.IGNORECASE
)


def parse_float(value):
    if value is None:
        return 0.0
    clean = str(value).replace("\n", "").replace(" ", "").replace(",", "")
    if clean == "":
        return 0.0
    if not re.match(r"^-?\d+(\.\d+)?$", clean):
        return 0.0
    try:
        return float(clean)
    except Exception:
        return 0.0


def clean_text(text):
    if not text:
        return ""
    return str(text).replace("\n", " ").strip()


def extract_year_from_text(text):
    if not text:
        return None

    m = re.search(
        r"(?:STATEMENT DATE|TARIKH PENYATA)\s*[:\s]+\d{1,2}/\d{1,2}/(\d{2,4})",
        text,
        re.IGNORECASE
    )
    if m:
        y = m.group(1)
        return y if len(y) == 4 else str(2000 + int(y))
    return None


def extract_closing_balance_from_text(text):
    if not text:
        return None
    m = _CIMB_CLOSING_RE.search(text)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def extract_statement_totals_from_text(text):
    """Extract statement totals from the footer when present.

    Common layout near end:
      No of Withdrawal  No of Deposits  Total Withdrawal  Total Deposits
      320              51              5,511,545.17      6,609,360.94

    Returns (total_withdrawal, total_deposits) or (None, None).
    """
    if not text:
        return (None, None)

    up = text.upper()
    if "TOTAL WITHDRAWAL" not in up or "TOTAL DEPOSITS" not in up:
        return (None, None)

    idx = up.rfind("TOTAL WITHDRAWAL")
    window = text[idx: idx + 900] if idx != -1 else text

    m = re.search(r"\b\d{1,6}\s+\d{1,6}\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})\b", window)
    if m:
        return (parse_float(m.group(1)), parse_float(m.group(2)))

    money = re.findall(r"-?[\d,]+\.\d{2}", window)
    if len(money) >= 2:
        return (parse_float(money[-2]), parse_float(money[-1]))

    return (None, None)


def format_date(date_str, year):
    if not date_str:
        return None
    date_str = clean_text(date_str)

    m = re.match(r"(\d{2})/(\d{2})/(\d{4})$", date_str)
    if m:
        dd, mm, yyyy = m.groups()
        return f"{yyyy}-{mm}-{dd}"

    m = re.match(r"(\d{2})/(\d{2})$", date_str)
    if m:
        dd, mm = m.groups()
        return f"{year}-{mm}-{dd}"

    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    return None


def _dedupe_transactions(transactions):
    """De-duplicate identical rows."""
    if not transactions:
        return []

    seen = set()
    out = []
    for t in transactions:
        key = (
            t.get("date"),
            clean_text(t.get("description")),
            clean_text(t.get("ref_no", "")),
            round(parse_float(t.get("debit")), 2),
            round(parse_float(t.get("credit")), 2),
            None if t.get("balance") is None else round(parse_float(t.get("balance")), 2),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _chronological_sort(rows):
    """
    CIMB statement order is reverse chronological (newest first).
    We convert to chronological (oldest first) by sorting:
      (date ascending, extracted_index descending)
    so within the same date we also reverse order.
    """
    def key(r):
        return (r.get("date") or "9999-99-99", -int(r.get("__idx", 0)))
    return sorted(rows, key=key)


def _extract_amounts_and_balance(line):
    toks = line.split()
    money_idx = [i for i, t in enumerate(toks) if _MONEY_TOKEN_RE.match(t)]
    if not money_idx:
        return [], None, None

    last_idx = money_idx[-1]
    balance = parse_float(toks[last_idx])
    amount_tokens = [toks[i] for i in money_idx[:-1]]

    first_money_idx = None
    for i, t in enumerate(toks):
        if t == "0" or _MONEY_TOKEN_RE.match(t):
            first_money_idx = i
            break

    return amount_tokens, balance, first_money_idx


def _parse_transactions_cimb_text(pdf, source_filename, detected_year, bank_name, closing_balance):
    """
    Text-mode parser:
    - collect raw rows with (date, desc, balance)
    - then reorder to chronological and infer debit/credit by balance deltas
    """
    raw_rows = []
    opening_balance = None
    latest_tx_date = None

    idx = 0
    cur = None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        for ln in lines:
            up = ln.upper()

            if up.startswith("OPENING BALANCE"):
                _, bal, _ = _extract_amounts_and_balance(ln)
                if bal is not None:
                    opening_balance = bal
                continue

            if "CLOSING BALANCE" in up and "BAKI" in up:
                continue

            m = re.match(r"^(\d{2}/\d{2}/\d{4})\s+(.*)$", ln)
            if m:
                cur = {"date": m.group(1), "parts": [m.group(2)], "page": page_num}

                amount_tokens, bal, first_money_idx = _extract_amounts_and_balance(ln)
                if bal is not None:
                    toks = ln.split()
                    desc = " ".join(toks[1:first_money_idx]) if first_money_idx is not None else " ".join(toks[1:])
                    date_iso = format_date(cur["date"], detected_year)
                    if date_iso:
                        idx += 1
                        raw_rows.append({
                            "date": date_iso,
                            "description": clean_text(desc),
                            "balance": round(bal, 2),
                            "amount_tokens": amount_tokens,
                            "page": page_num,
                            "__idx": idx,
                        })
                        if latest_tx_date is None or date_iso > latest_tx_date:
                            latest_tx_date = date_iso
                    cur = None
                continue

            if cur is not None:
                amount_tokens, bal, first_money_idx = _extract_amounts_and_balance(ln)
                if bal is not None:
                    toks = ln.split()
                    cur["parts"].append(" ".join(toks[:first_money_idx]) if first_money_idx is not None else ln)

                    date_iso = format_date(cur["date"], detected_year)
                    if date_iso:
                        idx += 1
                        raw_rows.append({
                            "date": date_iso,
                            "description": clean_text(" ".join(cur["parts"])),
                            "balance": round(bal, 2),
                            "amount_tokens": amount_tokens,
                            "page": cur["page"],
                            "__idx": idx,
                        })
                        if latest_tx_date is None or date_iso > latest_tx_date:
                            latest_tx_date = date_iso
                    cur = None
                else:
                    cur["parts"].append(ln)

    # Full-doc closing balance fallback
    if closing_balance is None:
        full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        closing_balance = extract_closing_balance_from_text(full_text)

    # FIX: reorder to chronological before delta inference
    raw_rows = _chronological_sort(raw_rows)

    transactions = []
    prev_balance = opening_balance

    for r in raw_rows:
        bal = parse_float(r.get("balance"))
        debit = credit = 0.0

        amount_tokens = r.get("amount_tokens") or []
        amt = parse_float(amount_tokens[-1]) if amount_tokens else 0.0

        if prev_balance is not None:
            delta = round(bal - prev_balance, 2)
            if delta > 0:
                credit = round(amt if amt > 0 else delta, 2)
            elif delta < 0:
                debit = round(amt if amt > 0 else -delta, 2)

        transactions.append({
            "date": r.get("date"),
            "description": r.get("description"),
            "debit": round(debit, 2),
            "credit": round(credit, 2),
            "balance": round(bal, 2),
            "page": r.get("page"),
            "source_file": source_filename,
            "bank": bank_name,
        })

        prev_balance = bal

    if closing_balance is not None:
        cb_date = latest_tx_date or (transactions[-1]["date"] if transactions else f"{detected_year}-01-01")
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
        })

    return _dedupe_transactions(transactions)


def parse_transactions_cimb(pdf, source_filename=""):
    """
    Main CIMB parser:
      - parse table rows if available
      - ensure full-doc footer scan for closing balance
      - FIX: reorder to chronological (statement is reverse chronological)
    """
    bank_name = "CIMB Bank"
    detected_year = None

    # Fast scan for branding + year
    for page in pdf.pages[:2]:
        text = page.extract_text() or ""
        if "CIMB ISLAMIC BANK" in text.upper():
            bank_name = "CIMB Islamic Bank"
        if not detected_year:
            detected_year = extract_year_from_text(text)

    if not detected_year:
        detected_year = str(datetime.now().year)

    # Full doc text for closing balance & totals (IMPORTANT)
    full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    closing_balance = extract_closing_balance_from_text(full_text)
    stmt_total_debit, stmt_total_credit = extract_statement_totals_from_text(full_text)

    # --- Table parse ---
    extracted = []
    idx = 0
    latest_tx_date = None

    for page_num, page in enumerate(pdf.pages, start=1):
        table = page.extract_table()
        if not table:
            continue

        for row in table:
            if not row or len(row) < 6:
                continue

            first_col = str(row[0]).lower() if row[0] else ""
            if "date" in first_col or "tarikh" in first_col:
                continue

            desc_text = str(row[1]).lower() if row[1] else ""
            if "opening balance" in desc_text:
                continue

            date_formatted = format_date(row[0], detected_year)
            if not date_formatted:
                continue

            debit_val = parse_float(row[3])   # Withdrawal
            credit_val = parse_float(row[4])  # Deposit

            # NOTE: Do NOT skip 0/0 blindly here; some banks spill rows, but for table mode
            # these are usually non-transaction/continuation. We keep the current behavior:
            if debit_val == 0.0 and credit_val == 0.0:
                continue

            bal = parse_float(row[5])
            if row[5] is None:
                continue

            if latest_tx_date is None or date_formatted > latest_tx_date:
                latest_tx_date = date_formatted

            idx += 1
            extracted.append({
                "date": date_formatted,
                "description": clean_text(row[1]),
                "ref_no": clean_text(row[2]),
                "debit": round(debit_val, 2),
                "credit": round(credit_val, 2),
                "balance": round(bal, 2),
                "page": page_num,
                "source_file": source_filename,
                "bank": bank_name,
                "__idx": idx,
            })

    extracted = _dedupe_transactions(extracted)

    # If no table rows, use text fallback (which now sorts chronologically before delta)
    if not extracted:
        return _parse_transactions_cimb_text(
            pdf,
            source_filename=source_filename,
            detected_year=detected_year,
            bank_name=bank_name,
            closing_balance=closing_balance,
        )

    # FIX: reorder table-extracted rows to chronological too
    extracted = _chronological_sort(extracted)

    # Append statement closing balance row (source-of-truth)
    if closing_balance is not None:
        cb_date = latest_tx_date or extracted[-1]["date"]
        extracted.append({
            "date": cb_date,
            "description": "CLOSING BALANCE / BAKI PENUTUP",
            "ref_no": "",
            "debit": 0.0,
            "credit": 0.0,
            "balance": round(float(closing_balance), 2),
            "page": None,
            "source_file": source_filename,
            "bank": bank_name,
            "is_statement_balance": True,
            "statement_total_debit": None if stmt_total_debit is None else round(float(stmt_total_debit), 2),
            "statement_total_credit": None if stmt_total_credit is None else round(float(stmt_total_credit), 2),
        })

    # Remove internal index field before returning
    for t in extracted:
        if "__idx" in t:
            del t["__idx"]

    return extracted
