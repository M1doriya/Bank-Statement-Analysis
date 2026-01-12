import re
from datetime import datetime

DATE_RE = re.compile(r"\d{1,2}/\d{2}/\d{2}")
AMOUNT_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}-?")
ZERO_RE = re.compile(r"^0?\.00-?$")


def extract_agrobank_summary_totals(pdf):
    total_debit = None
    total_credit = None

    for page in reversed(pdf.pages):
        text = page.extract_text() or ""

        for line in text.splitlines():
            u = line.upper()

            if "TOTAL DEBIT" in u:
                m = re.search(r"([\d,]+\.\d{2})", line)
                if m:
                    total_debit = float(m.group(1).replace(",", ""))

            if "TOTAL CREDIT" in u:
                m = re.search(r"([\d,]+\.\d{2})", line)
                if m:
                    total_credit = float(m.group(1).replace(",", ""))

        if total_debit is not None and total_credit is not None:
            break

    return total_debit, total_credit


def parse_agro_bank(pdf, source_file):
    """
    Agrobank parser
    - Opening / Closing balance extracted but NOT returned as transactions
    - Fully compatible with existing app.py
    """

    transactions = []
    previous_balance = None

    summary_debit, summary_credit = extract_agrobank_summary_totals(pdf)

    for page_num, page in enumerate(pdf.pages, start=1):

        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        words = sorted(words, key=lambda w: (w["top"], w["x0"]))

        i = 0
        while i < len(words):

            text = words[i]["text"]

            if DATE_RE.fullmatch(text):

                y_ref = words[i]["top"]
                same_line = [w for w in words if abs(w["top"] - y_ref) <= 2]

                description = " ".join(
                    w["text"] for w in same_line
                    if not DATE_RE.fullmatch(w["text"])
                    and not AMOUNT_RE.fullmatch(w["text"])
                    and not ZERO_RE.fullmatch(w["text"])
                ).strip()

                amounts = [
                    (w["x0"], w["text"])
                    for w in same_line
                    if AMOUNT_RE.fullmatch(w["text"])
                ]

                if not amounts:
                    i += 1
                    continue

                amounts.sort(key=lambda x: x[0])

                def to_float(v):
                    v = v.replace(",", "")
                    if v.endswith("-"):
                        return -float(v[:-1])
                    return float(v)

                balance = to_float(amounts[-1][1])
                iso_date = datetime.strptime(text, "%d/%m/%y").strftime("%Y-%m-%d")
                desc_upper = description.upper()

                # -------------------------------------------------
                # 🚫 HARD STOP: OPENING / CLOSING BALANCE
                # -------------------------------------------------
                if "BEGINNING BALANCE" in desc_upper:
                    previous_balance = balance
                    i += 1
                    continue

                if "CLOSING BALANCE" in desc_upper:
                    i += 1
                    continue

                # -------------------------------------------------
                # NORMAL TRANSACTION
                # -------------------------------------------------
                debit = credit = None

                if previous_balance is not None:
                    delta = balance - previous_balance
                    if delta > 0.0001:
                        credit = round(delta, 2)
                    elif delta < -0.0001:
                        debit = round(abs(delta), 2)

                transactions.append({
                    "date": iso_date,
                    "description": description,
                    "debit": debit,
                    "credit": credit,
                    "balance": round(balance, 2),
                    "page": page_num,
                    "bank": "Agrobank",
                    "source_file": source_file
                })

                previous_balance = balance

            i += 1

    # -------------------------------------------------
    # SUMMARY VALIDATION
    # -------------------------------------------------
    computed_debit = round(sum(t["debit"] or 0 for t in transactions), 2)
    computed_credit = round(sum(t["credit"] or 0 for t in transactions), 2)

    mismatch = False
    if summary_debit is not None and abs(computed_debit - summary_debit) > 0.01:
        mismatch = True
    if summary_credit is not None and abs(computed_credit - summary_credit) > 0.01:
        mismatch = True

    for t in transactions:
        t["summary_check"] = "#" if mismatch else ""

    return transactions
