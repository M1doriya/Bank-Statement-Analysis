import re
import fitz
from datetime import datetime

from core_utils import read_pdf_bytes


def parse_transactions_maybank(pdf_input, source_filename):
    # ---------------- OPEN PDF (Streamlit-safe) ----------------
    def open_doc(inp):
        # Preferred: bytes, UploadedFile, file-like, or path
        # Note: pdfplumber.PDF has .stream; keep that supported as well.
        if isinstance(inp, (bytes, bytearray)):
            return fitz.open(stream=bytes(inp), filetype="pdf")

        if hasattr(inp, "stream"):
            try:
                inp.stream.seek(0)
            except Exception:
                pass
            data = inp.stream.read()
            return fitz.open(stream=data, filetype="pdf")

        # Streamlit UploadedFile / file-like / path
        data = read_pdf_bytes(inp)
        return fitz.open(stream=data, filetype="pdf")

    doc = open_doc(pdf_input)

    # ---------------- BANK NAME / YEAR ----------------
    bank_name = "Maybank"
    statement_year = None

    STATEMENT_DATE_RE = re.compile(r"STATEMENT\s+DATE\s*:?\s*(\d{2})/(\d{2})/(\d{2})")

    for p in range(min(2, len(doc))):
        txt = doc[p].get_text("text").upper()

        if "MAYBANK ISLAMIC" in txt:
            bank_name = "Maybank Islamic"
        elif "MAYBANK" in txt:
            bank_name = "Maybank"

        m = STATEMENT_DATE_RE.search(txt)
        if m:
            statement_year = f"20{int(m.group(3)):02d}"
            break

    if not statement_year:
        statement_year = str(datetime.now().year)

    FOOTER_KEYWORDS = [
        "ENDING BALANCE",
        "LEDGER BALANCE",
        "TOTAL DEBITS",
        "TOTAL CREDITS",
        "END OF STATEMENT",
        "CHEQUES",
        "OVERDRAWN",
    ]

    # =========================================================
    # PARSER A: CLASSIC (NOW MULTI-LINE DESCRIPTION, NO X HARDCODE)
    # =========================================================
    DATE_RE_A_TOKEN = re.compile(
        r"^("
        r"\d{2}/\d{2}/\d{4}|"
        r"\d{2}/\d{2}|"
        r"\d{2}-\d{2}|"
        r"\d{2}\s+[A-Z]{3}"
        r")$",
        re.IGNORECASE,
    )
    AMOUNT_RE_A = re.compile(r"^(?:\d{1,3}(?:,\d{3})*|\d+)?\.\d{2}[+-]?$")

    def norm_date_a(token, year):
        token = token.strip().upper()
        for fmt in ("%d/%m/%Y", "%d/%m", "%d-%m", "%d %b"):
            try:
                if fmt == "%d/%m/%Y":
                    dt = datetime.strptime(token, fmt)
                else:
                    dt = datetime.strptime(f"{token}/{year}", fmt + "/%Y")
                return dt.strftime("%Y-%m-%d")
            except:
                pass
        return None

    def parse_amt_a(t):
        t = t.strip()
        sign = "+" if t.endswith("+") else "-" if t.endswith("-") else None
        v = float(t.replace(",", "").rstrip("+-"))
        return v, sign

    def _group_lines(words):
        rows = [
            {"x0": w[0], "y0": round(w[1], 1), "text": str(w[4]).strip()}
            for w in words
            if str(w[4]).strip()
        ]
        lines = {}
        for r in rows:
            lines.setdefault(r["y0"], []).append(r)
        for y in lines:
            lines[y].sort(key=lambda r: r["x0"])
        return rows, lines

    def _extract_desc_map_classic(page):
        """
        Continuation logic:
        - find date token by regex in first few tokens of the line
        - description = non-money tokens between date and first money column
        - continuation lines append non-money tokens in same region
        Keyed by y0 of the transaction-start line.
        """
        words = page.get_text("words")
        _, lines = _group_lines(words)

        ys = sorted(lines.keys())
        desc_by_y = {}

        current_y = None
        current_desc = []
        desc_left = None
        money_left = None

        for y in ys:
            items = lines[y]

            line_text = " ".join(it["text"] for it in items).upper()
            if any(k in line_text for k in FOOTER_KEYWORDS):
                break

            date_idx = None
            for idx, it in enumerate(items[:6]):
                if DATE_RE_A_TOKEN.match(it["text"]):
                    date_idx = idx
                    break

            money_positions = [it["x0"] for it in items if AMOUNT_RE_A.match(it["text"])]
            this_money_left = min(money_positions) if money_positions else None

            if date_idx is not None:
                if current_y is not None:
                    desc_by_y[current_y] = " ".join(current_desc).strip()

                current_y = y
                current_desc = []

                dt = items[date_idx]
                desc_left = dt["x0"] + 20
                money_left = this_money_left

                for it in items[date_idx + 1 :]:
                    if AMOUNT_RE_A.match(it["text"]):
                        break
                    current_desc.append(it["text"])

            else:
                if current_y is None:
                    continue

                for it in items:
                    if AMOUNT_RE_A.match(it["text"]):
                        continue
                    if desc_left is not None and it["x0"] < desc_left:
                        continue
                    if money_left is not None and it["x0"] >= money_left:
                        continue
                    current_desc.append(it["text"])

        if current_y is not None:
            desc_by_y[current_y] = " ".join(current_desc).strip()

        for k in list(desc_by_y.keys()):
            desc_by_y[k] = " ".join(desc_by_y[k].split())

        return desc_by_y

    def parse_classic():
        transactions = []
        previous_balance = None

        for page_index, page in enumerate(doc):
            words = page.get_text("words")
            rows, _ = _group_lines(words)
            desc_by_y = _extract_desc_map_classic(page)

            rows.sort(key=lambda r: (r["y0"], r["x0"]))
            used_y = set()

            for r in rows:
                token = r["text"]
                if not DATE_RE_A_TOKEN.match(token):
                    continue

                y = r["y0"]
                if y in used_y:
                    continue

                line = [w for w in rows if abs(w["y0"] - y) <= 3.0]
                line.sort(key=lambda w: w["x0"])

                date_iso = norm_date_a(token, statement_year)
                if not date_iso:
                    continue

                amounts = []
                for w in line:
                    clean_text = w["text"].replace(",", "").rstrip("+-")
                    try:
                        float(clean_text)
                        if "/" not in w["text"]:
                            amounts.append((w["x0"], w["text"]))
                    except:
                        continue

                if not amounts:
                    continue

                amounts.sort(key=lambda a: a[0])
                balance_val = float(amounts[-1][1].replace(",", "").rstrip("+-"))

                line_text = " ".join([w["text"] for w in line]).upper()
                if "BEGINNING BALANCE" in line_text and len(amounts) >= 2:
                    previous_balance = float(amounts[0][1].replace(",", ""))
                    txn_val = float(amounts[1][1].replace(",", "").rstrip("+-"))
                elif len(amounts) > 1:
                    txn_val = float(amounts[-2][1].replace(",", "").rstrip("+-"))
                else:
                    txn_val = 0.0

                description = desc_by_y.get(y, "").strip()
                debit = credit = 0.0

                if previous_balance is not None:
                    delta = round(balance_val - previous_balance, 2)
                    if delta > 0:
                        credit = abs(delta)
                    elif delta < 0:
                        debit = abs(delta)
                else:
                    debit = txn_val

                used_y.add(y)
                transactions.append(
                    {
                        "date": date_iso,
                        "description": description,
                        "debit": round(debit, 2),
                        "credit": round(credit, 2),
                        "balance": round(balance_val, 2),
                        "page": page_index + 1,
                        "bank": bank_name,
                        "source_file": source_filename,
                    }
                )

                previous_balance = balance_val

        return transactions

    # =========================================================
    # PARSER B: ISLAMIC SPLIT-DATE
    # =========================================================
    MONTHS = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}

    def is_day(t): return t.isdigit() and 1 <= int(t) <= 31
    def is_month(t): return t.capitalize() in MONTHS
    def is_year(t): return t.isdigit() and t.startswith("20")

    def parse_amount(v): return float(v.replace(",", ""))

    def looks_like_money(t):
        tt = t.replace(",", "")
        if "." not in tt:
            return False
        try:
            float(tt)
            return True
        except:
            return False

    def parse_split_date_islamic():
        transactions = []
        previous_balance = None

        for page_index, page in enumerate(doc):
            words = page.get_text("words")
            rows = [
                {"x": w[0], "y": round(w[1], 1), "text": str(w[4]).strip()}
                for w in words
                if str(w[4]).strip()
            ]
            rows.sort(key=lambda r: (r["y"], r["x"]))
            used_y = set()

            for i in range(len(rows) - 2):
                w1, w2, w3 = rows[i], rows[i + 1], rows[i + 2]
                if not (is_day(w1["text"]) and is_month(w2["text"]) and is_year(w3["text"])):
                    continue

                y_key = w1["y"]
                if y_key in used_y:
                    continue

                try:
                    date_iso = datetime.strptime(
                        f"{w1['text']} {w2['text']} {w3['text']}", "%d %b %Y"
                    ).strftime("%Y-%m-%d")
                except:
                    continue

                line = [w for w in rows if abs(w["y"] - y_key) <= 1.5]
                line.sort(key=lambda w: w["x"])

                desc_parts, amounts = [], []
                for w in line:
                    if w is w1 or w is w2 or w is w3:
                        continue
                    if looks_like_money(w["text"]):
                        amounts.append(w["text"])
                    else:
                        desc_parts.append(w["text"])

                if not amounts:
                    continue

                balance = parse_amount(amounts[-1])
                debit = credit = 0.0

                if previous_balance is not None:
                    delta = round(balance - previous_balance, 2)
                    if delta < 0:
                        debit = abs(delta)
                    elif delta > 0:
                        credit = delta
                else:
                    if len(amounts) >= 2:
                        txn_amt = parse_amount(amounts[-2])
                        desc_up = " ".join(desc_parts).upper()
                        if ("CR" in desc_up) or ("CREDIT" in desc_up):
                            credit = txn_amt
                        else:
                            debit = txn_amt

                transactions.append(
                    {
                        "date": date_iso,
                        "description": " ".join(desc_parts).strip(),
                        "debit": round(debit, 2),
                        "credit": round(credit, 2),
                        "balance": round(balance, 2),
                        "page": page_index + 1,
                        "bank": bank_name,
                        "source_file": source_filename,
                    }
                )

                previous_balance = balance
                used_y.add(y_key)

        return transactions

    # ---------------- RUN BOTH + CHOOSE / MERGE ----------------
    tx_a = parse_classic()
    tx_b = parse_split_date_islamic()

    tx = tx_a if len(tx_a) >= len(tx_b) else tx_b

    if tx_a and tx_b:
        seen = set()
        merged = []
        for t in (tx_a + tx_b):
            key = (
                t["date"],
                t["description"],
                t["debit"],
                t["credit"],
                t["balance"],
                t["page"],
                t["source_file"],
            )
            if key not in seen:
                seen.add(key)
                merged.append(t)
        tx = merged

    doc.close()
    return tx
