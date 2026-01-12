# affin_bank.py
import re
import pdfplumber

def _clean_amount(x: str):
    """Convert common money strings to float-like string, or None."""
    if x is None:
        return None
    x = x.strip()
    if not x:
        return None

    x = x.replace(",", "").replace(" ", "")

    if x.startswith("(") and x.endswith(")"):
        x = "-" + x[1:-1]

    if not re.fullmatch(r"-?\d+(\.\d{1,2})?", x):
        return None

    return x


def parse_affin_bank(pdf_input, source_file: str = ""):
    """
    Dummy Affin Bank parser.

    Accepts either:
      - a pdfplumber.PDF object, OR
      - a file-like object / streamlit UploadedFile

    Returns list[dict] with keys:
      date, description, debit, credit, balance, page, bank, source_file
    """

    bank_name = "Affin Bank"
    tx = []

    row_re = re.compile(
        r"(?P<date>\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\s+"
        r"(?P<desc>.+?)\s+"
        r"(?P<debit>\(?[\d,]+\.\d{2}\)?)?\s*"
        r"(?P<credit>\(?[\d,]+\.\d{2}\)?)?\s*"
        r"(?P<balance>\(?[\d,]+\.\d{2}\)?)\s*$"
    )

    def parse_pdf(pdf):
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for ln in lines:
                m = row_re.match(ln)
                if not m:
                    continue

                date = m.group("date")
                desc = (m.group("desc") or "").strip()

                debit = _clean_amount(m.group("debit"))
                credit = _clean_amount(m.group("credit"))
                balance = _clean_amount(m.group("balance"))

                if (debit is None or debit == "") and (credit is None or credit == ""):
                    continue

                tx.append({
                    "date": date,
                    "description": desc,
                    "debit": float(debit) if debit is not None else 0.0,
                    "credit": float(credit) if credit is not None else 0.0,
                    "balance": float(balance) if balance is not None else None,
                    "page": page_idx,
                    "bank": bank_name,
                    "source_file": source_file or ""
                })

    if hasattr(pdf_input, "pages"):
        parse_pdf(pdf_input)
        return tx

    try:
        pdf_input.seek(0)
    except Exception:
        pass

    try:
        with pdfplumber.open(pdf_input) as pdf:
            parse_pdf(pdf)
    except Exception as e:
        # Do not swallow errors: surface them to the caller for easier debugging.
        raise RuntimeError(f"Affin Bank parser failed for '{source_file}': {e}") from e

    return tx
