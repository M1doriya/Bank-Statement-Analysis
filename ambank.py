from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pdfplumber

# =========================================================
# AmBank Islamic (Malaysia) - Robust text parser
# =========================================================
# Key fixes vs your current version:
# - MULTI-LINE ROWS: the statement often prints BALANCE on the first line
#   and the DEBIT/CREDIT amount on the next line. We therefore parse by
#   accumulating lines until the next date anchor.
# - NO FAKE OD: the token "DR" can appear in the *summary* section (and
#   sometimes gets extracted as a stray token). For the provided AmBank PDFs
#   it MUST NOT be interpreted as "negative overdraft".
# - OPENING BALANCE: we anchor running delta using "Balance b/f" if present;
#   otherwise fall back to the "OPENING BALANCE / BAKI PEMBUKAAN" label.


# -----------------------------
# Regex patterns
# -----------------------------
TX_START_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

# money token without sign (AmBank tables are positive; any sign is handled separately)
MONEY_TOKEN_RE = re.compile(r"^\d{1,3}(?:,\d{3})*\.\d{2}$|^\d+\.\d{2}$")

# sometimes a suffix is attached to the number (e.g. 10,400.52DR in summary)
MONEY_WITH_SUFFIX_RE = re.compile(
    r"^(?P<num>\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2})(?P<suf>DR|CR)$",
    re.IGNORECASE,
)

# Header / noise lines
NOISE_PREFIX_RE = re.compile(
    r"^(?:ACCOUNT\s+NO\.|STATEMENT\s+DATE|CURRENCY|PAGE|ACCOUNT\s+STATEMENT|PENYATA\s+AKAUN|"
    r"PROTECTED\s+BY\s+PIDM|DILINDUNGI\s+OLEH\s+PIDM|CATEGORY|KATEGORI|BALANCE|BAKI|"
    r"NO\.\s*OF\s*TRANSACTION|BILANGAN\s+TRANSAKSI|ACCOUNT\s+SUMMARY|RINGKASAN\s+AKAUN|"
    r"OPENING\s+BALANCE|TOTAL\s+DEBITS?|TOTAL\s+CREDITS?|CLOSING\s+BALANCE|CHEQUES\s+NOT\s+CLEARED|"
    r"DATE\s*$|TARIKH\s*$|CHEQUE\s+NO\.|NO\.\s+CEK|TRANSACTION\s*$|TRANSAKSI\s*$|DEBIT\s*$|CREDIT\s*$|BALANCE\s*$)\b",
    re.IGNORECASE,
)

BAL_BF_RE = re.compile(r"\bBaki\s+Bawa\s+Ke\s+Hadapan\b|\bBalance\s+b/f\b", re.IGNORECASE)


_MONTH_MAP = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}


def parse_ambank(pdf: pdfplumber.PDF, filename: str) -> List[Dict]:
    """Entry point used by app.py (expects a pdfplumber.PDF object)."""

    statement_info = extract_statement_info(pdf)
    detected_year = statement_info.get("year") or datetime.now().year

    transactions: List[Dict] = []

    # running balance anchor
    prev_balance: Optional[float] = statement_info.get("opening_balance")

    # We parse per page, but preserve natural order (page order) then sort.
    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text(x_tolerance=1) or ""
        if not text.strip():
            continue

        page_lines = _normalize_lines(text)
        page_txs, prev_balance = _parse_transactions_from_lines(
            page_lines,
            page_num=page_num,
            filename=filename,
            detected_year=detected_year,
            prev_balance=prev_balance,
        )
        transactions.extend(page_txs)

    # Final sort (chronological, stable by page order)
    transactions = _sort_transactions(transactions)

    return transactions


# ---------------------------------------------------------------------
# STATEMENT INFO
# ---------------------------------------------------------------------

def extract_statement_info(pdf: pdfplumber.PDF) -> Dict:
    """Extract account number, statement period, and opening balance."""

    info: Dict = {
        "account_number": None,
        "statement_period": None,
        "currency": "MYR",
        "opening_balance": None,
        "year": None,
    }

    first_page = pdf.pages[0].extract_text(x_tolerance=1) or ""

    # Account number
    m = re.search(r"ACCOUNT\s+NO\..*?:\s*(\d+)", first_page, re.IGNORECASE)
    if m:
        info["account_number"] = m.group(1)

    # Statement period
    m = re.search(
        r"STATEMENT\s+DATE.*?:\s*(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})",
        first_page,
        re.IGNORECASE,
    )
    if m:
        info["statement_period"] = f"{m.group(1)} to {m.group(2)}"
        try:
            info["year"] = int(m.group(2)[-4:])
        except Exception:
            pass

    # Preferred opening balance source in these PDFs is the "Balance b/f" line
    m = re.search(
        r"Baki\s+Bawa\s+Ke\s+Hadapan\s*/\s*Balance\s+b/f\s+(?P<bal>[\d,]+\.\d{2})",
        first_page,
        re.IGNORECASE,
    )
    if m:
        info["opening_balance"] = float(m.group("bal").replace(",", ""))
        return info

    # Fallback opening balance label
    m = re.search(
        r"OPENING\s+BALANCE\s*/\s*BAKI\s+PEMBUKAAN\s+(?P<bal>[\d,]+\.\d{2})",
        first_page,
        re.IGNORECASE,
    )
    if m:
        info["opening_balance"] = float(m.group("bal").replace(",", ""))

    return info


# ---------------------------------------------------------------------
# PAGE / LINE PARSING
# ---------------------------------------------------------------------

def _normalize_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw in text.splitlines():
        ln = re.sub(r"\s+", " ", (raw or "")).strip()
        if not ln:
            continue
        lines.append(ln)
    return lines


def _parse_transactions_from_lines(
    lines: List[str],
    *,
    page_num: int,
    filename: str,
    detected_year: int,
    prev_balance: Optional[float],
) -> Tuple[List[Dict], Optional[float]]:
    txs: List[Dict] = []

    current_date_iso: Optional[str] = None
    current_buf: List[str] = []

    def flush():
        nonlocal prev_balance, current_date_iso, current_buf
        if not current_date_iso or not current_buf:
            current_date_iso, current_buf = None, []
            return

        tx, prev_balance = _finalize_tx(
            current_date_iso=current_date_iso,
            buf=current_buf,
            page_num=page_num,
            filename=filename,
            prev_balance=prev_balance,
        )
        if tx:
            txs.append(tx)
        current_date_iso, current_buf = None, []

    for ln in lines:
        # Skip clear header/noise lines
        if NOISE_PREFIX_RE.match(ln):
            continue

        # Some PDFs emit a lone 'DR' token near the summary section. Ignore.
        if ln.strip().upper() in {"DR", "CR"}:
            continue

        # Anchor opening balance from Balance b/f line if it appears mid-file
        if BAL_BF_RE.search(ln):
            # extract last numeric token as balance
            bal = _extract_last_balance_from_text(ln)
            if bal is not None:
                prev_balance = bal
            continue

        m = TX_START_RE.match(ln)
        if m:
            # new tx starts -> flush previous
            flush()

            dd = int(m.group("day"))
            mon = m.group("mon").upper()
            mm = _MONTH_MAP.get(mon)
            if not mm:
                continue

            current_date_iso = f"{detected_year:04d}-{mm}-{dd:02d}"
            rest = (m.group("rest") or "").strip()
            if rest:
                current_buf = [rest]
            else:
                current_buf = []
            continue

        # Continuation line for current tx
        if current_date_iso is not None:
            current_buf.append(ln)

    # flush last tx on page
    flush()

    return txs, prev_balance


# ---------------------------------------------------------------------
# TX FINALIZATION
# ---------------------------------------------------------------------

def _extract_last_balance_from_text(text: str) -> Optional[float]:
    """Return the right-most money token on the text, ignoring DR/CR suffix."""
    if not text:
        return None

    toks = text.split()
    for tok in reversed(toks):
        t = tok.strip().replace(",", "")

        m = MONEY_WITH_SUFFIX_RE.match(tok)
        if m:
            t = m.group("num").replace(",", "")

        if MONEY_TOKEN_RE.match(tok) or m:
            try:
                return float(t)
            except Exception:
                return None
    return None


def _finalize_tx(
    *,
    current_date_iso: str,
    buf: List[str],
    page_num: int,
    filename: str,
    prev_balance: Optional[float],
) -> Tuple[Optional[Dict], Optional[float]]:
    """Build a transaction dict from buffered lines."""

    joined = " ".join([b for b in buf if b]).strip()
    if not joined:
        return None, prev_balance

    balance = _extract_last_balance_from_text(joined)
    if balance is None:
        # if we cannot find a balance, we cannot infer reliably
        return None, prev_balance

    # description is joined text with the *last* balance token stripped
    desc = joined
    # strip a trailing "<amount>[DR|CR]?" at end
    desc = re.sub(
        r"\s*(\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2})\s*(?:DR|CR)?\s*$",
        "",
        desc,
        flags=re.IGNORECASE,
    ).strip()

    # clean repeated commas/spaces from pdf extraction
    desc = re.sub(r"\s+", " ", desc)
    desc = re.sub(r",\s*,+", ", ", desc)
    desc = re.sub(r",\s*$", "", desc).strip()

    # infer debit/credit from balance delta
    debit = 0.0
    credit = 0.0

    if prev_balance is not None:
        delta = round(balance - prev_balance, 2)
        if delta > 0:
            credit = delta
        elif delta < 0:
            debit = abs(delta)

    tx: Dict = {
        "date": current_date_iso,
        "description": desc,
        "debit": round(float(debit), 2),
        "credit": round(float(credit), 2),
        "balance": round(float(balance), 2),
        "page": int(page_num),
        "bank": "AmBank Islamic",
        "source_file": filename,
    }

    return tx, balance


# ---------------------------------------------------------------------
# Sorting helper
# ---------------------------------------------------------------------

def _sort_transactions(transactions: List[Dict]) -> List[Dict]:
    def _key(t: Dict):
        # ISO date -> safe lexical sort; tie-break by page
        return (t.get("date") or "", int(t.get("page") or 0))

    return sorted(transactions, key=_key)
