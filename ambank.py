# ambank.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pdfplumber


# =========================================================
# Regex patterns
# =========================================================

# Transaction starts look like: 01May <rest...>
TX_START_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

# Money tokens
MONEY_TOKEN_RE = re.compile(r"^\d{1,3}(?:,\d{3})*\.\d{2}$|^\d+\.\d{2}$")

# Sometimes text extraction glues suffix: 10,400.52DR / 10,400.52CR
MONEY_WITH_SUFFIX_RE = re.compile(
    r"^(?P<num>\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2})(?P<suf>DR|CR)$",
    re.IGNORECASE,
)

# Negative formats sometimes appear as -(1,234.56) or (1,234.56)
PAREN_MONEY_RE = re.compile(r"^\((?P<num>\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2})\)$")
NEG_MONEY_RE = re.compile(r"^-(?P<num>\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2})$")

# Header / noise lines (keep this conservative)
NOISE_PREFIX_RE = re.compile(
    r"^(?:ACCOUNT\s+NO\.|STATEMENT\s+DATE|CURRENCY|PAGE|ACCOUNT\s+STATEMENT|PENYATA\s+AKAUN|"
    r"PROTECTED\s+BY\s+PIDM|DILINDUNGI\s+OLEH\s+PIDM|CATEGORY|KATEGORI|"
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


# =========================================================
# Public entry point (your Streamlit app calls this)
# =========================================================
def parse_ambank(pdf: pdfplumber.PDF, filename: str) -> List[Dict]:
    """
    Parse AmBank statement and extract all transactions.

    Output fields include:
      - date (YYYY-MM-DD)
      - description
      - debit, credit
      - balance
      - page
      - seq (stable ordering within the file)
      - bank, source_file
    """
    statement_info = extract_statement_info(pdf)
    detected_year = statement_info.get("year") or datetime.now().year

    transactions: List[Dict] = []
    seq = 0

    # Opening anchor (best-effort)
    prev_balance: Optional[float] = statement_info.get("opening_balance")

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text(x_tolerance=1) or ""
        if not text.strip():
            continue

        lines = _normalize_lines(text)
        page_txs, prev_balance, seq = _parse_transactions_from_lines(
            lines,
            page_num=page_num,
            filename=filename,
            detected_year=int(detected_year),
            prev_balance=prev_balance,
            seq_start=seq,
        )
        transactions.extend(page_txs)

    # Sort deterministically (date + seq)
    transactions = _sort_transactions(transactions)

    return transactions


# =========================================================
# Statement Info
# =========================================================
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

    m = re.search(r"ACCOUNT\s+NO\..*?:\s*(\d+)", first_page, re.IGNORECASE)
    if m:
        info["account_number"] = m.group(1)

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

    # Preferred opening balance source: Balance b/f line
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


# =========================================================
# Page / Line Parsing (multi-line transaction support)
# =========================================================
def _normalize_lines(text: str) -> List[str]:
    out: List[str] = []
    for raw in text.splitlines():
        ln = re.sub(r"\s+", " ", (raw or "")).strip()
        if ln:
            out.append(ln)
    return out


def _parse_transactions_from_lines(
    lines: List[str],
    *,
    page_num: int,
    filename: str,
    detected_year: int,
    prev_balance: Optional[float],
    seq_start: int,
) -> Tuple[List[Dict], Optional[float], int]:
    """
    Parse transactions by buffering lines from a date anchor until next date anchor.
    Returns txs, updated prev_balance, updated seq.
    """
    txs: List[Dict] = []

    current_date_iso: Optional[str] = None
    current_buf: List[str] = []

    seq = seq_start

    def flush():
        nonlocal prev_balance, current_date_iso, current_buf, seq

        if not current_date_iso:
            current_buf = []
            return

        if not current_buf:
            current_date_iso = None
            return

        tx, prev_balance = _finalize_tx(
            date_iso=current_date_iso,
            buf=current_buf,
            page_num=page_num,
            filename=filename,
            prev_balance=prev_balance,
            seq=seq,
        )
        if tx:
            txs.append(tx)
            seq += 1

        current_date_iso = None
        current_buf = []

    for ln in lines:
        # skip header-ish / summary-ish noise lines
        if NOISE_PREFIX_RE.match(ln):
            continue

        # ignore lone DR/CR tokens which can appear near summary extraction
        if ln.strip().upper() in {"DR", "CR"}:
            continue

        # allow mid-file opening anchor (Balance b/f)
        if BAL_BF_RE.search(ln):
            bal = _extract_last_balance_from_text(ln)
            if bal is not None:
                prev_balance = bal
            continue

        m = TX_START_RE.match(ln)
        if m:
            flush()

            dd = int(m.group("day"))
            mon = m.group("mon").upper()
            mm = _MONTH_MAP.get(mon)
            if not mm:
                continue

            current_date_iso = f"{detected_year:04d}-{mm}-{dd:02d}"

            rest = (m.group("rest") or "").strip()
            current_buf = [rest] if rest else []
            continue

        # continuation line
        if current_date_iso is not None:
            current_buf.append(ln)

    flush()
    return txs, prev_balance, seq


# =========================================================
# TX Finalization
# =========================================================
def _extract_last_balance_from_text(text: str) -> Optional[float]:
    """
    Return the right-most balance-like number from the text.

    IMPORTANT:
      - We DO NOT treat DR as negative.
      - Only treat negative if explicit '-' or parentheses format is present.
    """
    if not text:
        return None

    toks = text.split()
    for tok in reversed(toks):
        raw = tok.strip()

        # Handle glued suffix (10,400.52DR) => 10,400.52
        ms = MONEY_WITH_SUFFIX_RE.match(raw)
        if ms:
            raw = ms.group("num")

        # Explicit negative: -(1,234.56) OR -123.45
        mneg = NEG_MONEY_RE.match(raw)
        if mneg:
            try:
                return -float(mneg.group("num").replace(",", ""))
            except Exception:
                return None

        # Parentheses negative: (1,234.56)
        mpar = PAREN_MONEY_RE.match(raw)
        if mpar:
            try:
                return -float(mpar.group("num").replace(",", ""))
            except Exception:
                return None

        # Normal positive money token
        if MONEY_TOKEN_RE.match(raw):
            try:
                return float(raw.replace(",", ""))
            except Exception:
                return None

    return None


def _finalize_tx(
    *,
    date_iso: str,
    buf: List[str],
    page_num: int,
    filename: str,
    prev_balance: Optional[float],
    seq: int,
) -> Tuple[Optional[Dict], Optional[float]]:
    joined = " ".join([b for b in buf if b]).strip()
    if not joined:
        return None, prev_balance

    balance = _extract_last_balance_from_text(joined)
    if balance is None:
        # cannot infer debit/credit without balance (for this approach)
        return None, prev_balance

    # strip trailing balance chunk (and optional DR/CR) from description
    desc = re.sub(
        r"\s*(\(?-?\d{1,3}(?:,\d{3})*\.\d{2}\)?|-?\d+\.\d{2})\s*(?:DR|CR)?\s*$",
        "",
        joined,
        flags=re.IGNORECASE,
    ).strip()

    # clean spacing/comma artifacts
    desc = re.sub(r"\s+", " ", desc)
    desc = re.sub(r",\s*,+", ", ", desc)
    desc = re.sub(r",\s*$", "", desc).strip()

    debit = 0.0
    credit = 0.0
    if prev_balance is not None:
        delta = round(balance - prev_balance, 2)
        if delta > 0:
            credit = delta
        elif delta < 0:
            debit = abs(delta)

    tx: Dict = {
        "date": date_iso,
        "description": desc,
        "debit": round(float(debit), 2),
        "credit": round(float(credit), 2),
        "balance": round(float(balance), 2),
        "page": int(page_num),
        "seq": int(seq),  # critical for stable monthly ending_balance
        "bank": "AmBank Islamic",
        "source_file": filename,
    }

    return tx, balance


def _sort_transactions(transactions: List[Dict]) -> List[Dict]:
    # Deterministic ordering across pages and within same date
    return sorted(
        transactions,
        key=lambda t: (
            t.get("date") or "",
            int(t.get("page") or 0),
            int(t.get("seq") or 0),
        ),
    )


# =========================================================
# Monthly Summary Builder (use this instead of your current groupby logic)
# =========================================================
def build_monthly_summary(transactions: List[Dict]):
    """
    Build a stable, correct monthly summary.

    Fixes your screenshot issue:
      - Ensures ending_balance is truly the last transaction balance in that month
        by sorting by (month, date, page, seq) before groupby first/last.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required for build_monthly_summary()") from e

    if not transactions:
        return pd.DataFrame(
            columns=[
                "month",
                "transaction_count",
                "total_debit",
                "total_credit",
                "net_change",
                "opening_balance",
                "ending_balance",
                "lowest_balance",
                "highest_balance",
                "source_files",
            ]
        )

    df = pd.DataFrame(transactions).copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "balance"])

    df["month"] = df["date"].dt.to_period("M").astype(str)

    # CRITICAL: stable sort before groupby first/last
    sort_cols = ["month", "date", "page", "seq"]
    for c in sort_cols:
        if c not in df.columns:
            df[c] = 0
    df = df.sort_values(sort_cols, ascending=True)

    g = df.groupby("month", sort=False)

    summary = pd.DataFrame(
        {
            "transaction_count": g.size(),
            "total_debit": g["debit"].sum(),
            "total_credit": g["credit"].sum(),
            "opening_balance": g["balance"].first(),
            "ending_balance": g["balance"].last(),
            "lowest_balance": g["balance"].min(),
            "highest_balance": g["balance"].max(),
            "source_files": g["source_file"].agg(lambda x: ", ".join(sorted(set(x)))),
        }
    ).reset_index()

    summary["total_debit"] = summary["total_debit"].round(2)
    summary["total_credit"] = summary["total_credit"].round(2)
    summary["opening_balance"] = summary["opening_balance"].round(2)
    summary["ending_balance"] = summary["ending_balance"].round(2)
    summary["lowest_balance"] = summary["lowest_balance"].round(2)
    summary["highest_balance"] = summary["highest_balance"].round(2)

    summary["net_change"] = (summary["ending_balance"] - summary["opening_balance"]).round(2)

    # Match your UI column ordering
    summary = summary[
        [
            "month",
            "transaction_count",
            "total_debit",
            "total_credit",
            "net_change",
            "ending_balance",
            "lowest_balance",
            "highest_balance",
            "source_files",
        ]
    ]

    return summary
