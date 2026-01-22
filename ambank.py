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
MONEY_ANYWHERE_RE = re.compile(r"(\(?-?\d{1,3}(?:,\d{3})*\.\d{2}\)?|-?\d+\.\d{2})")

# Balance b/f line
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
# Helpers
# =========================================================
def _safe_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    s = s.replace(",", "")
    s = s.strip("()")
    try:
        return float(s)
    except Exception:
        return None


def _normalize_lines(text: str) -> List[str]:
    lines = []
    for raw in (text or "").splitlines():
        l = re.sub(r"\s+", " ", raw).strip()
        if l:
            lines.append(l)
    return lines


def _to_iso_date(day: str, mon: str, year: int) -> Optional[str]:
    mon_u = (mon or "").upper()
    mm = _MONTH_MAP.get(mon_u)
    if not mm:
        return None
    dd = f"{int(day):02d}"
    return f"{year:04d}-{mm}-{dd}"


def _extract_last_balance_from_text(s: str) -> Optional[float]:
    """
    Extract the last money-like token in the line buffer, interpreted as 'balance'.
    """
    if not s:
        return None

    # Prefer last explicit money token with commas/decimals
    candidates = MONEY_ANYWHERE_RE.findall(s)
    if not candidates:
        return None

    # Take the last candidate as balance
    raw = candidates[-1].strip()
    raw = raw.replace(",", "")
    raw = raw.strip()
    # parentheses handling
    if raw.startswith("(") and raw.endswith(")"):
        raw = "-" + raw[1:-1]

    try:
        return float(raw)
    except Exception:
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
        "seq": int(seq),  # IMPORTANT: stable ordering within the file
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

    # Fallback: find first Balance b/f anywhere
    bf = BAL_BF_RE.search(first_page)
    if bf:
        # try to find a number after it
        tail = first_page[bf.end() : bf.end() + 120]
        nums = MONEY_ANYWHERE_RE.findall(tail)
        if nums:
            v = _safe_float(nums[0])
            if v is not None:
                info["opening_balance"] = v

    return info


# =========================================================
# Transaction Parsing
# =========================================================
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
    Parse transactions from normalized text lines.
    Uses a small buffer to combine multi-line descriptions until next date.
    """
    txs: List[Dict] = []
    buf: List[str] = []
    cur_date_iso: Optional[str] = None
    seq = seq_start

    def flush_current():
        nonlocal prev_balance, seq, buf, cur_date_iso
        if cur_date_iso is None:
            buf = []
            return
        tx, prev_balance2 = -1, prev_balance  # placeholder
        tx, prev_balance2 = _finalize_tx(
            date_iso=cur_date_iso,
            buf=buf,
            page_num=page_num,
            filename=filename,
            prev_balance=prev_balance,
            seq=seq,
        )
        if tx:
            txs.append(tx)
            prev_balance = prev_balance2
            seq += 1
        buf = []

    for line in lines:
        m = TX_START_RE.match(line)
        if m:
            # new tx starts
            flush_current()

            day = m.group("day")
            mon = m.group("mon")
            rest = (m.group("rest") or "").strip()
            cur_date_iso = _to_iso_date(day, mon, detected_year)
            if cur_date_iso is None:
                # invalid date, ignore
                cur_date_iso = None
                buf = []
                continue

            buf = [rest] if rest else []
        else:
            # continuation line
            if cur_date_iso is not None:
                buf.append(line)

    # flush last
    flush_current()

    return txs, prev_balance, seq


# =========================================================
# Public entry point (Streamlit app calls this)
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

    # Sort deterministically (date + page + seq)
    transactions = _sort_transactions(transactions)
    return transactions


# =========================================================
# Monthly Summary Builder (optional utility)
# =========================================================
def build_monthly_summary(transactions: List[Dict]):
    """
    Stable monthly summary:
    - Sorts by date, page, seq (or row order fallback)
    - Ending balance = last row after stable ordering
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
                "ending_balance",
                "lowest_balance",
                "highest_balance",
                "source_files",
            ]
        )

    df = pd.DataFrame(transactions).copy()
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_parsed"]).copy()

    df["debit"] = pd.to_numeric(df.get("debit", 0), errors="coerce").fillna(0.0)
    df["credit"] = pd.to_numeric(df.get("credit", 0), errors="coerce").fillna(0.0)
    df["balance"] = pd.to_numeric(df.get("balance", None), errors="coerce")
    df["page"] = pd.to_numeric(df.get("page", 0), errors="coerce").fillna(0).astype(int)

    if "seq" in df.columns:
        df["seq"] = pd.to_numeric(df["seq"], errors="coerce").fillna(0).astype(int)
    else:
        df["seq"] = range(len(df))

    df["month"] = df["date_parsed"].dt.strftime("%Y-%m")

    rows = []
    for month, g in df.groupby("month", sort=True):
        g = g.sort_values(["date_parsed", "page", "seq"], ascending=True)
        balances = g["balance"].dropna()

        ending_balance = round(float(balances.iloc[-1]), 2) if not balances.empty else None
        lowest_balance = round(float(balances.min()), 2) if not balances.empty else None
        highest_balance = round(float(balances.max()), 2) if not balances.empty else None

        rows.append(
            {
                "month": month,
                "transaction_count": int(len(g)),
                "total_debit": round(float(g["debit"].sum()), 2),
                "total_credit": round(float(g["credit"].sum()), 2),
                "net_change": round(float(g["credit"].sum() - g["debit"].sum()), 2),
                "ending_balance": ending_balance,
                "lowest_balance": lowest_balance,
                "highest_balance": highest_balance,
                "source_files": ", ".join(sorted(set(g.get("source_file", [])))),
            }
        )

    return pd.DataFrame(rows)
