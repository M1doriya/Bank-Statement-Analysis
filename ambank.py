# ambank.py
from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pdfplumber


# =========================================================
# Regex patterns
# =========================================================

TX_START_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s*(?P<rest>.*)$",
    re.IGNORECASE,
)

MONEY_TOKEN_RE = re.compile(r"^\d{1,3}(?:,\d{3})*\.\d{2}$|^\d+\.\d{2}$")


# =========================================================
# Helpers
# =========================================================

def _safe_money_to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if not re.match(r"^\d{1,3}(?:,\d{3})*\.\d{2}$|^\d+\.\d{2}$", s):
        return None
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def _money_tokens_from_text(text: str) -> List[float]:
    # Extract all "1,234.56" style tokens
    toks = re.findall(r"\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2}", text or "")
    out: List[float] = []
    for t in toks:
        v = _safe_money_to_float(t)
        if v is not None:
            out.append(v)
    return out


def _extract_last_balance_from_text(joined: str) -> Optional[float]:
    """
    Find the last money token in the buffered transaction text.
    Supports optional trailing DR/CR (we ignore label here; sign-handling is done elsewhere if needed).
    """
    if not joined:
        return None

    # last money-ish token
    m = re.search(r"(\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2})\s*(?:DR|CR)?\s*$", joined, re.I)
    if not m:
        return None

    raw = m.group(1)
    try:
        return float(raw.replace(",", ""))
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
        "seq": int(seq),  # stable within parsing
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
# Statement Info + Totals
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


def extract_ambank_statement_totals(pdf: pdfplumber.PDF, source_file: str = "") -> Dict[str, Optional[float]]:
    """
    Extract printed totals from the Account Summary section (source-of-truth):
      - opening_balance
      - total_debit (AMOUNT, not count)
      - total_credit (AMOUNT, not count)
      - ending_balance (closing)
    Works even when the amounts appear as separate lines above the labels (common in AmBank PDFs).
    """
    out: Dict[str, Optional[float]] = {
        "opening_balance": None,
        "total_debit": None,
        "total_credit": None,
        "ending_balance": None,
        "statement_month": None,   # YYYY-MM derived from period end date
        "source_file": source_file,
    }

    if not pdf.pages:
        return out

    first_page = pdf.pages[0].extract_text(x_tolerance=1) or ""

    # statement month from "01/04/2024 - 30/04/2024"
    m = re.search(
        r"STATEMENT\s+DATE.*?:\s*(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})",
        first_page,
        re.IGNORECASE,
    )
    if m:
        try:
            end_dt = datetime.strptime(m.group(2), "%d/%m/%Y")
            out["statement_month"] = end_dt.strftime("%Y-%m")
        except Exception:
            pass

    # opening balance (usually reliable)
    m = re.search(r"OPENING\s+BALANCE\s*/\s*BAKI\s+PEMBUKAAN\s+(?P<bal>[\d,]+\.\d{2})", first_page, re.I)
    if m:
        out["opening_balance"] = float(m.group("bal").replace(",", ""))

    # closing balance sometimes printed without amount on same line in extracted text,
    # so we primarily use the numeric block approach below.
    m = re.search(r"CLOSING\s+BALANCE\s*/\s*BAKI\s+PENUTUPAN\s+(?P<bal>[\d,]+\.\d{2})", first_page, re.I)
    if m:
        out["ending_balance"] = float(m.group("bal").replace(",", ""))

    # ---- Key logic: amounts in Account Summary block ----
    # In many PDFs the amounts appear as 4 separate lines:
    #   0.00
    #   <closing>
    #   <total credits>
    #   <total debits>
    # followed by the labeled rows (with counts).
    #
    # We capture the last 4 money tokens that occur BEFORE the opening balance token.
    all_amounts = _money_tokens_from_text(first_page)
    if out["opening_balance"] is not None and all_amounts:
        # find the last occurrence of opening balance in the stream (float compare with tolerance)
        ob = float(out["opening_balance"])
        idx_ob = None
        for i in range(len(all_amounts) - 1, -1, -1):
            if abs(all_amounts[i] - ob) < 0.005:
                idx_ob = i
                break

        if idx_ob is not None:
            prev = all_amounts[:idx_ob]  # amounts before opening balance
            if len(prev) >= 4:
                block4 = prev[-4:]
                # expected order: [cheques_not_cleared, closing, total_credits, total_debits]
                out["ending_balance"] = out["ending_balance"] or block4[1]
                out["total_credit"] = out["total_credit"] or block4[2]
                out["total_debit"] = out["total_debit"] or block4[3]

    # last fallback: if we still don't have totals, try a more direct search on the whole doc text
    if out["total_debit"] is None or out["total_credit"] is None or out["ending_balance"] is None:
        full_text = ""
        for p in pdf.pages[:2]:  # totals are always on page 1 for this format
            full_text += (p.extract_text(x_tolerance=1) or "") + "\n"
        # try to re-run the same block logic against first 2 pages text
        all_amounts2 = _money_tokens_from_text(full_text)
        if out["opening_balance"] is not None and all_amounts2:
            ob = float(out["opening_balance"])
            idx_ob = None
            for i in range(len(all_amounts2) - 1, -1, -1):
                if abs(all_amounts2[i] - ob) < 0.005:
                    idx_ob = i
                    break
            if idx_ob is not None:
                prev = all_amounts2[:idx_ob]
                if len(prev) >= 4:
                    block4 = prev[-4:]
                    out["ending_balance"] = out["ending_balance"] or block4[1]
                    out["total_credit"] = out["total_credit"] or block4[2]
                    out["total_debit"] = out["total_debit"] or block4[3]

    return out


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

        tx, prev_balance_new = _finalize_tx(
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
            prev_balance = prev_balance_new
        else:
            # keep prev_balance unchanged
            pass

        current_date_iso = None
        current_buf = []

    for ln in lines:
        m = TX_START_RE.match(ln)
        if m:
            # new tx anchor -> flush previous
            flush()

            day = int(m.group("day"))
            mon = m.group("mon").title()
            rest = (m.group("rest") or "").strip()

            # month mapping
            month_map = {
                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
            }
            mm = month_map.get(mon, 1)
            try:
                dt = datetime(detected_year, mm, day)
                current_date_iso = dt.strftime("%Y-%m-%d")
            except Exception:
                current_date_iso = None

            if current_date_iso:
                current_buf = [rest] if rest else []
            else:
                current_buf = []
        else:
            # continuation line
            if current_date_iso:
                current_buf.append(ln)

    flush()
    return txs, prev_balance, seq


# =========================================================
# Main Entry
# =========================================================

def parse_ambank(pdf: pdfplumber.PDF, filename: str = "") -> List[Dict]:
    info = extract_statement_info(pdf)
    detected_year = info.get("year") or datetime.utcnow().year

    transactions: List[Dict] = []
    prev_balance = info.get("opening_balance")
    seq = 0

    for page_idx, page in enumerate(pdf.pages, start=1):
        text = page.extract_text(x_tolerance=1) or ""
        lines = _normalize_lines(text)

        # quick skip obvious headers to reduce noise
        cleaned: List[str] = []
        for ln in lines:
            up = ln.upper()
            if up.startswith("DATE ") or up.startswith("TARIKH ") or "ACCOUNT SUMMARY" in up:
                continue
            cleaned.append(ln)

        txs, prev_balance, seq = _parse_transactions_from_lines(
            cleaned,
            page_num=page_idx,
            filename=filename,
            detected_year=int(detected_year),
            prev_balance=prev_balance,
            seq_start=seq,
        )
        transactions.extend(txs)

    transactions = _sort_transactions(transactions)
    return transactions
