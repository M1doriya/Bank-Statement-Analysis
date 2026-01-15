"""core_utils.py

Project-wide utilities used by Streamlit apps and bank parsers.

Goals:
1) Standardize input handling (PDF bytes)
2) Standardize transaction schema and types
3) Make date/amount parsing resilient across banks
4) Provide best-effort reconciliation/QA signals
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# PDF INPUT
# -----------------------------
def read_pdf_bytes(pdf_input: Any) -> bytes:
    """Return PDF bytes from:
    - bytes / bytearray
    - Streamlit UploadedFile (has getvalue)
    - file-like objects (has read)
    - filesystem path (str)
    """
    if isinstance(pdf_input, (bytes, bytearray)):
        return bytes(pdf_input)

    # Streamlit UploadedFile
    if hasattr(pdf_input, "getvalue"):
        data = pdf_input.getvalue()
        if data:
            return data

    # file-like
    if hasattr(pdf_input, "read"):
        try:
            pdf_input.seek(0)
        except Exception:
            pass
        data = pdf_input.read()
        if data:
            return data

    # path
    if isinstance(pdf_input, str):
        with open(pdf_input, "rb") as f:
            return f.read()

    raise ValueError("Unable to read PDF bytes from the provided input")


def bytes_to_pdfplumber(pdf_bytes: bytes):
    """Helper to open pdfplumber using bytes."""
    import pdfplumber  # local import to keep utils lightweight
    return pdfplumber.open(BytesIO(pdf_bytes))


# -----------------------------
# NORMALIZATION
# -----------------------------
_WS_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    return _WS_RE.sub(" ", str(text or "")).strip()


def safe_float(value: Any) -> float:
    """Convert numeric strings to float safely.

    Handles:
    - None / empty
    - commas
    - parentheses negatives: (1,234.56)
    - trailing +/-: 123.45- / 123.45+
    - currency symbols and stray text
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return 0.0

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # trailing sign
    trailing_sign = None
    if s.endswith("+"):
        trailing_sign = "+"
        s = s[:-1].strip()
    elif s.endswith("-"):
        trailing_sign = "-"
        s = s[:-1].strip()

    s = s.replace(",", "")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", "-", "."}:
        return 0.0

    try:
        f = float(s)
    except Exception:
        return 0.0

    if neg or trailing_sign == "-":
        f = -abs(f)
    elif trailing_sign == "+":
        f = abs(f)
    return float(f)


def normalize_date(date_value: Any, default_year: Optional[int] = None) -> Optional[str]:
    """Normalize many common bank-statement date formats to ISO YYYY-MM-DD.
    Returns None if parsing fails.
    """
    if date_value is None:
        return None

    s = normalize_text(date_value)
    if not s:
        return None

    # already ISO
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s

    # common patterns (day-first)
    patterns: List[Tuple[str, str]] = [
        (r"^\d{1,2}/\d{1,2}/\d{4}$", "%d/%m/%Y"),
        (r"^\d{1,2}-\d{1,2}-\d{4}$", "%d-%m-%Y"),
        (r"^\d{1,2}/\d{1,2}/\d{2}$", "%d/%m/%y"),
        (r"^\d{1,2}-\d{1,2}-\d{2}$", "%d-%m-%y"),
        (r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$", "%d %b %Y"),
        (r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{2}$", "%d %b %y"),
        (r"^\d{1,2}\s+[A-Za-z]{3}$", "%d %b"),
        (r"^\d{1,2}/\d{1,2}$", "%d/%m"),
        (r"^\d{1,2}-\d{1,2}$", "%d-%m"),
    ]

    for rx, fmt in patterns:
        if not re.fullmatch(rx, s):
            continue
        try:
            if fmt in {"%d %b", "%d/%m", "%d-%m"}:
                if default_year is None:
                    return None
                dt = datetime.strptime(f"{s} {default_year}", fmt + " %Y")
            else:
                dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # last-resort: dateutil
    try:
        from dateutil import parser as dateparser

        dt = dateparser.parse(
            s,
            dayfirst=True,
            yearfirst=False,
            default=datetime(default_year or 2000, 1, 1),
        )
        # if no explicit year and default_year is None, dt will use 2000 and likely be wrong -> reject
        if default_year is None and dt.year == 2000 and not re.search(r"\b\d{4}\b", s):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def infer_default_year(transactions: Iterable[Dict[str, Any]]) -> Optional[int]:
    """Infer a reasonable default year from any transaction that already contains a year."""
    for tx in transactions:
        d = normalize_text(tx.get("date"))
        if re.search(r"\b\d{4}\b", d):
            iso = normalize_date(d)
            if iso:
                return int(iso[:4])
    return None


def ensure_transaction_schema(
    tx: Dict[str, Any],
    *,
    default_bank: str,
    default_source_file: str,
    default_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a sanitized transaction dict with consistent keys and types."""
    raw_date = tx.get("date")
    date_iso = normalize_date(raw_date, default_year=default_year)

    description = normalize_text(tx.get("description"))
    debit = safe_float(tx.get("debit", 0))
    credit = safe_float(tx.get("credit", 0))

    # Some parsers store negative values; normalize to non-negative debit/credit where possible
    if debit < 0 and credit == 0:
        credit = abs(debit)
        debit = 0.0
    if credit < 0 and debit == 0:
        debit = abs(credit)
        credit = 0.0

    balance_raw = tx.get("balance", None)
    balance = safe_float(balance_raw) if balance_raw is not None and str(balance_raw).strip() != "" else None

    page_raw = tx.get("page")
    try:
        page = int(page_raw) if page_raw is not None and str(page_raw).strip() != "" else None
    except Exception:
        page = None

    bank = normalize_text(tx.get("bank")) or default_bank
    source_file = normalize_text(tx.get("source_file")) or default_source_file

    # Optional semantic fields (backwards-compatible)
    # - row_type: "transaction" | "opening_balance" | "closing_balance" | "statement_balance" | "summary" | "unknown"
    # - inference_method: "explicit" | "delta" | "unknown"
    # - confidence: "high" | "medium" | "low"
    #
    # Backwards-compatible support: some parsers used boolean markers.
    if tx.get("is_statement_balance") is True and not tx.get("row_type"):
        tx = dict(tx)
        tx["row_type"] = "statement_balance"

    row_type = normalize_text(tx.get("row_type") or tx.get("type") or "transaction").lower()
    if row_type in {"txn", "tx", "trans", "transactions"}:
        row_type = "transaction"
    if row_type in {"opening", "opening bal", "opening_balance"}:
        row_type = "opening_balance"
    if row_type in {"closing", "closing bal", "closing_balance", "ending_balance"}:
        row_type = "closing_balance"
    if row_type in {"statement", "statement_balance", "balance_row"}:
        row_type = "statement_balance"

    inference_method = normalize_text(tx.get("inference_method") or "").lower() or None
    if inference_method in {"", "none"}:
        inference_method = None
    confidence = normalize_text(tx.get("confidence") or "").lower() or None
    if confidence in {"", "none"}:
        confidence = None

    out: Dict[str, Any] = {
        "date": date_iso or normalize_text(raw_date),
        "description": description,
        "debit": round(float(debit), 2),
        "credit": round(float(credit), 2),
        "balance": round(float(balance), 2) if isinstance(balance, (int, float)) else None,
        "page": page,
        "bank": bank,
        "source_file": source_file,
        "row_type": row_type,
    }

    if inference_method:
        out["inference_method"] = inference_method
    if confidence:
        out["confidence"] = confidence

    # retain raw date if normalization changed it
    if date_iso and normalize_text(raw_date) and normalize_text(raw_date) != date_iso:
        out["_raw_date"] = normalize_text(raw_date)

    return out


def split_transactions_by_type(transactions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience: group transactions by row_type."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for tx in transactions:
        rt = normalize_text(tx.get("row_type") or "transaction").lower() or "transaction"
        groups.setdefault(rt, []).append(tx)
    return groups


def reconcile_statement(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Lightweight reconciliation/QA signals.

    This does NOT assume every statement has opening/closing lines.
    It provides best-effort checks to help you identify parsing drift.
    """
    if not transactions:
        return {"ok": True, "reason": "no_transactions"}

    groups = split_transactions_by_type(transactions)
    txns = groups.get("transaction", transactions)

    # Totals
    total_debit = round(sum(safe_float(t.get("debit", 0)) for t in txns), 2)
    total_credit = round(sum(safe_float(t.get("credit", 0)) for t in txns), 2)

    # Opening/closing (best effort)
    opening_candidates = groups.get("opening_balance", [])
    closing_candidates = groups.get("closing_balance", []) + groups.get("statement_balance", [])

    opening_balance = None
    if opening_candidates:
        # first opening balance with non-null balance
        for t in opening_candidates:
            b = t.get("balance")
            if b is not None:
                opening_balance = round(float(safe_float(b)), 2)
                break

    closing_balance = None
    if closing_candidates:
        for t in reversed(closing_candidates):
            b = t.get("balance")
            if b is not None:
                closing_balance = round(float(safe_float(b)), 2)
                break

    # If no explicit statement balances exist, try from last transaction balance
    if closing_balance is None:
        for t in reversed(txns):
            b = t.get("balance")
            if b is not None:
                closing_balance = round(float(safe_float(b)), 2)
                break

    # Reconciliation check when both opening + closing exist
    ok = True
    expected_closing = None
    diff = None
    if opening_balance is not None and closing_balance is not None:
        expected_closing = round(opening_balance + total_credit - total_debit, 2)
        diff = round(closing_balance - expected_closing, 2)
        # allow small rounding noise
        ok = abs(diff) <= 0.05

    return {
        "ok": ok,
        "total_debit": total_debit,
        "total_credit": total_credit,
        "opening_balance": opening_balance,
        "closing_balance": closing_balance,
        "expected_closing": expected_closing,
        "difference": diff,
    }


def normalize_transactions(
    transactions: List[Dict[str, Any]],
    *,
    default_bank: str,
    source_file: str,
) -> List[Dict[str, Any]]:
    """Normalize a list of transactions and infer year if needed."""
    year = infer_default_year(transactions)
    return [
        ensure_transaction_schema(
            tx,
            default_bank=default_bank,
            default_source_file=source_file,
            default_year=year,
        )
        for tx in transactions
    ]


def transaction_fingerprint(tx: Dict[str, Any]) -> str:
    """Create a stable fingerprint suitable for de-duplication."""
    parts = [
        normalize_text(tx.get("date")),
        normalize_text(tx.get("description")),
        f"{safe_float(tx.get('debit', 0)):.2f}",
        f"{safe_float(tx.get('credit', 0)):.2f}",
        "" if tx.get("balance") is None else f"{safe_float(tx.get('balance')):.2f}",
        normalize_text(tx.get("bank")),
    ]
    blob = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(blob).hexdigest()


def dedupe_transactions(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for tx in transactions:
        fp = transaction_fingerprint(tx)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(tx)
    return out
