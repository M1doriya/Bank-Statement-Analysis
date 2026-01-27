# cimb.py - CIMB Bank Parser (FIXED)
#
# Key fixes:
# 1) Extract Opening Balance from top of page 1 ("Opening Balance ....").
# 2) Handle CIMB reverse-order listing (latest first) by detecting descending dates and reversing
#    before computing debit/credit via balance delta (prevents flipped debit/credit).
# 3) Robust multiline description parsing from text when tables are unreliable.
# 4) Optional statement marker rows:
#    - Opening balance row: is_opening_balance=True
#    - Closing balance row: is_statement_balance=True
#
# Expected interface:
#   parse_transactions_cimb(pdfplumber_pdf, source_filename="") -> List[Dict]

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Regex
# -----------------------------
STMT_DATE_RE = re.compile(
    r"(?:STATEMENT\s+DATE|TARIKH\s+PENYATA)\s*[:\s]+(\d{1,2})/(\d{1,2})/(\d{2,4})",
    re.IGNORECASE,
)

OPENING_BAL_RE = re.compile(
    r"\bOPENING\s+BALANCE\b\s+(-?[\d,]+\.\d{2})",
    re.IGNORECASE,
)

CLOSING_BAL_RE = re.compile(
    r"CLOSING\s+BALANCE\s*/\s*BAKI\s+PENUTUP\s+(-?[\d,]+\.\d{2})",
    re.IGNORECASE,
)

DATE_AT_START_RE = re.compile(r"^(?P<d>\d{2})/(?P<m>\d{2})(?:/(?P<y>\d{2,4}))?\b")

# money tokens like 1,234.56 or -1,234.56
MONEY_TOKEN_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*\.\d{2}$")


def _safe_money_to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _prev_month(year: int, month: int) -> Tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def _extract_statement_year_and_month(pdf) -> Tuple[Optional[int], Optional[str]]:
    """
    CIMB statements often show Statement Date as NEXT month (e.g. 01/08/2024 for July statement).
    We return:
      - detected_year: year used for dates without year
      - statement_month: YYYY-MM for the covered month (shift statement date back by 1 month)
    """
    if not pdf.pages:
        return None, None

    text1 = (pdf.pages[0].extract_text() or "")
    m = STMT_DATE_RE.search(text1)
    if not m:
        return None, None

    dd = int(m.group(1))
    mm = int(m.group(2))
    yy_raw = m.group(3)
    yy = (2000 + int(yy_raw)) if len(yy_raw) == 2 else int(yy_raw)

    if not (1 <= mm <= 12 and 2000 <= yy <= 2100):
        return None, None

    py, pm = _prev_month(yy, mm)
    return py, f"{py:04d}-{pm:02d}"


def _extract_opening_balance(pdf) -> Optional[float]:
    if not pdf.pages:
        return None
    text1 = (pdf.pages[0].extract_text() or "")
    m = OPENING_BAL_RE.search(text1)
    if not m:
        return None
    return _safe_money_to_float(m.group(1))


def _extract_closing_balance(pdf) -> Optional[float]:
    # footer is often near the end; scan all pages (reverse first is faster)
    for page in reversed(pdf.pages):
        txt = page.extract_text() or ""
        m = CLOSING_BAL_RE.search(txt)
        if m:
            return _safe_money_to_float(m.group(1))

    # fallback: full text
    full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    m2 = CLOSING_BAL_RE.search(full_text)
    if m2:
        return _safe_money_to_float(m2.group(1))
    return None


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _parse_text_rows(pdf, source_filename: str, detected_year: int) -> List[Dict[str, Any]]:
    """
    Text-mode parser:
    - Detect a new transaction when a line starts with DD/MM(/YYYY)
    - Collect multiline description until we see a line that contains a trailing balance token
    - Extract balance as last money token on the line that finalizes the transaction
    """
    rows: List[Dict[str, Any]] = []

    cur: Optional[Dict[str, Any]] = None

    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        if not text:
            continue

        # Keep raw line order
        for raw in text.splitlines():
            ln = _normalize_spaces(raw)
            if not ln:
                continue

            # skip obvious headers
            up = ln.upper()
            if (
                "STATEMENT OF ACCOUNT" in up
                or "CURRENT ACCOUNT TRANSACTION DETAILS" in up
                or "DATE DESCRIPTION" in up
                or "TARIKH DISKRIPSI" in up
                or "ACCOUNT NO" in up
                or up.startswith("PAGE ")
                or "CIMB BANK BERHAD" in up
            ):
                continue

            # opening balance line itself (no date) -> ignore here (we extract separately)
            if up.startswith("OPENING BALANCE"):
                continue

            # Start of a tx line
            m = DATE_AT_START_RE.match(ln)
            if m:
                # flush previous incomplete tx if any
                if cur is not None:
                    # if we never found a balance line, drop it (can't compute reliably)
                    cur = None

                cur = {
                    "date_raw": m.group(0),  # e.g. 31/07/2024 or 31/07
                    "parts": [ln[m.end():].strip()],
                    "page": page_num,
                }
                continue

            # Continuation line
            if cur is not None:
                cur["parts"].append(ln)

                # If this continuation line (or combined) ends with balance token, finalize.
                # Strategy: look at the LAST line we just appended: often contains "... amount balance"
                last_line = ln
                toks = last_line.split()
                money_idxs = [i for i, t in enumerate(toks) if MONEY_TOKEN_RE.match(t)]
                if money_idxs:
                    # last money token = balance
                    bal_token = toks[money_idxs[-1]]
                    bal = _safe_money_to_float(bal_token)
                    if bal is not None:
                        # Build description by stripping trailing numeric columns from the final line
                        # and including previous parts as-is.
                        desc_parts: List[str] = []
                        # first part is from tx-start line remainder (already captured)
                        for p in cur["parts"][:-1]:
                            if p:
                                desc_parts.append(p)

                        # last line: remove all tokens from first money token onward
                        cut = money_idxs[0]
                        last_desc = " ".join(toks[:cut]).strip()
                        if last_desc:
                            desc_parts.append(last_desc)

                        desc = _normalize_spaces(" ".join(desc_parts))

                        # date
                        date_raw = cur["date_raw"]
                        # handle DD/MM/YYYY or DD/MM
                        dmY = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", date_raw)
                        dm = re.match(r"^(\d{2})/(\d{2})$", date_raw)

                        date_iso = None
                        if dmY:
                            dd, mm, yyyy = dmY.groups()
                            date_iso = f"{yyyy}-{mm}-{dd}"
                        elif dm:
                            dd, mm = dm.groups()
                            date_iso = f"{detected_year:04d}-{mm}-{dd}"

                        if date_iso:
                            rows.append(
                                {
                                    "date": date_iso,
                                    "description": desc,
                                    "debit": 0.0,   # filled later by delta after ordering fix
                                    "credit": 0.0,  # filled later by delta after ordering fix
                                    "balance": round(float(bal), 2),
                                    "page": cur["page"],
                                    "source_file": source_filename,
                                    "bank": "CIMB Bank",
                                }
                            )

                        cur = None

    return rows


def _dates_look_descending(rows: List[Dict[str, Any]]) -> bool:
    """
    Detect reverse-order statements:
    - If first non-null date is later than last non-null date => descending => reverse needed.
    """
    dates = [r.get("date") for r in rows if r.get("date")]
    if len(dates) < 2:
        return False
    try:
        return dates[0] > dates[-1]
    except Exception:
        return False


def _apply_balance_delta(rows: List[Dict[str, Any]]) -> None:
    """
    Given rows in CHRONOLOGICAL order, compute debit/credit by balance delta.
    Assumption: balances are post-transaction balances.
    """
    prev_bal: Optional[float] = None
    for r in rows:
        bal = r.get("balance")
        if bal is None:
            r["debit"] = 0.0
            r["credit"] = 0.0
            continue

        if prev_bal is None:
            # first row can't be delta-classified without opening balance
            r["debit"] = 0.0
            r["credit"] = 0.0
            prev_bal = float(bal)
            continue

        delta = round(float(bal) - float(prev_bal), 2)
        if delta > 0:
            r["credit"] = round(delta, 2)
            r["debit"] = 0.0
        elif delta < 0:
            r["debit"] = round(-delta, 2)
            r["credit"] = 0.0
        else:
            r["debit"] = 0.0
            r["credit"] = 0.0

        prev_bal = float(bal)


def parse_transactions_cimb(pdf, source_filename: str = "") -> List[Dict[str, Any]]:
    """
    Main CIMB parser (pdfplumber PDF object).
    Uses robust text parsing because CIMB tables are frequently inconsistent for pdfplumber.
    """
    if not getattr(pdf, "pages", None):
        return []

    detected_year, stmt_month = _extract_statement_year_and_month(pdf)
    if detected_year is None:
        # fallback: current year (best-effort)
        detected_year = datetime.now().year

    opening_bal = _extract_opening_balance(pdf)
    closing_bal = _extract_closing_balance(pdf)

    # Parse text rows (transactions)
    rows = _parse_text_rows(pdf, source_filename=source_filename, detected_year=int(detected_year))

    # CIMB commonly lists newest first (reverse chronological).
    # Reverse BEFORE delta calculation so debit/credit don't flip.
    if _dates_look_descending(rows):
        rows = list(reversed(rows))

    # Apply delta debit/credit
    _apply_balance_delta(rows)

    out: List[Dict[str, Any]] = []

    # Emit one opening balance marker row if found
    if opening_bal is not None:
        # Use statement month to pin a stable date for the marker (YYYY-MM-01)
        marker_date = None
        if stmt_month and re.match(r"^\d{4}-\d{2}$", stmt_month):
            marker_date = f"{stmt_month}-01"
        else:
            # fallback: first tx date if exists
            marker_date = rows[0]["date"] if rows else f"{detected_year}-01-01"

        out.append(
            {
                "date": marker_date,
                "description": "OPENING BALANCE",
                "debit": 0.0,
                "credit": 0.0,
                "balance": round(float(opening_bal), 2),
                "page": 1,
                "source_file": source_filename,
                "bank": "CIMB Bank",
                "is_opening_balance": True,
            }
        )

    # Normal tx rows
    out.extend(rows)

    # Emit closing balance marker row if found (optional but useful for reconciliation)
    if closing_bal is not None:
        marker_date = None
        if stmt_month and re.match(r"^\d{4}-\d{2}$", stmt_month):
            # end-of-month marker date (best-effort, not exact day)
            # use last tx date if exists to avoid future dates
            marker_date = rows[-1]["date"] if rows else f"{stmt_month}-28"
        else:
            marker_date = rows[-1]["date"] if rows else f"{detected_year}-01-01"

        out.append(
            {
                "date": marker_date,
                "description": "CLOSING BALANCE / BAKI PENUTUP",
                "debit": 0.0,
                "credit": 0.0,
                "balance": round(float(closing_bal), 2),
                "page": None,
                "source_file": source_filename,
                "bank": "CIMB Bank",
                "is_statement_balance": True,
            }
        )

    return out
