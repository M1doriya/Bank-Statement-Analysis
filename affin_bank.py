# affin_bank.py
"""
Affin Bank Malaysia statement parser (robust for scanned PDFs + OCR).

Why prior versions become inaccurate:
- OCR can merge rows, scramble token order, and mis-assign numeric columns.
- If you trust "debit/credit column tokens" directly, you will emit wrong rows.
- If row ordering is wrong, any delta-based inference becomes wrong.

This implementation:
1) Extracts structured lines using pdfplumber word coordinates (preferred).
2) Falls back to OCR only if a page has no extractable text.
3) Parses rows using a conservative rule:
   - Balance is the anchor.
   - Debit/Credit are computed from balance deltas (prev_balance -> curr_balance).
4) Enforces stable ordering via (page, y_top, seq).
5) Applies anomaly checks to avoid propagating OCR-misread balances.

Output transaction fields match the rest of the project:
  - date: YYYY-MM-DD
  - description: str
  - debit: float
  - credit: float
  - balance: float|None
  - page: int
  - bank: "Affin Bank"
  - source_file: filename
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None


# -----------------------------
# Regex
# -----------------------------
DATE_RE = re.compile(r"^(?P<d>\d{1,2})/(?P<m>\d{2})/(?P<y>\d{2,4})$")
MONEY_RE = re.compile(r"^(?:\d{1,3}(?:,\d{3})*|\d+)?\.\d{2}$")  # allows ".00"

BF_RE = re.compile(r"\bB/?F\b", re.I)
TOTAL_RE = re.compile(r"\bTOTAL\b", re.I)

# Some common noise markers; we only use these to skip obvious non-rows.
NOISE_HINTS = (
    "PAGE",
    "STATEMENT",
    "PENYATA",
    "ACCOUNT",
    "AFFIN BANK",
    "PIDM",
    "MEMBER",
)


# -----------------------------
# Helpers
# -----------------------------
def _to_iso_date(dmy: str) -> Optional[str]:
    m = DATE_RE.match(dmy.strip())
    if not m:
        return None
    dd = int(m.group("d"))
    mm = int(m.group("m"))
    yy = int(m.group("y"))
    if yy < 100:
        yy += 2000
    try:
        return datetime(yy, mm, dd).strftime("%Y-%m-%d")
    except Exception:
        return None


def _money_to_float(tok: str) -> Optional[float]:
    if tok is None:
        return None
    s = str(tok).strip()
    if not s:
        return None
    if s.startswith("."):
        s = "0" + s
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _is_noise_line(line: str) -> bool:
    up = (line or "").upper()
    return any(h in up for h in NOISE_HINTS)


def _ocr_page_text(page: pdfplumber.page.Page) -> str:
    """
    OCR a page (cropped to the table region) using pytesseract.
    """
    if pytesseract is None:
        return ""

    try:
        w, h = float(page.width), float(page.height)
        # Conservative crop: remove typical header/footer, keep table.
        top = 130
        bottom = h - 70
        cropped = page.crop((0, top, w, bottom))

        img = cropped.to_image(resolution=200).original
        return pytesseract.image_to_string(img, config="--psm 6") or ""
    except Exception:
        return ""


def _extract_lines_structured(page: pdfplumber.page.Page) -> List[Tuple[float, str]]:
    """
    Preferred extraction: use word coordinates and rebuild lines.
    Returns list of (y_top, line_text) sorted top->bottom.
    """
    words = page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
        extra_attrs=["x0", "x1", "top", "bottom"],
    ) or []

    if not words:
        return []

    # Group words into lines by y (top). We bucket by rounded top position.
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for w in words:
        top = float(w.get("top", 0.0))
        key = int(round(top / 2.0))  # 2pt tolerance
        buckets.setdefault(key, []).append(w)

    lines: List[Tuple[float, str]] = []
    for key, ws in buckets.items():
        ws_sorted = sorted(ws, key=lambda z: float(z.get("x0", 0.0)))
        text = " ".join((z.get("text") or "").strip() for z in ws_sorted).strip()
        if text:
            # approximate y_top back from bucket key
            lines.append((key * 2.0, re.sub(r"\s+", " ", text)))

    lines.sort(key=lambda t: t[0])
    return lines


def _extract_lines_text_or_ocr(page: pdfplumber.page.Page) -> List[Tuple[float, str]]:
    """
    If structured extraction works, use it (best ordering). Otherwise,
    try extract_text lines; if empty, OCR.
    Returns list of (y_top, line_text). For OCR/plain text, y_top is synthetic.
    """
    structured = _extract_lines_structured(page)
    if structured:
        return structured

    text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
    if text.strip():
        out: List[Tuple[float, str]] = []
        for i, ln in enumerate(text.splitlines()):
            ln = re.sub(r"\s+", " ", (ln or "").strip())
            if ln:
                out.append((float(i), ln))
        return out

    # OCR fallback
    ocr = _ocr_page_text(page)
    out2: List[Tuple[float, str]] = []
    for i, ln in enumerate((ocr or "").splitlines()):
        ln = re.sub(r"\s+", " ", (ln or "").strip())
        if ln:
            out2.append((float(i), ln))
    return out2


# -----------------------------
# Row parsing
# -----------------------------
class _RowBuf:
    __slots__ = ("date_iso", "desc_parts", "balance", "page", "y", "seq")

    def __init__(self, date_iso: str, desc: str, balance: Optional[float], page: int, y: float, seq: int):
        self.date_iso = date_iso
        self.desc_parts = [desc] if desc else []
        self.balance = balance
        self.page = page
        self.y = y
        self.seq = seq

    @property
    def desc(self) -> str:
        return " ".join([p for p in self.desc_parts if p]).strip()


def _parse_line_to_row_start(line: str) -> Optional[Tuple[str, str, Optional[float]]]:
    """
    Returns (date_iso, desc_head, balance_guess) if the line looks like a transaction start.
    Conservative: must begin with date token.
    We also attempt to read a balance token from the far-right money tokens.
    """
    tokens = (line or "").split()
    if not tokens:
        return None
    date_iso = _to_iso_date(tokens[0])
    if not date_iso:
        return None

    money_idxs = [i for i, t in enumerate(tokens) if MONEY_RE.match(t)]
    balance = None
    desc_head = ""

    if money_idxs:
        # Prefer the last money token as balance guess; OCR may add extra tokens,
        # but the balance is still very often the rightmost/last.
        balance = _money_to_float(tokens[money_idxs[-1]])
        desc_head = " ".join(tokens[1:money_idxs[0]]).strip()
    else:
        desc_head = " ".join(tokens[1:]).strip()

    return date_iso, desc_head, balance


def _append_continuation(cur: _RowBuf, line: str) -> None:
    """
    Append continuation text and opportunistically update balance if a plausible one appears.
    """
    tokens = (line or "").split()
    if not tokens:
        return

    money_idxs = [i for i, t in enumerate(tokens) if MONEY_RE.match(t)]
    if money_idxs:
        # Update balance with last money token if it parses.
        bal = _money_to_float(tokens[money_idxs[-1]])
        if bal is not None:
            cur.balance = bal

        # Only append the non-money part to description.
        desc_part = " ".join(tokens[:money_idxs[0]]).strip()
        if desc_part and not _is_noise_line(desc_part):
            cur.desc_parts.append(desc_part)
    else:
        if not _is_noise_line(line):
            cur.desc_parts.append(line)


def _finalize_rows_to_transactions(
    rows: List[_RowBuf],
    source_file: str,
    bank_name: str = "Affin Bank",
) -> List[Dict[str, Any]]:
    """
    Convert parsed rows to transactions using balance deltas.
    Applies ordering + anomaly filtering.
    """
    # Stable ordering
    rows_sorted = sorted(rows, key=lambda r: (r.page, r.y, r.seq))

    # Remove obvious summary rows and use B/F as an anchor (do not emit it)
    cleaned: List[_RowBuf] = []
    for r in rows_sorted:
        d = r.desc.upper()
        if TOTAL_RE.search(d):
            continue
        if BF_RE.search(d):
            # keep as anchor row but do not emit; we keep it in cleaned for delta anchoring
            cleaned.append(r)
            continue
        cleaned.append(r)

    txs: List[Dict[str, Any]] = []

    # Find first usable anchor balance (prefer B/F; otherwise first row with balance)
    prev_balance: Optional[float] = None
    for r in cleaned:
        if r.balance is not None:
            prev_balance = float(r.balance)
            # If it's a B/F row, we anchor and do not emit it.
            if BF_RE.search(r.desc):
                break
            else:
                # If first row is not B/F, we still anchor but will emit this row normally
                break

    # Anomaly thresholds:
    # - If balance is None, we can still emit row with 0/0 but it is not useful; skip.
    # - If a single step delta is absurdly large (relative to typical statement ops),
    #   it is likely OCR mixed rows. We drop that row.
    #
    # The absolute threshold is a safety valve; you may tune it per your data.
    ABS_DELTA_DROP = 5_000_000.00  # RM 5m

    for idx, r in enumerate(cleaned):
        desc = r.desc.strip()
        if not desc:
            continue

        # Do not emit B/F; only use for anchoring
        if BF_RE.search(desc):
            if r.balance is not None:
                prev_balance = float(r.balance)
            continue

        if r.balance is None:
            # Without balance we cannot classify robustly; skip to avoid corrupt totals.
            # (If you prefer, you can emit as unknown here.)
            continue

        curr_balance = float(r.balance)

        debit = 0.0
        credit = 0.0

        if prev_balance is not None:
            delta = round(curr_balance - prev_balance, 2)

            if abs(delta) > ABS_DELTA_DROP:
                # Almost certainly OCR merged/misread balance; skip this row.
                # Do NOT update prev_balance.
                continue

            if delta > 0:
                credit = float(delta)
            elif delta < 0:
                debit = float(-delta)

        # If prev_balance is None (rare), we cannot infer; keep 0/0, but still emit row with balance.
        txs.append(
            {
                "date": r.date_iso,
                "description": desc,
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(curr_balance, 2),
                "page": int(r.page),
                "bank": bank_name,
                "source_file": source_file or "",
            }
        )

        prev_balance = curr_balance

    return txs


# -----------------------------
# Public entry point (used by app.py)
# -----------------------------
def parse_affin_bank(pdf_input: Any, source_file: str = "") -> List[Dict[str, Any]]:
    """
    Entry used by your app:
      parse_affin_bank(pdf, filename) -> list of tx dicts

    Supports:
      - pdfplumber PDF object
      - file path
      - file-like (streamlit upload)
      - bytes
    """
    bank_name = "Affin Bank"
    rows: List[_RowBuf] = []

    def parse_pdf(pdf: pdfplumber.PDF) -> None:
        seq_global = 0
        for page_idx, page in enumerate(pdf.pages, start=1):
            lines = _extract_lines_text_or_ocr(page)
            if not lines:
                continue

            cur: Optional[_RowBuf] = None

            for y_top, raw in lines:
                line = re.sub(r"\s+", " ", (raw or "").strip())
                if not line:
                    continue

                # Skip obvious noise lines only when not in the middle of a transaction
                if cur is None and _is_noise_line(line):
                    continue

                start = _parse_line_to_row_start(line)
                if start:
                    # finalize previous buffer
                    if cur is not None:
                        rows.append(cur)

                    date_iso, desc_head, bal = start
                    seq_global += 1
                    cur = _RowBuf(date_iso=date_iso, desc=desc_head, balance=bal, page=page_idx, y=float(y_top), seq=seq_global)
                    continue

                # Continuation line
                if cur is not None:
                    _append_continuation(cur, line)

            # finalize last buffer for this page
            if cur is not None:
                rows.append(cur)

    # Handle input types
    if hasattr(pdf_input, "pages"):
        parse_pdf(pdf_input)
        return _finalize_rows_to_transactions(rows, source_file=source_file, bank_name=bank_name)

    # file-like safety: rewind
    try:
        if hasattr(pdf_input, "seek"):
            pdf_input.seek(0)
    except Exception:
        pass

    try:
        with pdfplumber.open(pdf_input) as pdf:
            parse_pdf(pdf)
    except Exception as e:
        raise RuntimeError(f"Affin Bank parser failed for '{source_file}': {e}") from e

    return _finalize_rows_to_transactions(rows, source_file=source_file, bank_name=bank_name)
