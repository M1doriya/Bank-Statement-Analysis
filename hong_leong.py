import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Patterns
# -----------------------------
DATE_RE = re.compile(r"^(?P<d>\d{2})-(?P<m>\d{2})-(?P<y>\d{4})$")
MONEY_TOKEN_RE = re.compile(r"^\d{1,3}(?:,\d{3})*\.\d{2}$")

OPENING_RE = re.compile(r"Balance\s+from\s+previous\s+statement", re.I)
OPENING_AMT_RE = re.compile(r"Balance\s+from\s+previous\s+statement\s+([\d,]+\.\d{2})", re.I)
TOTAL_ROW_RE = re.compile(
    r"(Total\s+Deposits|Total\s+Withdrawals|Closing\s+Balance|Important\s+Notices|"
    r"Jumlah\s+Simpanan|Jumlah\s+Pengeluaran|Baki\s+Akhir|Notis\s+Penting)",
    re.I,
)

NOISE_RE = re.compile(
    r"(Protected by PIDM|Dilindungi oleh PIDM|Hong Leong Islamic Bank|hlisb\.com\.my|Menara Hong Leong|"
    r"CURRENT ACCOUNT|AKAUN SEMASA|Page\s+No\.?)",
    re.I,
)


# =========================================================
# Public entrypoint used by your app.py
# =========================================================

def parse_hong_leong(pdf, filename: str) -> List[Dict]:
    """
    Production-safe Hong Leong Islamic statement parser.

    Key principle:
      - Balance column (balance_pdf) is authoritative when present.
      - Waterfall reconstruction is only a fallback.
      - balance_pdf anchors the running balance to prevent drift -> prevents false OD.
    """
    opening_balance = extract_opening_balance(pdf)
    running_balance = opening_balance

    tx_rows: List[Dict[str, Any]] = []

    for page_num, page in enumerate(pdf.pages, start=1):
        words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False, use_text_flow=True)
        if not words:
            continue

        # group into visual lines by y
        lines = _cluster_words_by_y(words, bucket=3)

        # detect columns dynamically per page (more robust than fixed X ranges)
        col = _detect_columns_from_money(lines)

        for line_words in lines:
            full_text = " ".join(w["text"] for w in line_words).strip()
            if not full_text:
                continue

            if TOTAL_ROW_RE.search(full_text):
                continue

            # Identify opening row even if no date is present on that exact line
            is_opening_row = bool(OPENING_RE.search(full_text))

            date_str = _extract_date_from_line(line_words)
            if not date_str and not is_opening_row:
                continue

            # Parse the line into (desc, credit, debit, balance_pdf)
            desc, credit, debit, balance_pdf = _parse_line_by_columns(
                line_words=line_words,
                date_str=date_str,
                col=col,
            )

            # Clean description
            desc = OPENING_RE.sub("", desc).strip()
            desc = NOISE_RE.sub("", desc).strip()
            desc = re.sub(r"\s+", " ", desc).strip()

            # If this is opening row: we primarily want its balance_pdf as an anchor
            meta = "OPENING_FLAG" if is_opening_row else ""

            tx_rows.append(
                {
                    "date": date_str or "",
                    "description": desc,
                    "credit": float(credit),
                    "debit": float(debit),
                    "balance_pdf": balance_pdf,  # Optional[float]
                    "page": int(page_num),
                    "meta": meta,
                }
            )

    # Build final transactions with anchored waterfall logic
    out: List[Dict] = []
    running_balance = float(opening_balance)

    for r in tx_rows:
        # Skip empty junk lines
        if not r["date"] and not r["description"] and r.get("meta") != "OPENING_FLAG":
            continue

        # Apply movement
        credit = float(r["credit"])
        debit = float(r["debit"])

        # If this row has a statement balance, anchor to it (most important fix)
        if r["balance_pdf"] is not None:
            # If it's the opening row, anchor without applying movements twice
            if r.get("meta") == "OPENING_FLAG":
                running_balance = float(r["balance_pdf"])
            else:
                # Normal transaction: apply movement, then reconcile/anchor to printed balance
                computed = round(running_balance + credit - debit, 2)
                printed = float(r["balance_pdf"])

                # If mismatch, trust printed balance and reset running balance
                # (Prevents drift that causes negative lowest balance)
                if abs(computed - printed) > 0.01:
                    running_balance = printed
                else:
                    running_balance = computed
        else:
            # No printed balance available: fallback to computed
            running_balance = round(running_balance + credit - debit, 2)

        # Do not emit the opening row as a “transaction” unless it has actual movement
        if r.get("meta") == "OPENING_FLAG" and credit == 0.0 and debit == 0.0:
            continue

        # Normalize date
        date_iso = ""
        if r["date"]:
            date_iso = datetime.strptime(r["date"], "%d-%m-%Y").date().isoformat()

        out.append(
            {
                "date": date_iso,
                "description": r["description"],
                "debit": round(debit, 2),
                "credit": round(credit, 2),
                "balance": round(running_balance, 2),
                "page": r["page"],
                "bank": "Hong Leong Islamic Bank",
                "source_file": filename,
            }
        )

    return out


# =========================================================
# Opening balance extraction
# =========================================================

def extract_opening_balance(pdf) -> float:
    text = pdf.pages[0].extract_text() or ""
    m = OPENING_AMT_RE.search(text)
    if not m:
        raise ValueError("Opening balance not found (Balance from previous statement).")
    return float(m.group(1).replace(",", ""))


# =========================================================
# Line clustering and parsing helpers
# =========================================================

def _cluster_words_by_y(words: List[Dict[str, Any]], bucket: int = 3) -> List[List[Dict[str, Any]]]:
    """
    Group pdfplumber words into lines using bucketed top coordinate.
    """
    lines: Dict[float, List[Dict[str, Any]]] = {}
    for w in words:
        y_key = round(float(w["top"]) / bucket) * bucket
        lines.setdefault(y_key, []).append(w)

    out: List[List[Dict[str, Any]]] = []
    for y in sorted(lines.keys()):
        line = sorted(lines[y], key=lambda x: float(x["x0"]))
        out.append(line)

    return out


def _extract_date_from_line(line_words: List[Dict[str, Any]]) -> Optional[str]:
    for w in line_words:
        t = w["text"].strip()
        if DATE_RE.match(t):
            return t
    return None


def _detect_columns_from_money(lines: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Detect deposit/withdrawal/balance columns using 1D clustering of money token x positions.
    This avoids fixed X ranges that break when statement scale/layout shifts.
    """
    xs: List[float] = []
    for line in lines:
        full = " ".join(w["text"] for w in line)
        if TOTAL_ROW_RE.search(full):
            continue
        for w in line:
            t = w["text"].strip()
            if MONEY_TOKEN_RE.match(t):
                # ignore far-left numeric occurrences inside descriptions
                if float(w["x0"]) >= 250:
                    xs.append((float(w["x0"]) + float(w["x1"])) / 2.0)

    # If insufficient evidence, fall back to your colleague's approximate zones
    if len(xs) < 15:
        return {"credit_mid": 385.0, "debit_mid": 485.0, "bal_mid": 560.0}

    centroids = _kmeans_1d(xs, k=3, iters=12)
    centroids.sort()
    return {"credit_mid": centroids[0], "debit_mid": centroids[1], "bal_mid": centroids[2]}


def _kmeans_1d(values: List[float], k: int = 3, iters: int = 10) -> List[float]:
    v = sorted(values)

    def q(p: float) -> float:
        idx = int(p * (len(v) - 1))
        return float(v[idx])

    centroids = [q(0.2), q(0.5), q(0.8)][:k]

    for _ in range(iters):
        buckets = [[] for _ in range(k)]
        for x in v:
            j = min(range(k), key=lambda i: abs(x - centroids[i]))
            buckets[j].append(x)
        for i in range(k):
            if buckets[i]:
                centroids[i] = float(sum(buckets[i]) / len(buckets[i]))

    return centroids


def _parse_line_by_columns(
    line_words: List[Dict[str, Any]],
    date_str: Optional[str],
    col: Dict[str, float],
) -> Tuple[str, float, float, Optional[float]]:
    """
    Parse one visual line:
      - description: everything non-money and not the date token
      - credit: money nearest to credit column centroid
      - debit:  money nearest to debit column centroid
      - balance_pdf: money nearest to balance column centroid (authoritative if present)
    """
    desc_parts: List[str] = []
    money_candidates: List[Tuple[float, float]] = []  # (mid_x, value)

    for w in line_words:
        txt = w["text"].strip()
        if not txt:
            continue

        if date_str and txt == date_str:
            continue

        if MONEY_TOKEN_RE.match(txt):
            val = float(txt.replace(",", ""))
            mid_x = (float(w["x0"]) + float(w["x1"])) / 2.0
            money_candidates.append((mid_x, val))
        else:
            desc_parts.append(txt)

    desc = " ".join(desc_parts).strip()

    # Assign money tokens to nearest centroid
    credit = 0.0
    debit = 0.0
    balance_pdf: Optional[float] = None

    for mid_x, val in money_candidates:
        dc = abs(mid_x - col["credit_mid"])
        dd = abs(mid_x - col["debit_mid"])
        db = abs(mid_x - col["bal_mid"])

        # choose closest
        if dc <= dd and dc <= db:
            credit = val
        elif dd <= dc and dd <= db:
            debit = val
        else:
            balance_pdf = val

    return desc, credit, debit, balance_pdf
