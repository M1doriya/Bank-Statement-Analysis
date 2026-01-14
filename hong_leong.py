import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Regex
# -----------------------------
DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
MONEY_RE = re.compile(r"^\d{1,3}(?:,\d{3})*\.\d{2}$")

OPENING_AMT_RE = re.compile(r"Balance\s+from\s+previous\s+statement\s+([\d,]+\.\d{2})", re.I)
OPENING_LINE_RE = re.compile(r"Balance\s+from\s+previous\s+statement", re.I)

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

OD_KEYWORDS_RE = re.compile(
    r"\b(overdraft|od\s+facility|od\s+limit|overdrawn|excess\s+limit|interest\s+on\s+overdraft)\b",
    re.I,
)


# =========================================================
# Public entrypoint
# =========================================================

def parse_hong_leong(pdf, filename: str) -> List[Dict]:
    """
    Robust HLIB parser:
      - Detect columns per page via clustering (no fixed X ranges).
      - Balance = rightmost money token on the dated line (most reliable).
      - Use balance delta to resolve debit/credit when classification is ambiguous.
      - Avoid false OD: if statement never shows negative balances, output won't either.
    """
    opening_balance = extract_opening_balance(pdf)
    overdraft_possible = pdf_mentions_overdraft(pdf)

    out: List[Dict] = []
    prev_balance: Optional[float] = opening_balance

    for page_num, page in enumerate(pdf.pages, start=1):
        words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False, use_text_flow=True)
        if not words:
            continue

        lines = cluster_words_by_y(words, bucket=3)

        # skip pages that are purely notices
        if not lines:
            continue

        # detect 3 numeric column centers using money token x positions
        col = detect_money_columns(lines)

        for line_words in lines:
            line_text = " ".join(w["text"] for w in line_words).strip()
            if not line_text:
                continue
            if TOTAL_ROW_RE.search(line_text):
                continue

            # opening line: do not emit transaction; just re-anchor prev_balance if we see a printed amount
            if OPENING_LINE_RE.search(line_text):
                b = rightmost_money_value(line_words)
                if b is not None:
                    prev_balance = b
                continue

            date_raw = extract_date_token(line_words)
            if not date_raw:
                continue

            date_iso = datetime.strptime(date_raw, "%d-%m-%Y").date().isoformat()

            # Extract all money tokens on THIS dated line (do not pollute with continuation lines)
            money_tokens = extract_money_tokens(line_words)

            # Balance is the rightmost money token (most reliable across layouts)
            bal = pick_balance_from_tokens(money_tokens)

            # Description: left-of-numbers text
            desc = extract_description_from_line(line_words, date_raw, min_number_x=col["min_number_x"])
            desc = NOISE_RE.sub("", desc).strip()
            desc = re.sub(r"\s+", " ", desc).strip()

            # Now decide debit/credit:
            credit, debit = classify_credit_debit(money_tokens, col)

            # If we have a balance and previous balance, use delta as arbitration when suspicious
            if bal is not None and prev_balance is not None:
                delta = round(bal - prev_balance, 2)

                suspicious = False
                # classic bad cases
                if credit > 0 and debit > 0:
                    suspicious = True
                if bal == credit or bal == debit:
                    suspicious = True

                # also suspicious if extracted amounts imply a very different delta
                implied = round(credit - debit, 2)
                if abs(implied - delta) > 0.01:
                    suspicious = True

                if suspicious:
                    # Trust the balance delta: single-sided movement
                    if delta > 0:
                        credit, debit = delta, 0.0
                    elif delta < 0:
                        credit, debit = 0.0, abs(delta)
                    else:
                        credit, debit = 0.0, 0.0

            # Skip non-transaction (no movement)
            if credit == 0.0 and debit == 0.0:
                # still advance anchor if we have balance
                if bal is not None:
                    prev_balance = bal
                continue

            # Final balance output:
            # If statement balance exists, trust it. Otherwise compute.
            if bal is not None:
                final_balance = float(bal)
                prev_balance = bal
            else:
                # fallback compute
                prev_balance = (prev_balance if prev_balance is not None else 0.0)
                prev_balance = round(prev_balance + credit - debit, 2)
                final_balance = float(prev_balance)

            # Final safety: if OD not possible and we somehow got negative due to missing balances,
            # clamp by not allowing negative unless the statement explicitly mentions OD.
            if (not overdraft_possible) and final_balance < 0:
                # We do NOT fabricate a positive number; we simply avoid outputting a false negative balance.
                # Best option: set to None so OD logic doesn't get polluted.
                final_balance = None

            out.append(
                {
                    "date": date_iso,
                    "description": desc,
                    "debit": round(float(debit), 2),
                    "credit": round(float(credit), 2),
                    "balance": None if final_balance is None else round(float(final_balance), 2),
                    "page": int(page_num),
                    "bank": "Hong Leong Islamic Bank",
                    "source_file": filename,
                }
            )

    return out


# =========================================================
# Opening balance / OD text scan
# =========================================================

def extract_opening_balance(pdf) -> float:
    text = pdf.pages[0].extract_text() or ""
    m = OPENING_AMT_RE.search(text)
    if not m:
        raise ValueError("Opening balance not found (Balance from previous statement).")
    return float(m.group(1).replace(",", ""))


def pdf_mentions_overdraft(pdf) -> bool:
    for p in pdf.pages:
        t = (p.extract_text() or "")
        if OD_KEYWORDS_RE.search(t):
            return True
    return False


# =========================================================
# Line clustering
# =========================================================

def cluster_words_by_y(words: List[Dict[str, Any]], bucket: int = 3) -> List[List[Dict[str, Any]]]:
    lines: Dict[float, List[Dict[str, Any]]] = {}
    for w in words:
        y_key = round(float(w["top"]) / bucket) * bucket
        lines.setdefault(y_key, []).append(w)

    out: List[List[Dict[str, Any]]] = []
    for y in sorted(lines.keys()):
        line = sorted(lines[y], key=lambda x: float(x["x0"]))
        out.append(line)

    return out


def extract_date_token(line_words: List[Dict[str, Any]]) -> Optional[str]:
    for w in line_words:
        t = w["text"].strip()
        if DATE_RE.match(t):
            return t
    return None


# =========================================================
# Money column detection via clustering
# =========================================================

def detect_money_columns(lines: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    xs: List[float] = []
    for line in lines:
        full = " ".join(w["text"] for w in line)
        if TOTAL_ROW_RE.search(full):
            continue
        for w in line:
            t = w["text"].strip()
            if MONEY_RE.match(t):
                mid_x = (float(w["x0"]) + float(w["x1"])) / 2.0
                # ignore description area
                if mid_x >= 250:
                    xs.append(mid_x)

    # Fallback (rare)
    if len(xs) < 15:
        credit_mid, debit_mid, bal_mid = 385.0, 485.0, 560.0
    else:
        centroids = kmeans_1d(xs, k=3, iters=12)
        centroids.sort()
        credit_mid, debit_mid, bal_mid = centroids[0], centroids[1], centroids[2]

    # min_number_x for description cutoff:
    min_number_x = credit_mid - 60
    return {
        "credit_mid": float(credit_mid),
        "debit_mid": float(debit_mid),
        "bal_mid": float(bal_mid),
        "min_number_x": float(min_number_x),
    }


def kmeans_1d(values: List[float], k: int = 3, iters: int = 10) -> List[float]:
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


# =========================================================
# Token extraction / classification
# =========================================================

def extract_money_tokens(line_words: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """
    Returns list of (mid_x, value) for money tokens on the line.
    """
    out: List[Tuple[float, float]] = []
    for w in line_words:
        t = w["text"].strip()
        if MONEY_RE.match(t):
            val = float(t.replace(",", ""))
            mid_x = (float(w["x0"]) + float(w["x1"])) / 2.0
            out.append((mid_x, val))
    return out


def pick_balance_from_tokens(tokens: List[Tuple[float, float]]) -> Optional[float]:
    if not tokens:
        return None
    # balance is rightmost money token
    return max(tokens, key=lambda t: t[0])[1]


def rightmost_money_value(line_words: List[Dict[str, Any]]) -> Optional[float]:
    toks = extract_money_tokens(line_words)
    return pick_balance_from_tokens(toks)


def classify_credit_debit(tokens: List[Tuple[float, float]], col: Dict[str, float]) -> Tuple[float, float]:
    """
    Assign non-balance tokens to credit/debit by proximity to detected column midpoints.
    """
    if not tokens:
        return 0.0, 0.0

    bal_val = pick_balance_from_tokens(tokens)
    # remove one instance of balance (rightmost token)
    rightmost = max(tokens, key=lambda t: t[0])
    remaining = [t for t in tokens if t != rightmost]

    credit = 0.0
    debit = 0.0

    for mid_x, val in remaining:
        dc = abs(mid_x - col["credit_mid"])
        dd = abs(mid_x - col["debit_mid"])
        # if closer to credit, treat as credit; else debit
        if dc <= dd:
            credit += val
        else:
            debit += val

    return round(credit, 2), round(debit, 2)


def extract_description_from_line(line_words: List[Dict[str, Any]], date_raw: str, min_number_x: float) -> str:
    parts: List[str] = []
    for w in line_words:
        t = w["text"].strip()
        if not t:
            continue
        if t == date_raw:
            continue
        # keep only left-side tokens for description
        if float(w["x0"]) < min_number_x and not MONEY_RE.match(t):
            parts.append(t)
    return " ".join(parts).strip()
