# feature_engine_v4.py
# Deterministic metrics/flags/integrity for v4-style HTML report.
# Works on canonical transactions schema:
#   date (YYYY-MM-DD preferred), description, debit, credit, balance, source_file, bank, page
# Never invent numbers. If missing, mark N/A in renderer.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import pandas as pd


ROUND_FIGURE_RE = re.compile(r"(?:^|[^0-9])(\d{1,3}(?:,\d{3})+|\d+)\.00(?:$|[^0-9])")

# Basic category rules (expand later)
CREDIT_RULES = [
    ("Salary/Income", re.compile(r"\bSALARY|PAYROLL|GAJI|WAGES\b", re.I)),
    ("Transfer In", re.compile(r"\bCREDIT TRANSFER|IBG|GIRO|DUITNOW|INWARD|TRANSFER IN|TT IN\b", re.I)),
    ("Cash Deposit", re.compile(r"\bCASH DEPOSIT|CDM\b", re.I)),
    ("Refund/Reversal", re.compile(r"\bREVERSAL|REFUND\b", re.I)),
]

DEBIT_RULES = [
    ("Transfer Out", re.compile(r"\bDEBIT TRANSFER|IBG|GIRO|DUITNOW|OUTWARD|TRANSFER OUT|TT OUT\b", re.I)),
    ("Cash Withdrawal", re.compile(r"\bATM|CASH WITHDRAWAL|WITHDRAWAL\b", re.I)),
    ("Charges/Fees", re.compile(r"\bCHARGE|FEE|COMMISSION|SERVICE CHG|MAINTENANCE\b", re.I)),
    ("Card/Online Spend", re.compile(r"\bPOS|VISA|MASTER|CARD|ONLINE\b", re.I)),
    ("Tax/Govt", re.compile(r"\bLHDN|TAX|CUKAI|GST|SST\b", re.I)),
    ("Utilities", re.compile(r"\bTENAGA|TNB|SYABAS|AIR|WATER|UNIFI|TM\b", re.I)),
]

def _to_float(x: Any) -> float:
    try:
        if x is None or x == "":
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x:,.2f}"

def _parse_dates(df: pd.DataFrame) -> pd.Series:
    # Prefer ISO; fallback general.
    iso = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    fallback = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    return iso.fillna(fallback)

def _classify_row(desc: str, is_credit: bool) -> str:
    d = (desc or "").strip()
    rules = CREDIT_RULES if is_credit else DEBIT_RULES
    for cat, rx in rules:
        if rx.search(d):
            return cat
    return "Other"

def _volatility_label(pct: Optional[float]) -> str:
    if pct is None:
        return "N/A"
    if pct < 30:
        return "LOW"
    if pct < 60:
        return "MODERATE"
    if pct < 100:
        return "HIGH"
    return "EXTREME"

def _integrity_rating(score_pct: Optional[float]) -> Tuple[str, str]:
    # returns (css_class, label)
    if score_pct is None:
        return ("warning", "N/A")
    if score_pct >= 90:
        return ("excellent", "EXCELLENT")
    if score_pct >= 80:
        return ("good", "GOOD")
    if score_pct >= 65:
        return ("warning", "FAIR")
    return ("danger", "POOR")

def _suspicious_kite_flying(df: pd.DataFrame) -> int:
    """
    Very conservative heuristic:
    - look for same amount appearing as credit and debit within 1 day (per account/source grouping)
    - returns number of matched pairs (lower bound).
    """
    if df.empty:
        return 0
    df2 = df.dropna(subset=["date_parsed"]).copy()
    if df2.empty:
        return 0

    df2["amt_credit"] = df2["credit"].where(df2["credit"] > 0, 0.0)
    df2["amt_debit"] = df2["debit"].where(df2["debit"] > 0, 0.0)

    credits = df2[df2["amt_credit"] > 0][["date_parsed", "amt_credit"]].copy()
    debits = df2[df2["amt_debit"] > 0][["date_parsed", "amt_debit"]].copy()

    if credits.empty or debits.empty:
        return 0

    # bucket by rounded amount to 2 decimals
    credits["amt"] = credits["amt_credit"].round(2)
    debits["amt"] = debits["amt_debit"].round(2)

    # join on amount, then check date diff <= 1 day
    merged = credits.merge(debits, on="amt", suffixes=("_cr", "_dr"))
    if merged.empty:
        return 0

    merged["day_diff"] = (merged["date_parsed_cr"] - merged["date_parsed_dr"]).abs().dt.days
    return int((merged["day_diff"] <= 1).sum())

def build_v4_context(
    transactions: List[Dict[str, Any]],
    company_name: str = "Company",
    od_limit: Optional[float] = None,
    declared_sales: Optional[float] = None,
    related_party_keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:

    if not transactions:
        raise ValueError("No transactions to analyze.")

    df = pd.DataFrame(transactions).copy()
    for col in ["date", "description", "debit", "credit", "balance", "source_file", "bank", "page"]:
        if col not in df.columns:
            df[col] = None

    df["debit"] = df["debit"].apply(_to_float)
    df["credit"] = df["credit"].apply(_to_float)
    df["balance"] = df["balance"].apply(lambda x: float(x) if x is not None and str(x).strip() != "" else None)
    df["source_file"] = df["source_file"].fillna("UNKNOWN_SOURCE")
    df["bank"] = df["bank"].fillna("Unknown Bank")
    df["page"] = pd.to_numeric(df["page"], errors="coerce")

    df["date_parsed"] = _parse_dates(df)

    # Overall period
    dt_valid = df["date_parsed"].dropna()
    period_start = dt_valid.min().strftime("%Y-%m-%d") if not dt_valid.empty else "N/A"
    period_end = dt_valid.max().strftime("%Y-%m-%d") if not dt_valid.empty else "N/A"

    # Accounts (best-effort: since many parsers do not extract account_no, we treat as single account)
    accounts_count = 1
    total_tx = int(len(df))
    total_credit = float(df["credit"].sum())
    total_debit = float(df["debit"].sum())
    net_cash_flow = total_credit - total_debit

    # Group by statement period (source_file) as your rule
    rows = []
    for sf, g in df.groupby("source_file", sort=True):
        g = g.copy()
        g["_row"] = range(len(g))
        g = g.sort_values(["date_parsed", "page", "_row"], na_position="last", kind="mergesort")

        gb = g["balance"].dropna()
        opening = float(gb.iloc[0]) if not gb.empty else None
        closing = float(gb.iloc[-1]) if not gb.empty else None
        high = float(gb.max()) if not gb.empty else None
        low = float(gb.min()) if not gb.empty else None
        swing = (high - low) if (high is not None and low is not None) else None

        g_dates = g["date_parsed"].dropna()
        st = g_dates.min().strftime("%Y-%m-%d") if not g_dates.empty else None
        en = g_dates.max().strftime("%Y-%m-%d") if not g_dates.empty else None

        rows.append({
            "source_file": sf,
            "period": f"{st} - {en}" if st and en else "N/A",
            "opening": opening,
            "credit": float(g["credit"].sum()),
            "debit": float(g["debit"].sum()),
            "closing": closing,
            "high": high,
            "low": low,
            "swing": swing,
            "tx": int(len(g)),
        })

    months_count = len(rows)
    avg_monthly_credit = (total_credit / months_count) if months_count > 0 else None

    # Volatility index: Average swing / average monthly credit (best-effort)
    swings = [r["swing"] for r in rows if r["swing"] is not None]
    avg_swing = (sum(swings) / len(swings)) if swings else None
    volatility_pct = None
    if avg_swing is not None and avg_monthly_credit and avg_monthly_credit > 0:
        volatility_pct = (avg_swing / avg_monthly_credit) * 100.0
    volatility_label = _volatility_label(volatility_pct)

    # Round-figure flags: count large .00 values (both debit/credit) that are “round”
    # Conservative threshold: >= 5,000.00 and divisible by 1,000
    def is_round(val: float) -> bool:
        if val < 5000:
            return False
        return abs(val % 1000) < 0.01

    round_flags = 0
    round_examples = []
    for _, r in df.iterrows():
        amt = r["credit"] if r["credit"] > 0 else r["debit"]
        if amt > 0 and is_round(float(amt)):
            round_flags += 1
            if len(round_examples) < 30:
                round_examples.append({
                    "date": r["date"],
                    "description": r["description"],
                    "amount": amt,
                    "type": "CREDIT" if r["credit"] > 0 else "DEBIT",
                    "source_file": r["source_file"]
                })

    # Integrity score: balance continuity check
    # For each source_file, sort and validate:
    # current_balance ≈ prev_balance + credit - debit
    checks = 0
    passes = 0
    df_bal = df.dropna(subset=["balance"]).copy()
    if not df_bal.empty:
        for sf, g in df_bal.groupby("source_file", sort=True):
            g = g.copy()
            g["_row"] = range(len(g))
            g = g.sort_values(["date_parsed", "page", "_row"], na_position="last", kind="mergesort")
            prev = None
            for _, rr in g.iterrows():
                bal = float(rr["balance"])
                if prev is None:
                    prev = bal
                    continue
                expected = prev + float(rr["credit"]) - float(rr["debit"])
                # tolerance 0.05
                checks += 1
                if abs(expected - bal) <= 0.05:
                    passes += 1
                prev = bal

    integrity_pct = (passes / checks * 100.0) if checks > 0 else None
    integrity_css, integrity_label = _integrity_rating(integrity_pct)

    # Kite flying heuristic
    kite_pairs = _suspicious_kite_flying(df)
    kite_risk = "LOW RISK" if kite_pairs == 0 else ("MODERATE" if kite_pairs < 5 else "HIGH")

    # Categories (for charts)
    df["credit_cat"] = df.apply(lambda r: _classify_row(r["description"], True) if r["credit"] > 0 else None, axis=1)
    df["debit_cat"] = df.apply(lambda r: _classify_row(r["description"], False) if r["debit"] > 0 else None, axis=1)

    credit_cat = (
        df[df["credit"] > 0]
        .groupby("credit_cat")["credit"].sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    debit_cat = (
        df[df["debit"] > 0]
        .groupby("debit_cat")["debit"].sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    credit_cat_rows = [{"category": str(r["credit_cat"]), "amount": float(r["credit"])} for _, r in credit_cat.iterrows()]
    debit_cat_rows = [{"category": str(r["debit_cat"]), "amount": float(r["debit"])} for _, r in debit_cat.iterrows()]

    # Related party matching (optional)
    related_party_keywords = related_party_keywords or []
    rp_hits = []
    rp_total_in = 0.0
    rp_total_out = 0.0
    if related_party_keywords:
        kws = [k.strip() for k in related_party_keywords if k.strip()]
        if kws:
            rx = re.compile("|".join(re.escape(k) for k in kws), re.I)
            hit_df = df[df["description"].fillna("").apply(lambda s: bool(rx.search(s)))].copy()
            for _, rr in hit_df.iterrows():
                rp_total_in += float(rr["credit"])
                rp_total_out += float(rr["debit"])
                if len(rp_hits) < 200:
                    rp_hits.append({
                        "date": rr["date"],
                        "description": rr["description"],
                        "debit": float(rr["debit"]),
                        "credit": float(rr["credit"]),
                        "source_file": rr["source_file"],
                    })

    # Deterministic executive observations (no GPT)
    positives = []
    concerns = []

    if integrity_pct is not None:
        positives.append(f"Data integrity score: {integrity_pct:.1f}% ({integrity_label}).")
    else:
        concerns.append("Integrity score: N/A (insufficient balance continuity data).")

    positives.append(f"Net cash flow: RM {_fmt_money(net_cash_flow)}.")
    positives.append(f"Total credits: RM {_fmt_money(total_credit)} across {total_tx:,} transactions.")

    if volatility_pct is not None:
        if volatility_label in ("HIGH", "EXTREME"):
            concerns.append(f"Volatility index: {volatility_pct:.0f}% ({volatility_label}).")
        else:
            positives.append(f"Volatility index: {volatility_pct:.0f}% ({volatility_label}).")
    else:
        concerns.append("Volatility index: N/A (missing swings or monthly credits).")

    if round_flags > 0:
        concerns.append(f"Round figure flags detected: {round_flags} transaction(s).")

    if kite_pairs > 0:
        concerns.append(f"Potential circular flow matches: {kite_pairs} (kite flying heuristic).")

    # Recommendations (deterministic)
    recommendations = []
    if integrity_pct is None or (integrity_pct is not None and integrity_pct < 80):
        recommendations.append({"priority": "high", "category": "Data Quality", "text": "Review parser accuracy (date/balance/debit-credit) and reconcile against statement totals."})
    if volatility_label in ("HIGH", "EXTREME"):
        recommendations.append({"priority": "medium", "category": "Cash Flow", "text": "Investigate large month-to-month swings and identify drivers of volatility."})
    if round_flags > 0:
        recommendations.append({"priority": "medium", "category": "Transaction Review", "text": "Review round-figure transfers for related-party movement or non-operating flows."})
    if related_party_keywords and (rp_total_in + rp_total_out) > 0:
        recommendations.append({"priority": "low", "category": "Related Party", "text": "Separate related-party transactions from business turnover if required by policy."})

    # Machine JSON block (per your schema intent, annualized always computed)
    annualized_credit_turnover = (total_credit / months_count) * 12 if months_count > 0 else 0.0
    income_ratio = (annualized_credit_turnover / declared_sales) if declared_sales and declared_sales > 0 else 0.0

    gem_5_bank_data = {
        "GEM_5_BANK_DATA": {
            "Annualized Credit Turnover": round(float(annualized_credit_turnover), 2),
            "Income Ratio": round(float(income_ratio), 4),
            "Months Analyzed": int(months_count),
            "Total Inflow": round(float(total_credit), 2),
            "Average Swing": round(float(avg_swing), 2) if avg_swing is not None else 0.0
        }
    }

    # Build context for template
    context = {
        "company_name": company_name,
        "period_start": period_start,
        "period_end": period_end,
        "months_count": months_count,
        "accounts_count": accounts_count,
        "transactions_count": total_tx,
        "total_credits": total_credit,

        "integrity_pct": integrity_pct,
        "integrity_css": integrity_css,
        "integrity_label": integrity_label,

        "kite_pairs": kite_pairs,
        "kite_risk": kite_risk,

        "volatility_pct": volatility_pct,
        "volatility_label": volatility_label,

        "round_flags": round_flags,
        "round_examples": round_examples,

        "rows": rows,  # per source_file summaries

        "credit_categories": credit_cat_rows,
        "debit_categories": debit_cat_rows,

        "positives": positives,
        "concerns": concerns,
        "recommendations": recommendations,

        "related_party_keywords": related_party_keywords,
        "related_party_total_in": rp_total_in,
        "related_party_total_out": rp_total_out,
        "related_party_hits": rp_hits,

        "gem_5_bank_data": gem_5_bank_data,
    }

    return context
