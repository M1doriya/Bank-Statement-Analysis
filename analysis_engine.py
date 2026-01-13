# analysis_engine.py
# Deterministic credit-analysis engine for parsed bank-statement transactions.
# Implements rules from your Custom GPT instructions:
# - Group by source_file (each PDF = one statement period)
# - Compute Opening, Debit, Credit, Ending, High, Low, Swing per statement
# - Totals + GEM_5_BANK_DATA
# - OD rules + 6-month rule for HTML display (engine still computes annualized always)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _to_float(x: Any) -> float:
    try:
        if x is None or x == "":
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _parse_date_series(s: pd.Series) -> pd.Series:
    # We assume most parsers output ISO; still safe fallback.
    # IMPORTANT: do NOT force dayfirst for ISO.
    # Try ISO first, then generic.
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    fallback = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return iso.fillna(fallback)


@dataclass
class AnalysisInputs:
    od_limit: Optional[float] = None
    declared_sales: Optional[float] = None


@dataclass
class StatementSummary:
    source_file: str
    start_date: Optional[str]
    end_date: Optional[str]
    opening: Optional[float]
    total_debit: float
    total_credit: float
    ending: Optional[float]
    high: Optional[float]
    low: Optional[float]
    swing: Optional[float]
    tx_count: int


@dataclass
class AnalysisOutputs:
    statement_summaries: List[StatementSummary]
    months_count: int
    total_inflow: float
    total_debit: float
    net_cash_flow: float
    average_swing: float

    # OD flags
    has_negative_balance: bool
    od_limit_required: bool

    # GEM_5_BANK_DATA
    gem_5_bank_data: Dict[str, Any]

    # QA warnings
    warnings: List[str]


def build_analysis(transactions: List[Dict[str, Any]], inputs: AnalysisInputs) -> AnalysisOutputs:
    warnings: List[str] = []

    if not transactions:
        raise ValueError("No transactions provided to analysis engine.")

    df = pd.DataFrame(transactions).copy()

    # Ensure required columns exist
    for col in ["date", "description", "debit", "credit", "balance", "source_file"]:
        if col not in df.columns:
            df[col] = None

    df["source_file"] = df["source_file"].fillna("UNKNOWN_SOURCE")
    df["debit"] = df["debit"].apply(_to_float)
    df["credit"] = df["credit"].apply(_to_float)
    df["balance"] = df["balance"].apply(lambda x: float(x) if x is not None and str(x).strip() != "" else None)

    # Parse dates for ordering; keep rows with invalid dates but they won’t help ordering.
    df["date_parsed"] = _parse_date_series(df["date"])

    # Group by source_file (per your instruction)
    statement_summaries: List[StatementSummary] = []

    # OD detection needs all balances that exist
    balances_all = df["balance"].dropna()
    has_negative_balance = bool((balances_all < 0).any()) if not balances_all.empty else False

    for source_file, g in df.groupby("source_file", sort=True):
        g = g.copy()

        # Prefer stable ordering by date then page then original order
        if "page" in g.columns:
            g["_page"] = pd.to_numeric(g["page"], errors="coerce").fillna(10**9).astype(int)
        else:
            g["_page"] = 10**9

        g["_row"] = range(len(g))
        g = g.sort_values(["date_parsed", "_page", "_row"], na_position="last", kind="mergesort")

        # Dates
        valid_dates = g["date_parsed"].dropna()
        start_date = valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else None
        end_date = valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else None

        # Balances
        gb = g["balance"].dropna()
        opening = float(gb.iloc[0]) if not gb.empty else None
        ending = float(gb.iloc[-1]) if not gb.empty else None
        high = float(gb.max()) if not gb.empty else None
        low = float(gb.min()) if not gb.empty else None
        swing = (high - low) if (high is not None and low is not None) else None

        total_debit = float(g["debit"].sum())
        total_credit = float(g["credit"].sum())

        tx_count = int(len(g))

        statement_summaries.append(
            StatementSummary(
                source_file=str(source_file),
                start_date=start_date,
                end_date=end_date,
                opening=opening,
                total_debit=round(total_debit, 2),
                total_credit=round(total_credit, 2),
                ending=ending,
                high=high,
                low=low,
                swing=round(swing, 2) if swing is not None else None,
                tx_count=tx_count,
            )
        )

    months_count = len(statement_summaries)

    total_inflow = round(float(df["credit"].sum()), 2)
    total_debit = round(float(df["debit"].sum()), 2)
    net_cash_flow = round(total_inflow - total_debit, 2)

    swings = [s.swing for s in statement_summaries if s.swing is not None]
    average_swing = round(float(sum(swings) / len(swings)), 2) if swings else 0.0

    # OD rules
    od_limit = inputs.od_limit
    od_limit_required = has_negative_balance and (od_limit is None)

    # Annualized credit turnover ALWAYS computed for JSON engine output
    # Formula: (Total Credit / months_count) * 12
    if months_count > 0:
        annualized_credit_turnover = (total_inflow / months_count) * 12
    else:
        annualized_credit_turnover = 0.0
        warnings.append("No statement periods found; annualized credit turnover set to 0.0.")

    declared_sales = inputs.declared_sales
    if declared_sales is None or declared_sales == 0:
        income_ratio = 0.0
    else:
        income_ratio = annualized_credit_turnover / declared_sales

    gem_5_bank_data = {
        "GEM_5_BANK_DATA": {
            "Annualized Credit Turnover": round(float(annualized_credit_turnover), 2),
            "Income Ratio": round(float(income_ratio), 4),
            "Months Analyzed": int(months_count),
            "Total Inflow": round(float(total_inflow), 2),
            "Average Swing": round(float(average_swing), 2),
        }
    }

    # QA Warnings
    if months_count < 6:
        warnings.append(f"Coverage is {months_count}/6 months (HTML will show some ratios as N/A).")

    # Basic sanity checks
    if df["date_parsed"].isna().mean() > 0.20:
        warnings.append("More than 20% of transaction dates could not be parsed. Review date formats/parsers.")

    # Balance presence
    if df["balance"].isna().mean() > 0.50:
        warnings.append("More than 50% of balances are missing. OD logic and Ending/High/Low may be unreliable.")

    return AnalysisOutputs(
        statement_summaries=statement_summaries,
        months_count=months_count,
        total_inflow=total_inflow,
        total_debit=total_debit,
        net_cash_flow=net_cash_flow,
        average_swing=average_swing,
        has_negative_balance=has_negative_balance,
        od_limit_required=od_limit_required,
        gem_5_bank_data=gem_5_bank_data,
        warnings=warnings,
    )
