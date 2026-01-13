# report_renderer.py
# Renders the HTML report using Jinja2 template and analysis outputs.

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import jinja2

from analysis_engine import AnalysisOutputs, AnalysisInputs


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "N/A"


def render_report_html(
    analysis: AnalysisOutputs,
    bank_name: str,
    inputs: AnalysisInputs,
    template_path: str = "templates/BankAnalysis_template.html",
) -> str:

    months_count = analysis.months_count

    # Coverage badge
    if months_count >= 6:
        badge_class = "good"
    elif months_count >= 4:
        badge_class = "warn"
    else:
        badge_class = "warn"

    # Executive summary (deterministic baseline; GPT can replace later)
    exec_summary = [
        f"Statements analyzed: {months_count}.",
        f"Total inflow: RM {_fmt_money(analysis.total_inflow)}; Total debit: RM {_fmt_money(analysis.total_debit)}.",
        f"Net cash flow: RM {_fmt_money(analysis.net_cash_flow)}.",
    ]

    # OD summary
    if not analysis.has_negative_balance:
        od_summary = ["No negative balances detected. Overdraft usage: N/A."]
    else:
        if inputs.od_limit is None:
            od_summary = ["Negative balances detected.", "OD Limit not provided: Utilization/Headroom shown as N/A – OD Limit Required."]
        else:
            od_summary = ["Negative balances detected.", f"OD Limit provided: RM {_fmt_money(inputs.od_limit)}. (Add utilization/headroom metrics if needed)"]

    # Statement rows
    statement_rows = []
    for s in analysis.statement_summaries:
        period = "N/A"
        if s.start_date and s.end_date:
            period = f"{s.start_date} to {s.end_date}"
        statement_rows.append(
            {
                "source_file": s.source_file,
                "period": period,
                "opening": _fmt_money(s.opening),
                "total_debit": _fmt_money(s.total_debit),
                "total_credit": _fmt_money(s.total_credit),
                "ending": _fmt_money(s.ending),
                "high": _fmt_money(s.high),
                "low": _fmt_money(s.low),
                "swing": _fmt_money(s.swing),
            }
        )

    # Ratios (6-month rule affects HTML display only)
    annualized_credit_display = "N/A (Requires 6 months)" if months_count < 6 else _fmt_money(analysis.gem_5_bank_data["GEM_5_BANK_DATA"]["Annualized Credit Turnover"])
    annualized_debit_display = "N/A (Requires 6 months)" if months_count < 6 else "N/A"  # you can implement if needed
    wcr_display = "N/A (Requires 6 months)" if months_count < 6 else "N/A"  # implement later if you have formula

    income_ratio_val = analysis.gem_5_bank_data["GEM_5_BANK_DATA"]["Income Ratio"]
    if inputs.declared_sales is None or inputs.declared_sales == 0:
        income_ratio_display = "N/A"
    else:
        income_ratio_display = f"{income_ratio_val:.4f}"

    ratios = [
        {"metric": "Annualized Credit Turnover", "value": annualized_credit_display},
        {"metric": "Annualized Debit", "value": annualized_debit_display},
        {"metric": "WCR", "value": wcr_display},
        {"metric": "Income Ratio", "value": income_ratio_display},
    ]

    observations = [
        "Review Data QA section for parser coverage and warnings.",
        "If OD limit is applicable, provide OD limit to enable utilization and headroom metrics.",
    ]

    # Load template
    env = jinja2.Environment(autoescape=True)
    with open(template_path, "r", encoding="utf-8") as f:
        template = env.from_string(f.read())

    html = template.render(
        header_title="Bank Statement Analysis Report",
        generated_at=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        bank_name=bank_name,
        months_count=months_count,
        coverage_badge_class=badge_class,
        total_inflow=f"RM {_fmt_money(analysis.total_inflow)}",
        total_debit=f"RM {_fmt_money(analysis.total_debit)}",
        net_cash_flow=f"RM {_fmt_money(analysis.net_cash_flow)}",
        average_swing=f"RM {_fmt_money(analysis.average_swing)}",
        exec_summary=exec_summary,
        od_summary=od_summary,
        statement_rows=statement_rows,
        ratios=ratios,
        observations=observations,
        warnings=analysis.warnings,
    )
    return html
