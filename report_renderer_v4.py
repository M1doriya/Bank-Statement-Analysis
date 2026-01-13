# report_renderer_v4.py
# Renders the v4-style interactive HTML using Jinja2 template + context dict.

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any
import json
import jinja2


def render_v4_html(context: Dict[str, Any], template_path: str = "templates/BankAnalysis_v4.html") -> str:
    env = jinja2.Environment(autoescape=True)
    with open(template_path, "r", encoding="utf-8") as f:
        template = env.from_string(f.read())

    # Inject chart data as JSON strings for Plotly in the template
    chart_payload = {
        "credit_categories": context.get("credit_categories", []),
        "debit_categories": context.get("debit_categories", []),
        "monthly_rows": context.get("rows", []),
    }

    return template.render(
        generated_at=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        ctx=context,
        chart_json=json.dumps(chart_payload),
        gem_json=json.dumps(context.get("gem_5_bank_data", {}), indent=2),
    )
