"""
GridWatch — src/genai_reporter.py
=====================================
Generates plain-English risk reports using the Anthropic Claude API.
Turns model outputs into reports that policymakers and utilities can act on.

Setup: export ANTHROPIC_API_KEY=your_key_here

Author: Jaykumar Patel
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import anthropic

log = logging.getLogger(__name__)
BASE_DIR     = Path(__file__).parent.parent
REPORTS_DIR  = BASE_DIR / "reports" / "generated"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


SAMPLE_RISK_DATA = [
    {"state":"ME","county":"Cumberland County",   "risk_score":0.87,"risk_level":"HIGH",
     "customers_at_risk":145000,"avg_outage_hrs":9.2, "top_factor":"Ice Storms",       "infra_age":48,"prior_outages":5},
    {"state":"ME","county":"Kennebec County",     "risk_score":0.79,"risk_level":"HIGH",
     "customers_at_risk":62000, "avg_outage_hrs":7.1, "top_factor":"High Winds",        "infra_age":42,"prior_outages":3},
    {"state":"NH","county":"Hillsborough County", "risk_score":0.73,"risk_level":"MEDIUM-HIGH",
     "customers_at_risk":215000,"avg_outage_hrs":6.5, "top_factor":"Nor'easters",       "infra_age":38,"prior_outages":3},
    {"state":"MA","county":"Worcester County",    "risk_score":0.65,"risk_level":"MEDIUM",
     "customers_at_risk":410000,"avg_outage_hrs":5.8, "top_factor":"Population Density","infra_age":35,"prior_outages":2},
    {"state":"VT","county":"Chittenden County",   "risk_score":0.58,"risk_level":"MEDIUM",
     "customers_at_risk":88000, "avg_outage_hrs":11.3,"top_factor":"Rural Isolation",   "infra_age":31,"prior_outages":2},
]


def generate_report(region_scores: list[dict],
                    report_type: str = "executive") -> str:
    """
    Calls Claude API to generate a structured risk report.

    report_type options:
      "executive"  — for utility leadership and policymakers
      "technical"  — for engineers and operations managers
      "public"     — plain English for residents
    """
    audience = {
        "executive": "utility executives and senior government policymakers. "
                     "Focus on financial risk, strategic priorities, and ROI of prevention.",
        "technical": "utility engineers and operations managers. "
                     "Include technical failure modes, infrastructure specifics, "
                     "and operational recommendations.",
        "public":    "general public and residents. Plain English. "
                     "Be honest but reassuring. What should residents know and do?"
    }.get(report_type, "utility executives")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    prompt = f"""You are a power grid resilience expert analyzing outage risk data for the Northeast US.
Context: US power outages cost $121–150 billion annually (DOE/ORNL 2024).

RISK DATA:
{json.dumps(region_scores, indent=2)}

Write a {report_type.upper()} RISK REPORT with these sections:

1. EXECUTIVE SUMMARY (2 sentences max — overall picture)
2. TOP 3 HIGHEST-RISK REGIONS (county, why it's high risk, estimated impact)
3. PRIMARY RISK DRIVERS (top 3 factors driving risk regionally)
4. SEASONAL OUTLOOK (what the coming season means for grid stress)
5. PRIORITY RECOMMENDATIONS (3 specific, actionable steps for utility planners)
6. INFRASTRUCTURE INVESTMENT PRIORITIES (which areas need capital investment first and why)

Audience: {audience}
Today's date: {datetime.now().strftime('%B %d, %Y')}

Use specific numbers from the data. Be direct and actionable."""

    log.info(f"Generating {report_type} report via Claude API...")
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )

    report = msg.content[0].text

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"gridwatch_{report_type}_{ts}.txt"
    with open(path, "w") as f:
        f.write(f"GRIDWATCH RISK REPORT — {report_type.upper()}\n")
        f.write(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    log.info(f"Report saved → {path}")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    print(generate_report(SAMPLE_RISK_DATA, "executive"))
