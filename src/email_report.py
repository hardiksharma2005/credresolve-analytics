# -*- coding: utf-8 -*-
"""
CredResolve Analytics - Executive Email Report Generator
Run from credresolve-analytics/ root: python src/email_report.py
"""
import base64
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

# ── Paths ───────────────────────────────────────────────────────────────────
P = "data/processed/"
A = "data/processed/analytics_outputs/"

# ── Color constants (same as dashboard) ─────────────────────────────────────
PRIMARY = "#2E86AB"
SUCCESS = "#28A745"
WARNING = "#FFC107"
DANGER  = "#DC3545"
DARK    = "#1a1a2e"

# =============================================================================
# SECTION 1 -- Data Loading
# =============================================================================
print("Loading data for email report...")

with open(A + "kpi_summary.json", encoding="utf-8") as _f:
    kpi = json.load(_f)

mrr_ts    = pd.read_csv(A + "mrr_timeseries.csv")
mrr_seg   = pd.read_csv(P + "mrr_by_segment.csv")
churn_seg = pd.read_csv(P + "churn_by_segment.csv")
churn_reg = pd.read_csv(P + "churn_by_region.csv")
csat_ts   = pd.read_csv(A + "csat_timeseries.csv")
feat_adop = pd.read_csv(P + "feature_adoption.csv")
cohort_hm = pd.read_csv(A + "cohort_heatmap.csv")
camp_chan  = pd.read_csv(P + "campaign_by_channel.csv")
mkt_funnel= pd.read_csv(A + "marketing_funnel.csv")
high_risk = pd.read_csv(A + "high_risk_customers.csv")
top_custs = pd.read_csv(A + "top_customers.csv")
ca_enrich = pd.read_csv(A + "customer_analytics_enriched.csv")
res_mon   = pd.read_csv(A + "resolution_time_monthly.csv")
supp_cat  = pd.read_csv(A + "support_category_analysis.csv")

print("  Data loaded.")


# =============================================================================
# SECTION 2 -- Chart Generation Helpers
# =============================================================================
def fig_to_base64(fig, width=600, height=350):
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
        encoded   = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:
        print(f"  [WARN] Chart export failed: {exc}")
        return ""


chart_style = dict(
    paper_bgcolor=DARK,
    plot_bgcolor="#0f3460",
    font=dict(color="#e0e0e0", size=11),
    margin=dict(l=40, r=20, t=45, b=40),
)


def apply_dark_style(fig):
    fig.update_layout(**chart_style)
    fig.update_layout(
        xaxis=dict(gridcolor="#1a3a5c", linecolor="#2a4a6a", tickfont=dict(color="#e0e0e0")),
        yaxis=dict(gridcolor="#1a3a5c", linecolor="#2a4a6a", tickfont=dict(color="#e0e0e0")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e0e0")),
    )
    return fig


def format_currency(value):
    v = float(value)
    if v >= 1_000_000: return f"${v / 1_000_000:.2f}M"
    if v >= 1_000:     return f"${v / 1_000:.1f}K"
    return f"${v:.0f}"


# =============================================================================
# SECTION 3 -- Generate All 8 Charts
# =============================================================================
print("Generating charts (this may take ~30 seconds with kaleido)...")

# -- Chart 1: MRR Trend -------------------------------------------------------
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=mrr_ts["invoice_month"], y=mrr_ts["total_mrr"],
    mode="lines", name="MRR",
    line=dict(color=PRIMARY, width=2),
    hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
))
fig1.add_trace(go.Scatter(
    x=mrr_ts["invoice_month"], y=mrr_ts["rolling_3m_avg"],
    mode="lines", name="3M Avg",
    line=dict(color=WARNING, width=1.5, dash="dash"),
))
anom = mrr_ts[mrr_ts["is_anomaly"].astype(bool)]
if len(anom):
    fig1.add_trace(go.Scatter(
        x=anom["invoice_month"], y=anom["total_mrr"],
        mode="markers", name="Anomaly",
        marker=dict(color=DANGER, size=8, symbol="circle"),
    ))
fig1.update_layout(title="Monthly Recurring Revenue Trend",
                   yaxis_tickprefix="$", yaxis_tickformat=",")
apply_dark_style(fig1)
mrr_ts_img = fig_to_base64(fig1)
print("  [1/8] MRR Trend")

# -- Chart 2: Churn by Segment ------------------------------------------------
_seg_colors = {"Enterprise": SUCCESS, "Mid-Market": WARNING, "SMB": DANGER}
churn_s = churn_seg.sort_values("churn_rate_pct")
fig2 = go.Figure(go.Bar(
    x=churn_s["churn_rate_pct"], y=churn_s["segment"],
    orientation="h",
    marker_color=[_seg_colors.get(s, PRIMARY) for s in churn_s["segment"]],
    text=churn_s["churn_rate_pct"].apply(lambda v: f"{v:.1f}%"),
    textposition="outside",
))
fig2.update_layout(title="Churn Rate by Customer Segment", xaxis_title="Churn Rate %",
                   xaxis_range=[0, 40])
apply_dark_style(fig2)
churn_segment_img = fig_to_base64(fig2)
print("  [2/8] Churn by Segment")

# -- Chart 3: Revenue by Segment Pie ------------------------------------------
rev_by_seg = mrr_seg.groupby("segment")["mrr"].sum().reset_index()
fig3 = px.pie(rev_by_seg, values="mrr", names="segment",
              color_discrete_sequence=[PRIMARY, SUCCESS, WARNING],
              hole=0.35, title="Revenue Distribution by Segment")
fig3.update_traces(textinfo="percent+label",
                   hovertemplate="%{label}: $%{value:,.0f}<extra></extra>")
apply_dark_style(fig3)
revenue_pie_img = fig_to_base64(fig3)
print("  [3/8] Revenue Pie")

# -- Chart 4: Feature Adoption ------------------------------------------------
feat_s = feat_adop.sort_values("adoption_pct", ascending=True)
fig4 = go.Figure(go.Bar(
    x=feat_s["adoption_pct"], y=feat_s["feature"],
    orientation="h",
    marker_color=PRIMARY,
    text=feat_s["adoption_pct"].apply(lambda v: f"{v:.1f}%"),
    textposition="outside",
))
fig4.update_layout(title="Product Feature Adoption Rate",
                   xaxis_title="% of Customers", xaxis_range=[0, 85])
apply_dark_style(fig4)
feature_adoption_img = fig_to_base64(fig4)
print("  [4/8] Feature Adoption")

# -- Chart 5: Cohort Heatmap --------------------------------------------------
period_cols = sorted([c for c in cohort_hm.columns if c.startswith("period_")])
z_vals      = cohort_hm[period_cols].values.tolist()
fig5 = go.Figure(go.Heatmap(
    z=z_vals,
    x=[c.replace("period_", "Q+") for c in period_cols],
    y=cohort_hm["cohort"].tolist(),
    colorscale=[[0, DANGER], [0.5, WARNING], [1.0, SUCCESS]],
    zmin=0, zmax=100, showscale=True,
    text=[[f"{v:.1f}%" for v in row] for row in z_vals],
    texttemplate="%{text}",
    hovertemplate="Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
))
fig5.update_layout(title="Cohort Retention Heatmap",
                   margin=dict(l=90, r=20, t=50, b=50))
apply_dark_style(fig5)
cohort_img = fig_to_base64(fig5, width=700, height=400)
print("  [5/8] Cohort Heatmap")

# -- Chart 6: CSAT Trend ------------------------------------------------------
fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x=csat_ts["ticket_month"], y=csat_ts["avg_csat"],
    mode="lines", name="CSAT",
    line=dict(color=SUCCESS, width=2),
))
if "rolling_3m_avg" in csat_ts.columns:
    fig6.add_trace(go.Scatter(
        x=csat_ts["ticket_month"], y=csat_ts["rolling_3m_avg"],
        mode="lines", name="3M Avg",
        line=dict(color=WARNING, width=1.5, dash="dash"),
    ))
# Dashed threshold line at CSAT = 3.0
fig6.update_layout(
    title="Customer Satisfaction Score Trend",
    yaxis_title="CSAT Score (1-5)", yaxis_range=[1, 5.5],
    shapes=[dict(
        type="line", x0=0, x1=1, y0=3.0, y1=3.0,
        xref="paper", yref="y",
        line=dict(color=DANGER, dash="dash", width=1.5),
    )],
    annotations=[dict(
        x=0.98, y=3.08, xref="paper", yref="y",
        text="Min 3.0", showarrow=False,
        font=dict(color=DANGER, size=10),
    )],
)
apply_dark_style(fig6)
csat_img = fig_to_base64(fig6)
print("  [6/8] CSAT Trend")

# -- Chart 7: Campaign ROI by Channel -----------------------------------------
def _roi_bar_color(r):
    return SUCCESS if r > 100 else (WARNING if r >= 0 else DANGER)

camp_s = camp_chan.sort_values("mean_roi", ascending=False)
fig7 = go.Figure(go.Bar(
    x=camp_s["channel"], y=camp_s["mean_roi"],
    marker_color=[_roi_bar_color(r) for r in camp_s["mean_roi"]],
    text=camp_s["mean_roi"].apply(lambda v: f"{v:,.0f}%"),
    textposition="outside",
))
fig7.update_layout(title="Campaign ROI by Channel", yaxis_title="Mean ROI %")
apply_dark_style(fig7)
campaign_roi_img = fig_to_base64(fig7)
print("  [7/8] Campaign ROI")

# -- Chart 8: Marketing Funnel ------------------------------------------------
fig8 = go.Figure(go.Funnel(
    y=["Impressions", "Clicks", "Leads", "Conversions"],
    x=[int(mkt_funnel["total_impressions"].sum()),
       int(mkt_funnel["total_clicks"].sum()),
       int(mkt_funnel["total_leads"].sum()),
       int(mkt_funnel["total_conversions"].sum())],
    textinfo="value+percent initial",
    marker=dict(color=[PRIMARY, SUCCESS, WARNING, DANGER]),
))
fig8.update_layout(title="Marketing Conversion Funnel")
apply_dark_style(fig8)
funnel_img = fig_to_base64(fig8)
print("  [8/8] Marketing Funnel")

print("  All charts generated.")


# =============================================================================
# SECTION 4 -- Compute Narrative Insights
# =============================================================================
print("Computing narrative insights...")

# Insight 1 -- SMB Churn
smb_active       = ca_enrich[(ca_enrich["segment"] == "SMB") & (ca_enrich["status"] == "Active")]
count_smb_active = len(smb_active)
smb_revenue      = float(smb_active["total_revenue"].sum())
smb_churn_rate   = float(kpi["churn"]["smb_rate_pct"])
smb_at_risk      = smb_revenue * (smb_churn_rate / 100)
insight_1_text   = (
    f"SMB segment churn rate stands at {smb_churn_rate:.1f}%, significantly above the company "
    f"average of {float(kpi['churn']['overall_rate_pct']):.1f}%. With {count_smb_active:,} active "
    f"SMB customers representing {format_currency(smb_revenue)} in cumulative revenue, targeted "
    f"retention campaigns could recover an estimated {format_currency(smb_at_risk)}."
)

# Insight 2 -- High Risk Accounts
count_critical  = len(high_risk[high_risk["risk_tier"] == "Critical"])
revenue_at_risk = float(kpi["risk"]["high_risk_revenue_at_stake"])
insight_2_text  = (
    f"{count_critical} customers are classified as Critical churn risk, with a combined lifetime "
    f"revenue of {format_currency(revenue_at_risk)} at stake. Immediate outreach to these accounts "
    f"is recommended before end of quarter."
)

# Insight 3 -- Feature Adoption
top_feature   = kpi["product"]["top_feature"]
top_pct       = kpi["product"]["top_feature_adoption_pct"]
low_row       = feat_adop.sort_values("adoption_pct").iloc[0]
low_feature   = low_row["feature"]
low_pct       = float(low_row["adoption_pct"])
insight_3_text = (
    f"'{top_feature}' leads product adoption at {top_pct}% of customers. However, '{low_feature}' "
    f"adoption remains at {low_pct:.1f}%, suggesting onboarding gaps or feature discoverability "
    f"issues that product and CS teams should investigate."
)

# Insight 4 -- Marketing Performance
best_channel  = kpi["marketing"]["best_channel"]
best_chan_row  = camp_chan[camp_chan["channel"] == best_channel]
best_roi      = float(best_chan_row["mean_roi"].iloc[0]) if len(best_chan_row) else 0.0
worst_chan_row = camp_chan.sort_values("mean_roi").iloc[0]
worst_channel = worst_chan_row["channel"]
worst_roi     = float(worst_chan_row["mean_roi"])
avg_roi_val   = float(kpi["marketing"]["avg_roi_pct"])
insight_4_text = (
    f"{best_channel} delivers the highest campaign ROI at {best_roi:.1f}%. The overall average "
    f"campaign ROI is {avg_roi_val:.1f}%. {worst_channel} is underperforming with {worst_roi:.1f}% "
    f"ROI -- budget reallocation toward {best_channel} and Content Marketing is recommended."
)

# Insight 5 -- CSAT & Support Quality
avg_csat_val  = float(kpi["support"]["avg_csat"])
cat_avg_sat   = supp_cat.groupby("category")["avg_satisfaction"].mean().reset_index()
worst_sat_row = cat_avg_sat.dropna().sort_values("avg_satisfaction").iloc[0]
worst_cat     = worst_sat_row["category"]
worst_score   = float(worst_sat_row["avg_satisfaction"])
cat_avg_res   = supp_cat.groupby("category")["avg_resolution_hours"].mean()
worst_res_time = float(cat_avg_res.get(worst_cat, 0))
insight_5_text = (
    f"Overall CSAT stands at {avg_csat_val}/5.00. The '{worst_cat}' ticket category shows the "
    f"lowest satisfaction score at {worst_score:.2f}/5.00 with an average resolution time of "
    f"{worst_res_time:.1f} hours. Prioritizing SLA improvements in this category will have the "
    f"highest customer satisfaction impact."
)

print("  Insights computed.")


# =============================================================================
# SECTION 5 -- Build Template Context
# =============================================================================
mrr_mom    = float(kpi["mrr"]["mom_change_pct"])
churn_r    = float(kpi["churn"]["overall_rate_pct"])
ret_r      = float(kpi["retention"]["overall_rate_pct"])
csat_v     = float(kpi["support"]["avg_csat"])
nrr_v      = float(kpi["retention"]["nrr_pct"])
ytd_growth = float(kpi["mrr"]["ytd_growth_pct"])

def _churn_color(r):
    return DANGER if r > 25 else (WARNING if r > 15 else SUCCESS)

hr_display = high_risk.head(10)[
    ["customer_id", "segment", "region", "plan_type", "churn_risk_score", "risk_tier"]
].copy()
hr_display["churn_risk_score"] = hr_display["churn_risk_score"].round(1)

tc_display = top_custs.head(5)[
    ["customer_id", "segment", "region", "plan_type", "total_revenue"]
].copy()
tc_display["total_revenue"] = tc_display["total_revenue"].apply(format_currency)

ctx = {
    # Report metadata
    "report_date":   "June 2024",
    "generated_at":  datetime.now().strftime("%B %d, %Y at %H:%M"),

    # KPI values & colors
    "mrr_current":              format_currency(kpi["mrr"]["current"]),
    "mrr_mom_pct":              f"{mrr_mom:+.1f}%",
    "mrr_color":                SUCCESS if mrr_mom >= 0 else DANGER,
    "churn_rate":               f"{churn_r:.1f}%",
    "churn_color":              _churn_color(churn_r),
    "retention_rate":           f"{ret_r:.1f}%",
    "retention_color":          SUCCESS if ret_r > 80 else WARNING,
    "csat_score":               f"{csat_v:.2f}",
    "csat_color":               SUCCESS if csat_v > 4 else (WARNING if csat_v > 3 else DANGER),
    "nrr_pct":                  f"{nrr_v:.1f}%",
    "nrr_color":                SUCCESS if nrr_v >= 100 else WARNING,
    "total_revenue":            format_currency(kpi["revenue"]["total_lifetime"]),
    "avg_revenue_per_customer": format_currency(kpi["revenue"]["avg_revenue_per_customer"]),
    "ytd_growth":               f"{ytd_growth:+.1f}%",
    "ytd_color":                SUCCESS if ytd_growth >= 0 else DANGER,
    "avg_roi":                  f"{avg_roi_val:.1f}%",
    "best_channel":             kpi["marketing"]["best_channel"],
    "high_risk_count":          kpi["risk"]["high_risk_customers"],
    "critical_risk_count":      kpi["risk"]["critical_risk_customers"],
    "revenue_at_risk":          format_currency(kpi["risk"]["high_risk_revenue_at_stake"]),
    "open_tickets":             kpi["support"]["open_tickets"],
    "avg_resolution_hrs":       f"{float(kpi['support']['avg_resolution_hours']):.1f}",

    # Base64 chart images
    "mrr_ts_img":           mrr_ts_img,
    "churn_segment_img":    churn_segment_img,
    "revenue_pie_img":      revenue_pie_img,
    "feature_adoption_img": feature_adoption_img,
    "cohort_img":           cohort_img,
    "csat_img":             csat_img,
    "campaign_roi_img":     campaign_roi_img,
    "funnel_img":           funnel_img,

    # Narrative insights
    "insight_1": insight_1_text,
    "insight_2": insight_2_text,
    "insight_3": insight_3_text,
    "insight_4": insight_4_text,
    "insight_5": insight_5_text,

    # Table data
    "high_risk_table":    hr_display.to_dict("records"),
    "top_customers_table": tc_display.to_dict("records"),
}


# =============================================================================
# SECTION 6 -- Render & Save
# =============================================================================
print("Rendering template...")

out_dir  = Path("output/reports")
out_dir.mkdir(parents=True, exist_ok=True)

env      = Environment(loader=FileSystemLoader("templates"), autoescape=False)
template = env.get_template("report_email.html")
html_out = template.render(**ctx)

out_path = out_dir / "executive_report_june2024.html"
out_path.write_text(html_out, encoding="utf-8")

size_kb = out_path.stat().st_size / 1024
print(f"Report generated: {out_path}")
print(f"File size: {size_kb:.1f} KB")
