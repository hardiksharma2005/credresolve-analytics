# -*- coding: utf-8 -*-
"""
CredResolve Analytics Dashboard
Run from credresolve-analytics/ root: python src/dashboard.py
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dash_table, dcc, html

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
P = "data/processed/"
A = "data/processed/analytics_outputs/"

# ── Color scheme ───────────────────────────────────────────────────────────────
PRIMARY  = "#2E86AB"
SUCCESS  = "#28A745"
WARNING  = "#FFC107"
DANGER   = "#DC3545"
DARK     = "#1a1a2e"
CARD_BG  = "#16213e"
TEXT     = "#e0e0e0"

# =============================================================================
# SECTION 1 -- Data Loading
# =============================================================================
print("Loading dashboard data...")

customers  = pd.read_csv(P + "customers_clean.csv",       parse_dates=["signup_date"])
mrr_ts     = pd.read_csv(A + "mrr_timeseries.csv")
mrr_seg    = pd.read_csv(P + "mrr_by_segment.csv")
mrr_reg    = pd.read_csv(P + "mrr_by_region.csv")
churn_seg  = pd.read_csv(P + "churn_by_segment.csv")
churn_reg  = pd.read_csv(P + "churn_by_region.csv")
csat_ts    = pd.read_csv(A + "csat_timeseries.csv")
feat_adop  = pd.read_csv(P + "feature_adoption.csv")
feat_mon   = pd.read_csv(A + "feature_adoption_monthly.csv")
cohort_hm  = pd.read_csv(A + "cohort_heatmap.csv")
camp_chan   = pd.read_csv(P + "campaign_by_channel.csv")
camp_seg_a = pd.read_csv(A + "campaign_by_segment_analytics.csv")
mkt_funnel = pd.read_csv(A + "marketing_funnel.csv")
supp_cat   = pd.read_csv(A + "support_category_analysis.csv")
tix_seg    = pd.read_csv(A + "tickets_by_segment.csv")
high_risk  = pd.read_csv(A + "high_risk_customers.csv")
top_custs  = pd.read_csv(A + "top_customers.csv")
ca_enrich  = pd.read_csv(A + "customer_analytics_enriched.csv",
                         parse_dates=["signup_date"])
usage_seg  = pd.read_csv(A + "usage_by_segment.csv")
res_mon    = pd.read_csv(A + "resolution_time_monthly.csv")
avg_ret    = pd.read_csv(A + "avg_retention_curve.csv")
tix_vol    = pd.read_csv(P + "ticket_volume_monthly.csv")
camps      = pd.read_csv(P + "marketing_clean.csv",
                         parse_dates=["start_date", "end_date"])

with open(A + "kpi_summary.json") as _f:
    kpi = json.load(_f)

# Compute churn_by_plan (not saved by ETL/analytics)
_tot  = ca_enrich.groupby("plan_type")["customer_id"].count().rename("total_customers")
_chu  = (ca_enrich[ca_enrich["status"] == "Churned"]
         .groupby("plan_type")["customer_id"].count().rename("churned_customers"))
churn_plan = pd.concat([_tot, _chu], axis=1).fillna(0).reset_index()
churn_plan["churned_customers"] = churn_plan["churned_customers"].astype(int)
churn_plan["churn_rate_pct"]    = (churn_plan["churned_customers"] /
                                   churn_plan["total_customers"] * 100).round(2)

all_months = sorted(mrr_ts["invoice_month"].unique().tolist())
print(f"  Loaded. {len(ca_enrich):,} customers | {len(all_months)} months of data.")


# =============================================================================
# SECTION 2 -- App Initialization
# =============================================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
           suppress_callback_exceptions=True)


# =============================================================================
# SECTION 3 -- Reusable Helper Functions
# =============================================================================
def make_kpi_card(title, value, subtitle, color, icon=""):
    label = f"{icon} {title}".strip() if icon else title
    return dbc.Card(
        dbc.CardBody([
            html.P(label, className="mb-1",
                   style={"color": "#aaa", "fontSize": "0.78rem", "fontWeight": "500"}),
            html.H4(str(value), style={"color": color, "fontWeight": "bold", "margin": "4px 0"}),
            html.P(str(subtitle),
                   style={"color": "#888", "fontSize": "0.73rem", "marginBottom": 0}),
        ]),
        style={"backgroundColor": CARD_BG, "border": f"1px solid {color}44", "height": "100%"}
    )


def format_currency(value):
    v = float(value)
    if v >= 1_000_000: return f"${v / 1_000_000:.2f}M"
    if v >= 1_000:     return f"${v / 1_000:.1f}K"
    return f"${v:.0f}"


def _dark_layout(fig, title="", height=350):
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=14)),
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        font=dict(color=TEXT),
        margin=dict(l=55, r=20, t=55, b=50),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        xaxis=dict(gridcolor="#2a2a4a", linecolor="#444", tickfont=dict(color=TEXT)),
        yaxis=dict(gridcolor="#2a2a4a", linecolor="#444", tickfont=dict(color=TEXT)),
    )


def make_time_series_fig(df, x_col, y_col, title, color=PRIMARY,
                          anomaly_col=None, rolling_col=None, height=350):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col], mode="lines", name=y_col,
        line=dict(color=color, width=2),
        hovertemplate=f"%{{x}}<br>Value: %{{y:,.2f}}<extra></extra>"
    ))
    if rolling_col and rolling_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[rolling_col], mode="lines", name="3M Avg",
            line=dict(color=WARNING, width=1.5, dash="dash"),
            hovertemplate=f"%{{x}}<br>3M Avg: %{{y:,.2f}}<extra></extra>"
        ))
    if anomaly_col and anomaly_col in df.columns:
        anom = df[df[anomaly_col].astype(bool)]
        if len(anom):
            fig.add_trace(go.Scatter(
                x=anom[x_col], y=anom[y_col], mode="markers", name="Anomaly",
                marker=dict(color=DANGER, size=9, symbol="circle"),
                hovertemplate=f"%{{x}}<br>ANOMALY: %{{y:,.2f}}<extra></extra>"
            ))
    _dark_layout(fig, title, height)
    return fig


def _graph_card(fig):
    return dbc.Card(
        dbc.CardBody(dcc.Graph(figure=fig, config={"displayModeBar": False})),
        style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a"}
    )


def _filter_ca(region, segment, plan):
    df = ca_enrich.copy()
    if region and "All" not in region:
        df = df[df["region"].isin(region)]
    if segment and "All" not in segment:
        df = df[df["segment"].isin(segment)]
    if plan and plan != "All":
        df = df[df["plan_type"] == plan]
    return df


# =============================================================================
# SECTION 4 -- Global Filter Bar
# =============================================================================
_region_opts = [{"label": "All", "value": "All"}] + [
    {"label": r, "value": r} for r in sorted(customers["region"].unique())]
_seg_opts    = [{"label": "All", "value": "All"}] + [
    {"label": s, "value": s} for s in sorted(customers["segment"].unique())]
_plan_opts   = [{"label": p, "value": p} for p in ["All", "Basic", "Pro", "Enterprise"]]
_n_mo        = len(all_months)
_mo_marks    = {0: {"label": all_months[0],  "style": {"color": TEXT, "fontSize": "0.7rem"}},
                _n_mo - 1: {"label": all_months[-1], "style": {"color": TEXT, "fontSize": "0.7rem"}}}

_dd_style = {"backgroundColor": "#0d1b2e", "color": "#000", "fontSize": "0.82rem"}

filter_bar = dbc.Row([
    dbc.Col([
        html.Label("Region", style={"color": "#aaa", "fontSize": "0.78rem"}),
        dcc.Dropdown(id="filter-region", options=_region_opts,
                     value=["All"], multi=True, style=_dd_style,
                     className="dbc"),
    ], md=2),
    dbc.Col([
        html.Label("Segment", style={"color": "#aaa", "fontSize": "0.78rem"}),
        dcc.Dropdown(id="filter-segment", options=_seg_opts,
                     value=["All"], multi=True, style=_dd_style,
                     className="dbc"),
    ], md=2),
    dbc.Col([
        html.Label("Plan Type", style={"color": "#aaa", "fontSize": "0.78rem"}),
        dcc.Dropdown(id="filter-plan", options=_plan_opts, value="All",
                     style=_dd_style, className="dbc"),
    ], md=2),
    dbc.Col([
        html.Label("Month Range", style={"color": "#aaa", "fontSize": "0.78rem"}),
        dcc.RangeSlider(
            id="filter-daterange", min=0, max=_n_mo - 1,
            value=[0, _n_mo - 1], marks=_mo_marks, step=1,
            tooltip={"always_visible": False, "placement": "bottom"},
        ),
    ], md=5, style={"paddingTop": "4px"}),
    dbc.Col([
        html.Br(),
        dbc.Button("Reset", id="btn-reset", color="secondary", size="sm",
                   style={"marginTop": "2px", "width": "100%"}),
    ], md=1),
], className="mb-3 py-2 px-3",
   style={"backgroundColor": CARD_BG, "borderRadius": "8px", "border": "1px solid #2a2a4a"})


# =============================================================================
# SECTION 5 -- Layout
# =============================================================================
_tab_style     = {"backgroundColor": CARD_BG, "color": "#aaa",
                  "borderTop": "none", "padding": "10px 16px", "fontSize": "0.88rem"}
_tab_sel_style = {"backgroundColor": PRIMARY,  "color": "#fff",
                  "borderTop": "none", "padding": "10px 16px", "fontSize": "0.88rem",
                  "fontWeight": "bold"}

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H3("CredResolve Analytics",
                        style={"color": PRIMARY, "fontWeight": "bold", "margin": 0}), md=8),
        dbc.Col(html.P("Executive Dashboard  |  June 2024",
                       style={"color": "#777", "textAlign": "right",
                              "margin": 0, "paddingTop": "10px", "fontSize": "0.85rem"}), md=4),
    ], className="mb-2 py-2", style={"borderBottom": f"2px solid {PRIMARY}"}),

    filter_bar,

    dcc.Tabs(id="main-tabs", value="tab-overview", children=[
        dcc.Tab(label="Overview",          value="tab-overview",
                style=_tab_style, selected_style=_tab_sel_style),
        dcc.Tab(label="Revenue",           value="tab-revenue",
                style=_tab_style, selected_style=_tab_sel_style),
        dcc.Tab(label="Churn & Retention", value="tab-churn",
                style=_tab_style, selected_style=_tab_sel_style),
        dcc.Tab(label="Marketing",         value="tab-marketing",
                style=_tab_style, selected_style=_tab_sel_style),
        dcc.Tab(label="Support & Product", value="tab-support",
                style=_tab_style, selected_style=_tab_sel_style),
    ], style={"marginBottom": "16px"}),

    html.Div(id="tab-content"),

], fluid=True, style={"backgroundColor": DARK, "minHeight": "100vh", "padding": "20px"})


# =============================================================================
# SECTION 6 -- Tab 1: Overview
# =============================================================================
def render_overview(region, segment, plan):
    df = _filter_ca(region, segment, plan)
    active_count = (df["status"] == "Active").sum()

    mrr_val   = kpi["mrr"]["current"]
    mom_pct   = kpi["mrr"]["mom_change_pct"]
    churn_val = kpi["churn"]["overall_rate_pct"]
    ret_val   = kpi["retention"]["overall_rate_pct"]
    csat_val  = kpi["support"]["avg_csat"]

    mom_color   = SUCCESS if mom_pct >= 0 else DANGER
    churn_color = DANGER  if churn_val > 25 else (WARNING if churn_val > 15 else SUCCESS)
    ret_color   = SUCCESS if ret_val  > 80 else WARNING
    csat_color  = SUCCESS if csat_val > 4  else (WARNING if csat_val > 3  else DANGER)

    # Row 1 -- KPI cards
    row1 = dbc.Row([
        dbc.Col(make_kpi_card("Current MRR",
                              format_currency(mrr_val),
                              f"MoM: {mom_pct:+.2f}%", mom_color), width=True),
        dbc.Col(make_kpi_card("Churn Rate",
                              f"{churn_val:.1f}%",
                              "Overall customers", churn_color), width=True),
        dbc.Col(make_kpi_card("Retention Rate",
                              f"{ret_val:.1f}%",
                              "Retained customers", ret_color), width=True),
        dbc.Col(make_kpi_card("Avg CSAT",
                              f"{csat_val:.2f} / 5",
                              "Customer satisfaction", csat_color), width=True),
        dbc.Col(make_kpi_card("Active Customers",
                              f"{active_count:,}",
                              "After filters", PRIMARY), width=True),
    ], className="mb-3 g-2")

    # Row 2 -- MRR time series + Churn by segment
    mrr_fig = make_time_series_fig(
        mrr_ts, "invoice_month", "total_mrr", "MRR Over Time",
        PRIMARY, "is_anomaly", "rolling_3m_avg"
    )
    mrr_fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")

    seg_colors_map = {"Enterprise": SUCCESS, "Mid-Market": WARNING, "SMB": DANGER}
    churn_seg_sorted = churn_seg.sort_values("churn_rate_pct")
    churn_fig = go.Figure(go.Bar(
        x=churn_seg_sorted["churn_rate_pct"],
        y=churn_seg_sorted["segment"],
        orientation="h",
        marker_color=[seg_colors_map.get(s, PRIMARY) for s in churn_seg_sorted["segment"]],
        text=churn_seg_sorted["churn_rate_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
    ))
    _dark_layout(churn_fig, "Churn Rate by Segment")
    churn_fig.update_layout(xaxis_title="Churn Rate %", xaxis_range=[0, 38])

    row2 = dbc.Row([
        dbc.Col(_graph_card(mrr_fig),   md=7),
        dbc.Col(_graph_card(churn_fig), md=5),
    ], className="mb-3 g-2")

    # Row 3 -- Feature adoption + Top 10 customers
    feat_s = feat_adop.sort_values("adoption_pct", ascending=True)
    feat_fig = go.Figure(go.Bar(
        x=feat_s["adoption_pct"], y=feat_s["feature"], orientation="h",
        marker_color=PRIMARY,
        text=feat_s["adoption_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
    ))
    _dark_layout(feat_fig, "Feature Adoption Rate (%)")
    feat_fig.update_layout(xaxis_title="% of Customers", xaxis_range=[0, 85])

    t10       = top_custs.head(10).sort_values("total_revenue")
    seg_cmap  = {"Enterprise": PRIMARY, "Mid-Market": WARNING, "SMB": SUCCESS}
    top10_fig = go.Figure(go.Bar(
        x=t10["total_revenue"], y=t10["customer_id"], orientation="h",
        marker_color=[seg_cmap.get(s, PRIMARY) for s in t10["segment"]],
        hovertemplate="%{y}<br>Revenue: $%{x:,.0f}<extra></extra>"
    ))
    _dark_layout(top10_fig, "Top 10 Customers by Revenue")
    top10_fig.update_layout(xaxis_tickprefix="$", xaxis_tickformat=",")

    row3 = dbc.Row([
        dbc.Col(_graph_card(feat_fig),  md=6),
        dbc.Col(_graph_card(top10_fig), md=6),
    ], className="mb-3 g-2")

    # Row 4 -- High risk table
    hr_cols  = ["customer_id", "segment", "region", "plan_type",
                "churn_risk_score", "total_revenue", "risk_tier"]
    hr_disp  = high_risk.head(20)[hr_cols].copy()
    hr_disp["total_revenue"] = hr_disp["total_revenue"].apply(lambda v: f"${v:,.0f}")

    risk_table = dash_table.DataTable(
        data=hr_disp.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in hr_cols],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#0a0a1a", "color": TEXT, "fontWeight": "bold",
                      "fontSize": "0.8rem"},
        style_cell={"backgroundColor": CARD_BG, "color": TEXT,
                    "border": "1px solid #2a2a4a", "fontSize": "0.78rem", "padding": "6px"},
        style_data_conditional=[
            {"if": {"filter_query": '{risk_tier} = "Critical"'},
             "backgroundColor": "#3d1010", "color": "#ff8080"},
            {"if": {"filter_query": '{risk_tier} = "High"'},
             "backgroundColor": "#3d2a10", "color": WARNING},
            {"if": {"filter_query": '{risk_tier} = "Medium"'},
             "backgroundColor": "#3d3910", "color": "#ffd700"},
        ]
    )
    row4 = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("High Risk Customers  (top 20 by score)",
                                   style={"color": DANGER, "margin": 0, "fontSize": "0.9rem"}),
                           style={"backgroundColor": CARD_BG, "border": "none"}),
            dbc.CardBody(risk_table),
        ], style={"backgroundColor": CARD_BG, "border": "1px solid #3d1010"}))
    ], className="mb-3")

    return html.Div([row1, row2, row3, row4])


# =============================================================================
# SECTION 7 -- Tab 2: Revenue
# =============================================================================
def render_revenue(region, segment, plan):
    df = _filter_ca(region, segment, plan)

    tot_life  = kpi["revenue"]["total_lifetime"]
    avg_rev   = kpi["revenue"]["avg_revenue_per_customer"]
    ytd_gr    = kpi["mrr"]["ytd_growth_pct"]
    nrr       = kpi["retention"]["nrr_pct"]

    row1 = dbc.Row([
        dbc.Col(make_kpi_card("Total Lifetime Revenue", format_currency(tot_life),
                              "All-time cumulative", SUCCESS), width=True),
        dbc.Col(make_kpi_card("Avg Rev / Customer", format_currency(avg_rev),
                              "Across all customers", PRIMARY), width=True),
        dbc.Col(make_kpi_card("MRR YTD Growth", f"{ytd_gr:+.2f}%",
                              "Jan-Jun 2024", SUCCESS if ytd_gr >= 0 else DANGER), width=True),
        dbc.Col(make_kpi_card("Net Revenue Retention", f"{nrr:.1f}%",
                              "Latest month NRR", SUCCESS if nrr >= 100 else WARNING), width=True),
    ], className="mb-3 g-2")

    # Row 2 -- MRR time series (wide) + Revenue by segment pie
    mrr_fig = make_time_series_fig(
        mrr_ts, "invoice_month", "total_mrr", "MRR Over Time",
        PRIMARY, "is_anomaly", "rolling_3m_avg", height=380
    )
    mrr_fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")

    rev_by_seg = mrr_seg.groupby("segment")["mrr"].sum().reset_index()
    pie_fig = px.pie(rev_by_seg, values="mrr", names="segment",
                     color_discrete_sequence=[PRIMARY, WARNING, SUCCESS],
                     hole=0.4)
    pie_fig.update_traces(textinfo="percent+label",
                          hovertemplate="%{label}: $%{value:,.0f}<extra></extra>")
    _dark_layout(pie_fig, "Revenue Share by Segment", height=380)

    row2 = dbc.Row([
        dbc.Col(_graph_card(mrr_fig), md=8),
        dbc.Col(_graph_card(pie_fig), md=4),
    ], className="mb-3 g-2")

    # Row 3 -- Revenue by region bar + Revenue by plan type bar
    rev_by_reg = mrr_reg.groupby("region")["mrr"].sum().reset_index().sort_values("mrr", ascending=False)
    reg_fig    = go.Figure(go.Bar(
        x=rev_by_reg["region"], y=rev_by_reg["mrr"],
        marker_color=PRIMARY,
        text=rev_by_reg["mrr"].apply(format_currency),
        textposition="outside",
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"
    ))
    _dark_layout(reg_fig, "Total Revenue by Region")
    reg_fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")

    rev_by_plan = (ca_enrich.groupby("plan_type")["total_revenue"]
                   .sum().reset_index().sort_values("total_revenue", ascending=False))
    plan_colors = {"Basic": WARNING, "Pro": PRIMARY, "Enterprise": SUCCESS}
    plan_fig    = go.Figure(go.Bar(
        x=rev_by_plan["plan_type"], y=rev_by_plan["total_revenue"],
        marker_color=[plan_colors.get(p, PRIMARY) for p in rev_by_plan["plan_type"]],
        text=rev_by_plan["total_revenue"].apply(format_currency),
        textposition="outside",
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"
    ))
    _dark_layout(plan_fig, "Total Revenue by Plan Type")
    plan_fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")

    row3 = dbc.Row([
        dbc.Col(_graph_card(reg_fig),  md=6),
        dbc.Col(_graph_card(plan_fig), md=6),
    ], className="mb-3 g-2")

    # Row 4 -- Revenue concentration (decile buckets)
    ca_s      = ca_enrich.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    total_all = ca_s["total_revenue"].sum()
    n         = len(ca_s)
    buckets   = []
    for i in range(10):
        lo_i = int(i * n / 10); hi_i = int((i + 1) * n / 10)
        buckets.append({
            "bucket": f"{i*10}-{(i+1)*10}%",
            "rev_pct": ca_s.iloc[lo_i:hi_i]["total_revenue"].sum() / total_all * 100,
        })
    conc_df  = pd.DataFrame(buckets)
    top20_sh = kpi["revenue"]["top_20pct_customers_revenue_share"]

    conc_fig = go.Figure(go.Bar(
        x=conc_df["bucket"], y=conc_df["rev_pct"],
        marker_color=[DANGER if i < 2 else (WARNING if i < 4 else PRIMARY)
                      for i in range(len(conc_df))],
        hovertemplate="Customers %{x}<br>Revenue share: %{y:.1f}%<extra></extra>"
    ))
    conc_fig.add_annotation(
        text=f"Top 20% customers = {top20_sh:.1f}% of revenue",
        xref="paper", yref="paper", x=0.98, y=0.97,
        showarrow=False, font=dict(color=DANGER, size=12),
        bgcolor="rgba(0,0,0,0.6)", bordercolor=DANGER, borderwidth=1
    )
    _dark_layout(conc_fig, "Revenue Concentration by Customer Percentile Bucket")
    conc_fig.update_layout(xaxis_title="Customer Percentile (top -> bottom)",
                           yaxis_title="% of Total Revenue")

    row4 = dbc.Row([dbc.Col(_graph_card(conc_fig))], className="mb-3")

    return html.Div([row1, row2, row3, row4])


# =============================================================================
# SECTION 8 -- Tab 3: Churn & Retention
# =============================================================================
def render_churn(region, segment, plan):
    row1 = dbc.Row([
        dbc.Col(make_kpi_card("Overall Churn",
                              f"{kpi['churn']['overall_rate_pct']:.2f}%",
                              "All segments combined", DANGER), width=True),
        dbc.Col(make_kpi_card("SMB Churn",
                              f"{kpi['churn']['smb_rate_pct']}%",
                              "Highest risk segment", DANGER), width=True),
        dbc.Col(make_kpi_card("Mid-Market Churn",
                              f"{kpi['churn']['midmarket_rate_pct']}%",
                              "Moderate risk", WARNING), width=True),
        dbc.Col(make_kpi_card("Enterprise Churn",
                              f"{kpi['churn']['enterprise_rate_pct']}%",
                              "Lowest churn", SUCCESS), width=True),
    ], className="mb-3 g-2")

    # Row 2 -- Churn by region + Churn by plan
    def _churn_color(rate):
        return DANGER if rate > 25 else (WARNING if rate > 15 else SUCCESS)

    reg_s   = churn_reg.sort_values("churn_rate_pct", ascending=False)
    reg_fig = go.Figure(go.Bar(
        x=reg_s["region"], y=reg_s["churn_rate_pct"],
        marker_color=[_churn_color(r) for r in reg_s["churn_rate_pct"]],
        text=reg_s["churn_rate_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
    ))
    _dark_layout(reg_fig, "Churn Rate by Region")
    reg_fig.update_layout(yaxis_title="Churn Rate %")

    plan_s   = churn_plan.sort_values("churn_rate_pct", ascending=False)
    plan_fig = go.Figure(go.Bar(
        x=plan_s["plan_type"], y=plan_s["churn_rate_pct"],
        marker_color=[_churn_color(r) for r in plan_s["churn_rate_pct"]],
        text=plan_s["churn_rate_pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
    ))
    _dark_layout(plan_fig, "Churn Rate by Plan Type")
    plan_fig.update_layout(yaxis_title="Churn Rate %")

    row2 = dbc.Row([
        dbc.Col(_graph_card(reg_fig),  md=6),
        dbc.Col(_graph_card(plan_fig), md=6),
    ], className="mb-3 g-2")

    # Row 3 -- Cohort heatmap (full width)
    period_cols = sorted([c for c in cohort_hm.columns if c.startswith("period_")])
    z_vals      = cohort_hm[period_cols].values.tolist()
    x_labels    = [c.replace("period_", "Q+") for c in period_cols]
    y_labels    = cohort_hm["cohort"].tolist()

    hm_fig = go.Figure(go.Heatmap(
        z=z_vals, x=x_labels, y=y_labels,
        colorscale=[[0, DANGER], [0.5, WARNING], [1, SUCCESS]],
        text=[[f"{v:.1f}%" for v in row] for row in z_vals],
        texttemplate="%{text}",
        showscale=True, zmin=0, zmax=100,
        hovertemplate="Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.1f}%<extra></extra>"
    ))
    _dark_layout(hm_fig, "Cohort Retention Heatmap  --  % of Customers Still Active", height=450)
    hm_fig.update_layout(margin=dict(l=110, r=20, t=60, b=60))

    row3 = dbc.Row([dbc.Col(_graph_card(hm_fig))], className="mb-3")

    # Row 4 -- Avg retention curve + High-risk by segment stacked bar
    ret_fig = make_time_series_fig(
        avg_ret, "period_number", "avg_retention_pct",
        "Average Retention Curve Across All Cohorts", PRIMARY
    )
    ret_fig.update_layout(xaxis_title="Quarters Since Signup", yaxis_title="Avg Retention %",
                          yaxis_range=[0, 105])

    risk_by_seg = (high_risk.groupby(["segment", "risk_tier"])["customer_id"]
                   .count().reset_index().rename(columns={"customer_id": "count"}))
    tier_colors = {"Critical": DANGER, "High": WARNING, "Medium": "#ffd700"}
    stk_fig     = px.bar(risk_by_seg, x="segment", y="count", color="risk_tier",
                          barmode="stack",
                          color_discrete_map=tier_colors,
                          labels={"count": "Customers", "risk_tier": "Risk Tier"})
    _dark_layout(stk_fig, "High-Risk Customers by Segment")

    row4 = dbc.Row([
        dbc.Col(_graph_card(ret_fig), md=6),
        dbc.Col(_graph_card(stk_fig), md=6),
    ], className="mb-3 g-2")

    return html.Div([row1, row2, row3, row4])


# =============================================================================
# SECTION 9 -- Tab 4: Marketing
# =============================================================================
def render_marketing(region, segment, plan):
    avg_roi  = kpi["marketing"]["avg_roi_pct"]
    best_ch  = kpi["marketing"]["best_channel"]
    tot_conv = kpi["marketing"]["total_conversions"]
    tot_rev  = kpi["marketing"]["total_campaign_revenue"]

    row1 = dbc.Row([
        dbc.Col(make_kpi_card("Avg Campaign ROI", f"{avg_roi:,.1f}%",
                              "Across all channels", SUCCESS if avg_roi > 0 else DANGER), width=True),
        dbc.Col(make_kpi_card("Best Channel", best_ch,
                              "By mean ROI", SUCCESS), width=True),
        dbc.Col(make_kpi_card("Total Conversions", f"{tot_conv:,}",
                              "All campaigns", PRIMARY), width=True),
        dbc.Col(make_kpi_card("Campaign Revenue", format_currency(tot_rev),
                              "Total generated", SUCCESS), width=True),
    ], className="mb-3 g-2")

    # Row 2 -- ROI by channel + Marketing funnel
    chan_s = camp_chan.sort_values("mean_roi", ascending=True)

    def _roi_color(r):
        return SUCCESS if r > 100 else (WARNING if r >= 0 else DANGER)

    roi_fig = go.Figure(go.Bar(
        x=chan_s["mean_roi"], y=chan_s["channel"], orientation="h",
        marker_color=[_roi_color(r) for r in chan_s["mean_roi"]],
        text=chan_s["mean_roi"].apply(lambda v: f"{v:,.0f}%"),
        textposition="outside",
        hovertemplate="%{y}: %{x:,.1f}%<extra></extra>"
    ))
    _dark_layout(roi_fig, "Mean ROI by Channel")
    roi_fig.update_layout(xaxis_title="Avg ROI %")

    fi  = mkt_funnel
    fn_fig = go.Figure(go.Funnel(
        y=["Impressions", "Clicks", "Leads", "Conversions"],
        x=[fi["total_impressions"].sum(), fi["total_clicks"].sum(),
           fi["total_leads"].sum(),       fi["total_conversions"].sum()],
        textinfo="value+percent initial",
        marker=dict(color=[PRIMARY, SUCCESS, WARNING, DANGER]),
        hovertemplate="%{label}: %{value:,}<extra></extra>"
    ))
    _dark_layout(fn_fig, "Marketing Funnel  (all channels combined)")

    row2 = dbc.Row([
        dbc.Col(_graph_card(roi_fig), md=6),
        dbc.Col(_graph_card(fn_fig),  md=6),
    ], className="mb-3 g-2")

    # Row 3 -- Campaign scatter + ROI by segment
    max_val = max(camps["cost"].max(), camps["revenue_generated"].max()) * 1.05
    sc_fig  = px.scatter(
        camps, x="cost", y="revenue_generated",
        size="conversions", color="channel",
        hover_data={"campaign_id": True, "roi": ":.1f"},
        size_max=28, opacity=0.8,
        labels={"cost": "Campaign Cost ($)", "revenue_generated": "Revenue Generated ($)"},
    )
    sc_fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode="lines", name="Break-even",
        line=dict(color="#555", dash="dash", width=1.5), showlegend=True
    ))
    _dark_layout(sc_fig, "Campaign Cost vs Revenue  (size = conversions)", height=380)
    sc_fig.update_layout(xaxis_tickprefix="$", xaxis_tickformat=",",
                         yaxis_tickprefix="$", yaxis_tickformat=",")

    seg_s   = camp_seg_a.sort_values("avg_roi", ascending=True)
    segs_fig = go.Figure(go.Bar(
        x=seg_s["avg_roi"], y=seg_s["target_segment"], orientation="h",
        marker_color=[_roi_color(r) for r in seg_s["avg_roi"]],
        text=seg_s["avg_roi"].apply(lambda v: f"{v:,.0f}%"),
        textposition="outside",
        hovertemplate="%{y}: %{x:,.1f}%<extra></extra>"
    ))
    _dark_layout(segs_fig, "Avg ROI by Target Segment", height=380)

    row3 = dbc.Row([
        dbc.Col(_graph_card(sc_fig),   md=7),
        dbc.Col(_graph_card(segs_fig), md=5),
    ], className="mb-3 g-2")

    # Row 4 -- Top campaigns table
    top10c = (camps.sort_values("roi", ascending=False)
              .head(10)[["campaign_id", "channel", "target_segment",
                          "cost", "revenue_generated", "roi", "conversions"]].copy())
    top10c["cost"]              = top10c["cost"].apply(lambda v: f"${v:,}")
    top10c["revenue_generated"] = top10c["revenue_generated"].apply(format_currency)
    top10c["roi"]               = top10c["roi"].apply(lambda v: f"{v:,.1f}%")
    top10c["conversions"]       = top10c["conversions"].apply(lambda v: f"{v:,}")

    camp_table = dash_table.DataTable(
        data=top10c.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in top10c.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#0a0a1a", "color": TEXT, "fontWeight": "bold",
                      "fontSize": "0.8rem"},
        style_cell={"backgroundColor": CARD_BG, "color": TEXT,
                    "border": "1px solid #2a2a4a", "fontSize": "0.78rem", "padding": "6px"},
    )
    row4 = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("Top 10 Campaigns by ROI",
                                   style={"color": SUCCESS, "margin": 0, "fontSize": "0.9rem"}),
                           style={"backgroundColor": CARD_BG, "border": "none"}),
            dbc.CardBody(camp_table),
        ], style={"backgroundColor": CARD_BG, "border": "1px solid #2a2a4a"}))
    ], className="mb-3")

    return html.Div([row1, row2, row3, row4])


# =============================================================================
# SECTION 10 -- Tab 5: Support & Product
# =============================================================================
def render_support(region, segment, plan):
    avg_res  = kpi["support"]["avg_resolution_hours"]
    avg_csat = kpi["support"]["avg_csat"]
    open_tix = kpi["support"]["open_tickets"]
    last30   = tix_vol.sort_values("ticket_month").iloc[-1]["ticket_count"]

    csat_c = SUCCESS if avg_csat > 4 else (WARNING if avg_csat > 3 else DANGER)

    row1 = dbc.Row([
        dbc.Col(make_kpi_card("Avg Resolution Time", f"{avg_res:.1f} hrs",
                              "Resolved tickets only", WARNING), width=True),
        dbc.Col(make_kpi_card("Avg CSAT Score", f"{avg_csat:.2f} / 5",
                              "Customer satisfaction", csat_c), width=True),
        dbc.Col(make_kpi_card("Open Tickets", f"{open_tix:,}",
                              "Awaiting resolution", DANGER), width=True),
        dbc.Col(make_kpi_card("Tickets (latest month)", f"{int(last30):,}",
                              "Latest month volume", PRIMARY), width=True),
    ], className="mb-3 g-2")

    # Row 2 -- CSAT trend + Resolution time by category
    csat_sorted = csat_ts.sort_values("ticket_month").reset_index(drop=True)
    csat_fig    = make_time_series_fig(
        csat_sorted, "ticket_month", "avg_csat",
        "CSAT Score Trend", PRIMARY, rolling_col="rolling_3m_avg"
    )
    # Red markers where CSAT < 3.0
    low_csat = csat_sorted[csat_sorted["avg_csat"] < 3.0]
    if len(low_csat):
        csat_fig.add_trace(go.Scatter(
            x=low_csat["ticket_month"], y=low_csat["avg_csat"],
            mode="markers", name="Low CSAT (<3)",
            marker=dict(color=DANGER, size=10, symbol="x"),
        ))
    csat_fig.update_layout(yaxis_range=[1, 5.5], yaxis_title="CSAT (1-5)",
                           shapes=[dict(type="line", x0=csat_sorted["ticket_month"].iloc[0],
                                        x1=csat_sorted["ticket_month"].iloc[-1],
                                        y0=3, y1=3,
                                        line=dict(color=DANGER, dash="dot", width=1))])

    # Mean resolution hours by category (from support_category_analysis)
    res_by_cat = (supp_cat.groupby("category")["avg_resolution_hours"]
                  .mean().reset_index().sort_values("avg_resolution_hours"))
    res_cat_fig = go.Figure(go.Bar(
        x=res_by_cat["avg_resolution_hours"], y=res_by_cat["category"],
        orientation="h", marker_color=WARNING,
        text=res_by_cat["avg_resolution_hours"].apply(lambda v: f"{v:.1f}h"),
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f} hrs<extra></extra>"
    ))
    _dark_layout(res_cat_fig, "Avg Resolution Time by Category")
    res_cat_fig.update_layout(xaxis_title="Hours")

    row2 = dbc.Row([
        dbc.Col(_graph_card(csat_fig),    md=6),
        dbc.Col(_graph_card(res_cat_fig), md=6),
    ], className="mb-3 g-2")

    # Row 3 -- Ticket volume by month + Tickets by segment
    tix_s = tix_vol.sort_values("ticket_month")
    vol_fig = go.Figure(go.Bar(
        x=tix_s["ticket_month"], y=tix_s["ticket_count"],
        marker_color=PRIMARY,
        hovertemplate="%{x}: %{y:,}<extra></extra>"
    ))
    _dark_layout(vol_fig, "Ticket Volume by Month")
    vol_fig.update_layout(xaxis_tickangle=-45, yaxis_title="Tickets")

    seg_fig = go.Figure(go.Bar(
        x=tix_seg["segment"],
        y=tix_seg["avg_tickets_per_customer"],
        marker_color=[seg_colors_map.get(s, PRIMARY) for s in tix_seg["segment"]],
        text=tix_seg["avg_tickets_per_customer"].apply(lambda v: f"{v:.1f}"),
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f} tickets/customer<extra></extra>"
    ))
    _dark_layout(seg_fig, "Avg Tickets per Customer by Segment")
    seg_fig.update_layout(yaxis_title="Avg Tickets / Customer")

    row3 = dbc.Row([
        dbc.Col(_graph_card(vol_fig), md=7),
        dbc.Col(_graph_card(seg_fig), md=5),
    ], className="mb-3 g-2")

    # Row 4 -- Feature usage by segment grouped bar + Segment x Feature heatmap
    features = sorted(usage_seg["feature"].unique())
    segments = sorted(usage_seg["segment"].unique())
    seg_pal  = [PRIMARY, WARNING, SUCCESS]

    usage_fig = go.Figure()
    for i, seg in enumerate(segments):
        d = usage_seg[usage_seg["segment"] == seg].set_index("feature")
        usage_fig.add_trace(go.Bar(
            name=seg,
            x=features,
            y=[d.loc[f, "mean_usage_count"] if f in d.index else 0 for f in features],
            marker_color=seg_pal[i % len(seg_pal)],
            hovertemplate=f"{seg} | %{{x}}: %{{y:.1f}}<extra></extra>"
        ))
    usage_fig.update_layout(barmode="group")
    _dark_layout(usage_fig, "Avg Feature Usage Count by Segment")
    usage_fig.update_layout(xaxis_tickangle=-20, legend_title="Segment")

    usage_pivot = usage_seg.pivot(index="segment", columns="feature",
                                   values="mean_usage_count").fillna(0)
    hm2_fig = go.Figure(go.Heatmap(
        z=usage_pivot.values.tolist(),
        x=usage_pivot.columns.tolist(),
        y=usage_pivot.index.tolist(),
        colorscale="Blues",
        text=[[f"{v:.1f}" for v in row] for row in usage_pivot.values.tolist()],
        texttemplate="%{text}",
        hovertemplate="Segment: %{y}<br>Feature: %{x}<br>Avg Usage: %{z:.1f}<extra></extra>",
        showscale=True,
    ))
    _dark_layout(hm2_fig, "Feature Usage Heatmap  (avg count by segment)", height=300)

    row4 = dbc.Row([
        dbc.Col(_graph_card(usage_fig), md=6),
        dbc.Col(_graph_card(hm2_fig),  md=6),
    ], className="mb-3 g-2")

    return html.Div([row1, row2, row3, row4])


# Segment color map for render_support (defined after render_overview uses it)
seg_colors_map = {"Enterprise": SUCCESS, "Mid-Market": WARNING, "SMB": DANGER}


# =============================================================================
# SECTION 11 -- Main Callback: Tab Routing
# =============================================================================
@app.callback(
    Output("tab-content", "children"),
    [Input("main-tabs",     "value"),
     Input("filter-region",  "value"),
     Input("filter-segment", "value"),
     Input("filter-plan",    "value")],
)
def render_tab(tab, region, segment, plan):
    # Normalize filter values
    if not region  or "All" in region:  region  = None
    if not segment or "All" in segment: segment = None
    if not plan    or plan == "All":    plan    = None

    if tab == "tab-overview":  return render_overview(region, segment, plan)
    if tab == "tab-revenue":   return render_revenue(region, segment, plan)
    if tab == "tab-churn":     return render_churn(region, segment, plan)
    if tab == "tab-marketing": return render_marketing(region, segment, plan)
    if tab == "tab-support":   return render_support(region, segment, plan)
    return html.Div("Unknown tab", style={"color": TEXT})


# =============================================================================
# SECTION 12 -- Reset Callback
# =============================================================================
@app.callback(
    [Output("filter-region",  "value"),
     Output("filter-segment", "value"),
     Output("filter-plan",    "value")],
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return ["All"], ["All"], "All"


# =============================================================================
# SECTION 13 -- Run
# =============================================================================
if __name__ == "__main__":
    print("Dashboard running at http://localhost:8050")
    app.run(debug=True, port=8050)
