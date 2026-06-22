import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_APP_DATA = os.path.join(_ROOT, "data", "app")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Electricity Demand Forecasting — SMARD / Bundesnetzagentur",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (shared visual system) ─────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f1923; }
    [data-testid="stSidebar"] * { color: #e8edf2 !important; }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f8ef7 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        line-height: 1.15;
    }
    .hero-tagline {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        font-style: italic;
    }
    [data-testid="metric-container"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px !important;
    }
    [data-testid="metric-container"] label { color: #94a3b8 !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.8rem !important;
    }
    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px 3px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        border-left: 4px solid #4f8ef7;
        padding-left: 12px;
        margin: 1.5rem 0 1rem;
    }
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #1e293b;
    }
    .timeline-item {
        border-left: 3px solid #4f8ef7;
        padding: 0 0 16px 20px;
        position: relative;
    }
    .timeline-item::before {
        content: "";
        width: 12px; height: 12px;
        background: #4f8ef7;
        border-radius: 50%;
        position: absolute;
        left: -7px; top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_json(name: str) -> dict:
    with open(os.path.join(_APP_DATA, name)) as f:
        return json.load(f)


@st.cache_data
def load_forecast() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(_APP_DATA, "forecast_vs_actual.csv"),
                     parse_dates=["timestamp"])
    return df


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_APP_DATA, "feature_importance.csv"))


# ── Shared components ─────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def footer() -> None:
    st.markdown(
        '<div class="footer">'
        'Built by <strong>Utkarsh Rai</strong> &nbsp;|&nbsp; '
        'German Electricity Demand Forecasting &nbsp;|&nbsp; '
        'Data: SMARD.de (Bundesnetzagentur) &amp; Open-Meteo'
        '</div>',
        unsafe_allow_html=True,
    )


BADGE_COLORS = {
    "Python": ("#3b82f6", "#fff"),
    "CatBoost": ("#f97316", "#fff"),
    "XGBoost": ("#22c55e", "#fff"),
    "LightGBM": ("#84cc16", "#0f172a"),
    "Prophet": ("#a855f7", "#fff"),
    "MLflow": ("#06b6d4", "#fff"),
    "Streamlit": ("#ef4444", "#fff"),
    "Pandas": ("#eab308", "#0f172a"),
}


def tech_badges() -> None:
    html = ""
    for name, (bg, fg) in BADGE_COLORS.items():
        html += f'<span class="badge" style="background:{bg};color:{fg};">{name}</span>'
    st.markdown(html, unsafe_allow_html=True)


PLOT_BG = {"plot_bgcolor": "#0f172a", "paper_bgcolor": "#0f172a", "font_color": "#e2e8f0"}


# ── Page 1: Project Overview ──────────────────────────────────────────────────
def page_overview() -> None:
    results = load_json("model_results.json")
    summary = load_json("dataset_summary.json")
    bench = results["benchmark"]

    st.markdown(
        '<div class="hero-title">Electricity Demand Forecasting — '
        'SMARD · Bundesnetzagentur</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-tagline">'
        '"Predicting Germany\'s electricity demand 24 hours ahead'
        'and beating the grid operator\'s own forecast \u2014 a time-series analysis"</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        Germany's grid operators must predict national electricity demand a day in
        advance so the right amount of power is generated to ensure minimal risks blackouts and
        too much waste of resources such as money and fuel. This project builds a machine-learning model
        that forecasts hourly demand **24 hours ahead** and is **more accurate than the
        official forecast** the transmission operators publish.

        On an honest test of the first half of 2026 the data and model never saw during
        training — it predicts demand with an average error of just
        **{results['best_mape']:.2f}%**, compared with **{bench['same_holdout_mape']:.2f}%**
        for the official grid-operator forecast over the same period. The whole pipeline
        runs from public data (SMARD.de electricity data and Open-Meteo weather) through
        feature engineering, six forecasting models, and a leakage audit that keeps the
        numbers trustworthy.
        """
    )

    st.divider()

    page_header("Key Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model Error (MAPE)", f"{results['best_mape']:.2f}%",
              help="CatBoost, average % error on the 2026 hold-out the model never trained on")
    c2.metric("Grid Operator's Error", f"{bench['same_holdout_mape']:.2f}%",
              help="Official TSO day-ahead forecast, same 2026 period — the benchmark to beat")
    c3.metric("Improvement vs Operator", f"{results['improvement_vs_benchmark_pct']:.0f}%",
              help="How much lower the model's error is than the official forecast")
    c4.metric("Hours Forecast (test)", f"{summary['n_test_hours']:,}",
              help=f"{summary['test_start']} → {summary['test_end']}, hourly")

    st.caption(
        "MAPE = Mean Absolute Percentage Error: the average size of the forecast miss, "
        "as a percentage of actual demand. Lower is better. A 2.83% MAPE means the "
        "forecast is, on average, within 2.83% of what actually happened."
    )

    st.divider()

    page_header("Tech Stack")
    tech_badges()

    st.divider()

    page_header("What This Project Covers")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(
            "**Data & Features**\n\n"
            "Six years of hourly German grid load joined with weather data, turned into "
            "calendar, lag, and rolling features that capture the daily and weekly demand rhythm."
        )
    with col_b:
        st.info(
            "**Six Models Compared**\n\n"
            "From a simple last-week baseline through SARIMA and Prophet to gradient-boosted "
            "trees (LightGBM, XGBoost, CatBoost), each tested on the same unseen 2026 data."
        )
    with col_c:
        st.info(
            "**Leakage-Audited**\n\n"
            "A self-audit caught features that were secretly peeking at the answer. After "
            "fixing it, the honest model still beats the grid operator's forecast."
        )

    footer()


# ── Page 2: Explore the Data ──────────────────────────────────────────────────
def page_eda() -> None:
    page_header("Explore the Data",
                "What German electricity demand actually looks like and why the model works")

    profiles = load_json("eda_profiles.json")

    st.markdown(
        "Electricity demand is one of the most predictable time series in the world, "
        "because it follows human routines. The three patterns below are exactly what "
        "the model's features are built to capture and which is why a well-engineered model "
        "can forecast it so accurately."
    )

    st.divider()

    # ── Hour of day + Day of week ─────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        page_header("The Daily Rhythm")
        hod = profiles["hour_of_day"]
        hours = sorted(hod, key=lambda x: int(x))
        fig = go.Figure(go.Scatter(
            x=[int(h) for h in hours],
            y=[hod[h] for h in hours],
            mode="lines+markers",
            line={"color": "#4f8ef7", "width": 3},
            marker={"size": 6, "color": "#a78bfa"},
            hovertemplate="%{x}:00 — %{y:,.0f} MWh<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Hour of day", yaxis_title="Average load (MWh)",
            height=360, **PLOT_BG, margin={"t": 10},
        )
        fig.update_xaxes(gridcolor="#1e293b", dtick=4)
        fig.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Demand dips overnight, climbs to a mid-morning peak (~09:00), and rises again "
            "in the early evening (~18:00–20:00). This daily swing of ~20,000 MWh is why the "
            "**lag_24** feature demand exactly 24 hours earlier is so powerful."
        )

    with col2:
        page_header("The Weekly Rhythm")
        dow = profiles["day_of_week"]
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        vals = [dow[str(i)] for i in range(7)]
        colors = ["#4f8ef7"] * 5 + ["#f59e0b", "#ef4444"]
        fig = go.Figure(go.Bar(
            x=labels, y=vals, marker_color=colors,
            hovertemplate="%{x} — %{y:,.0f} MWh<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Day of week", yaxis_title="Average load (MWh)",
            height=360, **PLOT_BG, margin={"t": 10},
        )
        fig.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Weekday demand runs ~10–15% above Saturday and ~20% above Sunday, as factories "
            "and offices power down for the weekend. This is what the **lag_168** feature "
            "(demand exactly one week earlier) captures the same hour, same day-type."
        )

    st.divider()

    # ── Month / seasonal ──────────────────────────────────────────────────────
    page_header("The Seasonal Rhythm")
    mon = profiles["month"]
    mlabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mvals = [mon[str(i)] for i in range(1, 13)]
    fig = go.Figure(go.Bar(
        x=mlabels, y=mvals, marker_color="#4f8ef7",
        hovertemplate="%{x} — %{y:,.0f} MWh<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Month", yaxis_title="Average load (MWh)",
        height=360, **PLOT_BG, margin={"t": 10},
    )
    fig.update_yaxes(gridcolor="#1e293b")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Winter (Dec–Feb) runs ~10% above summer, driven by heating, shorter daylight, and "
        "sustained industrial activity. The **month** and **temperature** features let the "
        "model track this slower seasonal cycle."
    )

    st.divider()

    page_header("Key Findings")
    findings = [
        ("The daily cycle is the strongest signal",
         "Load peaks mid-morning and early evening with a ~20,000 MWh swing. The model's "
         "single most useful input is what demand was at this same hour 24 hours ago."),
        ("The weekly pattern is nearly as strong",
         "Weekday demand sits 10–20% above the weekend. Demand one week ago (lag_168) is the "
         "model's top predictor capturing both the hour and the day-type at once."),
        ("Demand shocks are visible in the history",
         "Spring 2020 shows a sustained drop from COVID-19 lockdowns; 2022 shows a "
         "structurally lower baseline as industry cut consumption during the gas-price crisis. "
         "These regime changes are exactly why classical models like SARIMA struggle here."),
        ("The grid operator is already very good",
         "The official day-ahead forecast is strong, so beating it required well-engineered "
         "weather, calendar, and holiday features and not just raw autoregressive structure."),
    ]
    for title, body in findings:
        st.info(f"**{title}**\n\n{body}")

    footer()


# ── Page 3: Model Results ─────────────────────────────────────────────────────
def page_model_results() -> None:
    page_header("Model Results",
                "Six models, one honest test, and the forecast that beats the grid operator")

    results = load_json("model_results.json")
    bench = results["benchmark"]
    fc = load_forecast()

    # ── The centerpiece: three-line overlay ───────────────────────────────────
    page_header("Forecast vs Actual vs Grid Operator")
    st.markdown(
        "This is the heart of the project. The chart compares actual demand (what really "
        "happened) against the model's 24-hour-ahead forecast and the official grid-operator "
        "forecast of all on the same hours. Use the slider to zoom into any window."
    )

    n = len(fc)
    default_end = min(24 * 14, n)
    rng = st.slider(
        "Window (hours from start of 2026 test set)",
        0, n, (0, default_end), step=24,
    )
    view = fc.iloc[rng[0]:rng[1]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view["timestamp"], y=view["actual"], mode="lines",
        name="Actual demand", line={"color": "#4f8ef7", "width": 2.5},
        hovertemplate="%{x}<br>Actual: %{y:,.0f} MWh<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=view["timestamp"], y=view["catboost"], mode="lines",
        name="Our model (CatBoost)", line={"color": "#22c55e", "width": 2, "dash": "dot"},
        hovertemplate="%{x}<br>Model: %{y:,.0f} MWh<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=view["timestamp"], y=view["tso"], mode="lines",
        name="Grid operator (TSO)", line={"color": "#f59e0b", "width": 1.5, "dash": "dash"},
        hovertemplate="%{x}<br>TSO: %{y:,.0f} MWh<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Time", yaxis_title="Grid load (MWh)",
        height=460, **PLOT_BG,
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(15,23,42,0.7)"},
        margin={"t": 10},
    )
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b")
    st.plotly_chart(fig, use_container_width=True)
    st.success(
        f"Across the full 2026 test set, the model's average error is "
        f"**{results['best_mape']:.2f}%** versus the grid operator's "
        f"**{bench['same_holdout_mape']:.2f}%** — a **{results['improvement_vs_benchmark_pct']:.0f}% "
        f"improvement** on the exact same hours. The green model line tracks the blue actual "
        f"line more tightly than the orange operator line, especially around the daily peaks."
    )

    st.divider()

    # ── Comparison table ──────────────────────────────────────────────────────
    page_header("How Every Model Did")
    st.markdown(
        "All six models were tested on the same unseen 2026 data. **A note on fairness:** "
        "the gradient-boosted models and the simple baseline are scored on *hourly* demand "
        "(the hard version), while SARIMA and Prophet were scored on *daily totals* (a smoother, "
        "easier target). Their numbers are shown for the modelling story, not as a head-to-head."
    )
    rows = []
    for m in results["models"]:
        rows.append({
            "Model": m["name"] + (" ★" if m["note"] == "best" else ""),
            "Error (MAPE)": "diverged" if m["mape"] is None else f"{m['mape']:.2f}%",
            "Tested on": m["resolution"],
            "Note": m["note"] if m["note"] != "best" else "best model",
        })
    rows.append({
        "Model": "Grid operator (TSO)",
        "Error (MAPE)": f"{bench['same_holdout_mape']:.2f}%",
        "Tested on": "hourly", "Note": "the benchmark to beat",
    })
    tbl = pd.DataFrame(rows)
    st.dataframe(
        tbl.style.apply(
            lambda r: ["background:#172554; font-weight:700" if "★" in str(r["Model"])
                       else ("background:#1a2e1a" if "TSO" in str(r["Model"]) else "")
                       for _ in r],
            axis=1,
        ),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Benchmark note: on the full 2020–2026 history the grid operator's MAPE is "
        f"~{bench['full_history_mape']:.2f}%; on the identical 2026 hold-out used to score the "
        f"models here it is {bench['same_holdout_mape']:.2f}%. The same-period figure is the "
        "fair comparison and the one used throughout this dashboard."
    )

    st.divider()

    # ── Why each classical model behaved as it did ────────────────────────────
    col_s, col_p = st.columns(2)
    with col_s:
        page_header("Why SARIMA Diverged")
        st.markdown(
            "**SARIMA** (Seasonal AutoRegressive Integrated Moving Average) is a classical "
            "statistical method from the 1970s, developed by statisticians Box and Jenkins. "
            "It assumes the seasonal patterns here, the weekly demand rhythm stays *stable* "
            "over time and projects it forward.\n\n"
            "On this data it diverged badly (its forecast drifted steadily downward). That is "
            "**expected, not a bug**: German demand went through structural shifts (COVID, the "
            "2022 energy crisis), and a model assuming a fixed pattern cannot track those. "
            "Documenting this failure is what justifies moving to tree-based models that make "
            "no such stability assumption."
        )
    with col_p:
        page_header("Where Prophet Fits")
        st.markdown(
            "**Prophet** is an open-source forecasting library released in 2017 by Facebook's "
            "Core Data Science team (now Meta). It splits a series into trend, seasonality, and "
            "holiday effects and is designed to be easy to use on business time series.\n\n"
            "It scored a respectable 3.70% but on *daily totals*, which smooth away the "
            "intraday peaks that make forecasting hard. That is not directly comparable to the "
            "hourly models. It is included to show the full modelling progression from simple "
            "to statistical to machine-learning approaches."
        )

    st.divider()

    # ── Leak audit ────────────────────────────────────────────────────────────
    page_header("The Leakage Audit — Why These Numbers Are Honest")
    st.markdown(
        "Mid-project, a self-audit found **data leakage**: some features were secretly using "
        "information from the exact hour being predicted is like peeking at the answer. This "
        "makes a model look better than it really is. Every leaking feature was fixed, the "
        "models were retrained, and the honest (slightly higher) errors are what you see above. "
        "Reporting the before/after openly is the point and it is the difference between a model "
        "that *looks* good and one that *is* good."
    )
    audit = results["leak_audit"]
    arows = [{
        "Model": a["model"],
        "Before (with leak)": f"{a['before']:.2f}%",
        "After (honest)": f"{a['after']:.2f}%",
        "Change": f"+{a['delta_pp']:.2f} pp",
    } for a in audit]
    st.dataframe(pd.DataFrame(arows), use_container_width=True, hide_index=True)
    st.info(
        "Every model still beats the grid operator after the fix. The leak was inflating the "
        "apparent accuracy by only 0.3–0.6 percentage points — the models were genuinely good; "
        "they just couldn't *prove* it until the leak was removed."
    )

    st.divider()

    # ── Feature importance ────────────────────────────────────────────────────
    page_header("What the Model Pays Attention To")
    fi = load_feature_importance().sort_values("importance")
    plain = {
        "lag_168": "Demand 1 week ago",
        "lag_24": "Demand 24 hours ago",
        "hour": "Hour of day",
        "rolling_mean_24h": "Avg demand, prior 24h",
        "day_of_week": "Day of week",
        "is_holiday": "Public holiday?",
        "is_weekend": "Weekend?",
        "month": "Month",
        "rolling_std_24h": "Demand volatility, prior 24h",
        "rolling_mean_168h": "Avg demand, prior week",
        "temp_2m": "Temperature",
        "temp_lag_24": "Temperature 24h ago",
        "temp_rolling_mean_72h": "Avg temperature, prior 72h",
    }
    fi["label"] = fi["feature"].map(plain).fillna(fi["feature"])
    fig = px.bar(
        fi, x="importance", y="label", orientation="h",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_layout(
        xaxis_title="Importance", yaxis_title="",
        height=440, **PLOT_BG, coloraxis_showscale=False, margin={"t": 10},
    )
    fig.update_xaxes(gridcolor="#1e293b")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Last week's demand and yesterday's demand dominates the model is, at heart, learning "
        "the daily and weekly rhythms seen on the previous page. Temperature contributes little, "
        "because for *demand* (unlike renewable *generation*) the calendar matters far more than "
        "the weather."
    )

    st.divider()

    page_header("Conclusion : What This Project Shows")
    st.markdown(
        """
        **The result.** Out of six forecasting approaches, the best model (CatBoost) predicts
        Germany's national electricity demand a full day ahead with an average error of just
        **2.83%**. On the exact same hours, the official forecast published by the grid operators —
        the people whose job this is — was off by **4.40%**. In other words, this model is about
        **36% more accurate than the professional benchmark**, built entirely from free, public data.

        **Why you can trust the number.** Partway through the project, an internal check caught a
        subtle mistake: a few of the model's inputs were accidentally "peeking"(data leak) at the very hour they
        were supposed to predict like a student who can see the answer key during the exam. That
        makes any model look better than it really is. Rather than quietly leave it in, the issue was
        found, fixed, and documented openly: the honest accuracy went from a flattering 2.24% to a
        truthful 2.83%. The slightly higher number is the *real* one and being able to show that
        difference, with a full audit trail, is what separates a forecast that merely looks good from
        one that actually holds up.

        **What it means in practice.** For a grid operator, sharper demand forecasts mean scheduling
        the right amount of power: less risk of shortfalls, less wasted fuel, lower costs, and fewer
        emissions across a roughly 60,000-MWh national load. Even a single percentage point of
        accuracy, sustained across every hour of the year, is operationally significant. This project
        shows that a transparent, reproducible, open-data pipeline can match and beat an
        established professional benchmark, while being honest about exactly how good it is and where
        its limits lie.
        """
    )

    footer()


# ── Page 4: How I Built This ──────────────────────────────────────────────────
def page_how_i_built() -> None:
    page_header("How I Built This", "Architecture, decisions, limitations, and what comes next")

    page_header("Pipeline Architecture")
    st.graphviz_chart("""
    digraph pipeline {
        rankdir=LR;
        node [shape=box style=filled fontname="Helvetica" fontsize=11];

        A [label="SMARD.de\\nHourly Load"   fillcolor="#1e3a5f" fontcolor="#93c5fd" color="#3b82f6"];
        W [label="Open-Meteo\\nWeather"      fillcolor="#1e3a5f" fontcolor="#93c5fd" color="#3b82f6"];
        B [label="Data Cleaner\\n(cleaner.py)" fillcolor="#1a2e1a" fontcolor="#86efac" color="#22c55e"];
        E [label="Feature Engineer\\n(engineer.py)" fillcolor="#1a2e1a" fontcolor="#86efac" color="#22c55e"];
        L [label="Leakage Audit\\n(horizon shift)" fillcolor="#2d1f00" fontcolor="#fcd34d" color="#f59e0b"];
        F [label="6 Models\\n(baseline→boosting)" fillcolor="#1e1e3a" fontcolor="#c4b5fd" color="#8b5cf6"];
        H [label="MLflow\\nTracking"         fillcolor="#001a1a" fontcolor="#67e8f9" color="#06b6d4"];
        I [label="Best Model\\n(CatBoost .pkl)"  fillcolor="#2d1414" fontcolor="#fca5a5" color="#ef4444"];
        J [label="Streamlit App\\n(this app)" fillcolor="#1e3a5f" fontcolor="#93c5fd" color="#3b82f6"];

        A -> B; W -> B; B -> E -> L -> F;
        F -> H [label="logs" fontsize=9];
        F -> I [label="saves" fontsize=9];
        I -> J;
    }
    """)

    st.divider()

    page_header("Build Timeline")
    timeline = [
        ("Step 1", "Data Pipeline",
         "Pulled six years of hourly German grid load from SMARD.de and matching weather "
         "from Open-Meteo, then cleaned and joined them on a shared hourly timeline."),
        ("Step 2", "Feature Engineering",
         "Built calendar features (hour, day-of-week, month, holidays), lag features "
         "(demand 24h and 168h ago), rolling averages, and temperature features."),
        ("Step 3", "Six-Model Comparison",
         "Ran a Seasonal-Naive baseline, SARIMA, Prophet, LightGBM, XGBoost, and CatBoost, "
         "each scored on the same 2026 hold-out with time-series cross-validation."),
        ("Step 4", "Leakage Audit & Fix",
         "Found that rolling and temperature features were peeking at the predicted hour. "
         "Shifted every such feature by the 24h forecast horizon and retrained honestly."),
        ("Step 5", "Tracking & Dashboard",
         "Logged every run with MLflow for reproducibility, then built this dashboard around "
         "the forecast-vs-actual-vs-operator comparison."),
    ]
    for step, title, desc in timeline:
        st.markdown(
            f'<div class="timeline-item">'
            f'<strong style="color:#4f8ef7">{step} — {title}</strong><br>'
            f'<span style="color:#94a3b8;font-size:0.9rem">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    col_d, col_l = st.columns(2)
    with col_d:
        page_header("Key Decisions")
        decisions = [
            ("Gradient-boosted trees over classical models",
             "SARIMA assumes a stable seasonal pattern and diverged on a series with COVID and "
             "energy-crisis regime shifts. Tree models make no such assumption and won clearly."),
            ("CatBoost as the production model",
             "All three boosting models beat the benchmark, but CatBoost had the lowest honest "
             "error (2.83%) and handles the categorical-style calendar features cleanly."),
            ("Keeping the diverged SARIMA in the lineup",
             "Documenting a model that failed and explaining why is more honest than quietly "
             "dropping it, and it justifies the progression to machine-learning methods."),
            ("Auditing for leakage before trusting any number",
             "A model that looks great because it peeks at the answer is worthless in production. "
             "The audit, and the slightly higher honest numbers, are the trustworthy result."),
        ]
        for title, body in decisions:
            with st.expander(title):
                st.write(body)

    with col_l:
        page_header("Lessons Learned")
        lessons = [
            ("Leakage hides in rolling windows",
             "A rolling average that includes the current hour silently leaks the target. The "
             "fix: shift the series by the forecast horizon first, it is simple but easy to miss."),
            ("Honest beats impressive",
             "The leaked model showed 2.24%; the honest one shows 2.83%. The honest number, with "
             "an audit trail, is far more valuable in an interview than the inflated one."),
            ("Match the benchmark to the test window",
             "The operator's headline 3.79% is a full-history figure. Compared fairly on the same "
             "2026 hours it is 4.40% which is always the benchmark on identical periods."),
            ("For demand, the calendar beats the weather",
             "Temperature barely moved the model. Human routines (hour, day, week) drive demand "
             "far more than weather which matters more for renewable generation."),
        ]
        for title, body in lessons:
            with st.expander(title):
                st.write(body)

    st.divider()

    # ── Limitations (separate section, as chosen) ─────────────────────────────
    page_header("Limitations")
    st.markdown(
        "Being clear about what this project does **not** do is part of doing it honestly:"
    )
    limitations = [
        ("Weather is observed, not forecasted",
         "In real operation you have a *weather forecast* for tomorrow, not the actual "
         "temperature. This project uses observed temperature shifted by 24 hours as a stand-in, "
         "because archived forecast runs were not available. Real-world error would be slightly "
         "higher though temperature barely affects this model, so the effect is small."),
        ("Tested on a single time window",
         "The model is scored on one continuous period (the first half of 2026). That is one "
         "realistic snapshot, not a distribution across many periods, so the 2.83% is a single "
         "result rather than a long-run average."),
        ("National aggregate only",
         "It forecasts total German demand, not regional breakdowns, the generation mix, or "
         "electricity prices. Those are separate, harder problems."),
        ("Atypical periods left in the training data",
         "COVID-2020 and the 2022 energy crisis are unusual regimes still present in the history. "
         "They were not down-weighted, which may slightly bias the model toward those conditions."),
    ]
    for title, body in limitations:
        st.warning(f"**{title}**\n\n{body}")

    st.divider()

    # ── Future work (separate section, as chosen) ─────────────────────────────
    page_header("Future Work")
    st.markdown("Where this project would go next, in rough priority order:")
    future = [
        ("Use real day-ahead weather forecasts",
         "Pull archived forecast runs from Open-Meteo's historical-forecast API so the model "
         "trains and tests on the same kind of weather data it would have in production."),
        ("Rolling-origin backtesting",
         "Re-score the model across many overlapping windows (not just H1 2026) to report an "
         "error *range* and confidence interval instead of a single number."),
        ("Probabilistic forecasts",
         "Predict not just a single demand number but a range (e.g. 10th–90th percentile), which "
         "grid operators need for reserve planning."),
        ("Hyperparameter tuning",
         "The boosting models use near-default settings. An Optuna search (as in the companion "
         "churn project) could squeeze out further accuracy."),
        ("Hourly Prophet for a fair comparison",
         "Re-run Prophet at hourly resolution so every model sits on one truly comparable axis."),
    ]
    for title, body in future:
        st.info(f"**{title}**\n\n{body}")

    st.divider()

    page_header("Project Links")
    lc, _, _ = st.columns(3)
    with lc:
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-View%20Source-181717?logo=github&style=for-the-badge)]"
            "(https://github.com/UtkarshRai-ds/german-electricity-demand-forecasting)"
        )
        st.caption("Full source code, the leakage audit, and reproducibility instructions.")

    footer()


# ── Sidebar navigation ────────────────────────────────────────────────────────
def main() -> None:
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#e2e8f0;margin-bottom:0.3rem'>⚡ Demand Forecasting</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#64748b;font-size:0.8rem;margin-top:0'>"
            "SMARD · Bundesnetzagentur · Time-Series Analysis</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        page = st.radio(
            "Navigate",
            options=[
                "🏠 Project Overview",
                "📊 Explore the Data",
                "🤖 Model Results",
                "🔧 How I Built This",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown(
            "<p style='color:#475569;font-size:0.75rem'>"
            "Built by <strong style='color:#94a3b8'>Utkarsh Rai</strong><br>"
            "German Electricity Demand · 2026"
            "</p>",
            unsafe_allow_html=True,
        )

    if page == "🏠 Project Overview":
        page_overview()
    elif page == "📊 Explore the Data":
        page_eda()
    elif page == "🤖 Model Results":
        page_model_results()
    elif page == "🔧 How I Built This":
        page_how_i_built()


if __name__ == "__main__":
    main()
