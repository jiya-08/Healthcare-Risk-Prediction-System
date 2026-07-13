import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics — HealthGuard AI",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Roboto:wght@300;400;500;700&display=swap');

    :root {
        /* ========================================================================= */
        /*  THEME COLOR PALETTE (EDIT THESE TO CHANGE THE GLOBAL LOOK)              */
        /* ========================================================================= */
        --theme-teal: #2EC4B6;        /* Primary Theme Color (Teal) - Sidebar Background */
        --theme-ice-blue: #CBF3F0;    /* Light Highlight/Hover Color (Ice Blue) */
        --theme-white: #FFFFFF;       /* Base clean white color */
        --theme-orange-lt: #FFBF69;   /* Hero Banner / Page Header Background (Light Orange) */
        --theme-orange-dk: #FF9F1C;   /* Accent Action Color (Bright Orange) - Buttons, active tabs */

        /* ========================================================================= */
        /*  DESIGN SYSTEM ELEMENT MAPPINGS                                          */
        /* ========================================================================= */
        --primary: var(--theme-orange-dk);       /* Primary Accent (Bright Orange) */
        --primary-light: var(--theme-ice-blue);  /* Primary Light (Ice Blue) */
        --primary-dark: #E68A00;                 
        --success: #34A853;                      
        --success-light: #E6F4EA;   
        --danger: #EA4335;                       
        --danger-light: #FCE8E6;    
        --warning: #FBBC04;         
        --warning-light: #FEF7E0;   
        --bg-primary: #F8F9FA;                   
        --bg-card: var(--theme-white);           
        --text-primary: #202124;                 
        --text-secondary: #5F6368;               
        --border: #DADCE0;                       
        --border-light: #E8EAED;    
        --shadow-sm: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
        --shadow-md: 0 1px 3px 0 rgba(60,64,67,0.3), 0 4px 8px 3px rgba(60,64,67,0.15);
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-full: 9999px;
        --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        --font-sans: 'Inter', 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: var(--font-sans);
    }

    h1, h2, h3, h4, h5, h6, p, label, .stMetric, .stSelectbox, .stNumberInput, .stTextInput, .stButton button, .stTabs [data-baseweb="tab"] {
        font-family: var(--font-sans) !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--theme-teal) !important; /* Sidebar Background: Teal (#2EC4B6) */
    }

    /* Collapse button icon color */
    section[data-testid="stSidebar"] button svg,
    section[data-testid="stSidebar"] button svg * {
        fill: var(--theme-white) !important;      /* Color: White (#FFFFFF) */
        color: var(--theme-white) !important;
    }
    
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] p {
        color: var(--theme-white) !important;     /* Text Color: White (#FFFFFF) */
    }

    /* Streamlit Sidebar Page Navigation Links */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span {
        color: var(--theme-white) !important;     /* Link text color: White (#FFFFFF) */
        font-weight: 500 !important;
        transition: var(--transition);
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
        background-color: rgba(255, 255, 255, 0.15) !important; /* Semi-transparent white */
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: rgba(255, 255, 255, 0.25) !important; /* Active link indicator background */
        font-weight: 600 !important;
    }

    /* High contrast inputs inside sidebar */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] input {
        background-color: var(--theme-white) !important; /* Input background: White */
        color: #1E293B !important;                       /* Input text: Dark Slate */
        border: 1px solid #DADCE0 !important;
        border-radius: 8px !important;
    }

    /* Dropdown popup option readability */
    div[data-baseweb="popover"] *,
    div[role="listbox"] * {
        color: #1E293B !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-md);
        padding: 1.25rem 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        font-size: 1.75rem !important;
    }

    /* Charts */
    .stPlotlyChart {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-md);
        padding: 0.5rem;
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }
    .stPlotlyChart:hover { box-shadow: var(--shadow-md); }

    /* ── Tabs ────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        padding: 4px 4px 0px 4px !important;
        box-shadow: none !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #F1F3F4 !important;
        border-radius: 8px 8px 0px 0px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important;
        border-bottom: none !important;
        transition: var(--transition) !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E8EAED !important;
        color: var(--primary) !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-color: var(--primary) !important;
        box-shadow: none !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Page Header */
    .page-header {
        background: var(--theme-orange-lt) !important; /* Page Header Background: Light Orange (#FFBF69) */
        border-radius: var(--radius-lg);
        padding: 2rem 2.5rem;
        color: var(--theme-white) !important;          /* Header Text: White (#FFFFFF) */
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-md);
        margin-bottom: 1.5rem;
    }
    .page-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -15%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 70%);
        border-radius: 50%;
    }
    .page-header h1 {
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.3px;
        position: relative;
        z-index: 1;
    }
    .page-header p {
        opacity: 0.85;
        margin: 0;
        font-size: 0.95rem;
        position: relative;
        z-index: 1;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-light);
    }

    /* ── Scrollbars ─────────────────────────────────────────────────── */
    ::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
    }
    ::-webkit-scrollbar-track {
        background: #F1F3F4 !important;
        border-radius: var(--radius-full) !important;
    }
    ::-webkit-scrollbar-thumb {
        background: #C1C7D0 !important;
        border-radius: var(--radius-full) !important;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #A6B0C0 !important;
    }

    /* Sidebar specific scrollbar styling */
    section[data-testid="stSidebar"] ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.15) !important;
        border-radius: var(--radius-full) !important;
    }
    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.25) !important;
        border-radius: var(--radius-full) !important;
    }
    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.4) !important;
    }

    /* Responsive */
    @media (max-width: 640px) {
        .page-header h1 { font-size: 1.3rem !important; }
        div[data-testid="stMetric"] { padding: 0.75rem 1rem; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.25rem !important; }
    }
    @media (min-width: 1441px) {
        .stMainBlockContainer { max-width: 1400px; margin: 0 auto; }
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    patients = pd.read_csv("patients.csv")
    diagnoses = pd.read_csv("diagnoses.csv")
    labs = pd.read_csv("labs.csv")
    outcomes = pd.read_csv("outcomes.csv")

    df = patients.merge(diagnoses, on="DiagnosisID", how="left")
    df = df.merge(outcomes, on="OutcomeID", how="left")
    df = df.merge(labs, left_on="PatientID", right_on="PatientID", how="left")

    df["AdmissionDate"] = pd.to_datetime(df["AdmissionDate"], errors="coerce")
    df["DischargeDate"] = pd.to_datetime(df["DischargeDate"], errors="coerce")
    df["LengthOfStay"] = (df["DischargeDate"] - df["AdmissionDate"]).dt.days.abs()

    return patients, diagnoses, labs, outcomes, df


patients_df, diagnoses_df, labs_df, outcomes_df, merged_df = load_data()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 1rem 0.5rem;">
        <h2 style="font-size:1.4rem;font-weight:800;margin:0;">🛡️ HealthGuard AI</h2>
        <p style="font-size:0.75rem;opacity:0.7;text-transform:uppercase;letter-spacing:1.5px;margin:0.25rem 0 0;">Analytics Console</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<p style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;opacity:0.5;margin:0 0 0.5rem 0;">📊 Filter Options</p>', unsafe_allow_html=True)

    age_range = st.slider(
        "Age Range",
        min_value=int(patients_df["Age"].min()),
        max_value=int(patients_df["Age"].max()),
        value=(int(patients_df["Age"].min()), int(patients_df["Age"].max()))
    )

    selected_diagnoses = st.multiselect(
        "Diagnoses",
        options=diagnoses_df["DiagnosisName"].tolist(),
        default=diagnoses_df["DiagnosisName"].tolist()
    )

    selected_outcomes = st.multiselect(
        "Outcomes",
        options=outcomes_df["OutcomeName"].tolist(),
        default=outcomes_df["OutcomeName"].tolist()
    )

# Apply filters
filtered_df = merged_df[
    (merged_df["Age"] >= age_range[0]) &
    (merged_df["Age"] <= age_range[1]) &
    (merged_df["DiagnosisName"].isin(selected_diagnoses)) &
    (merged_df["OutcomeName"].isin(selected_outcomes))
]


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>📊 Healthcare Analytics</h1>
    <p>Comprehensive data exploration and statistical insights across all patient records, diagnoses, and outcomes.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KEY METRICS
# ─────────────────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric("Patients (Filtered)", f"{len(filtered_df):,}")
with m2:
    st.metric("Avg Age", f"{filtered_df['Age'].mean():.0f} yrs" if len(filtered_df) > 0 else "—")
with m3:
    st.metric("Avg Cost", f"${filtered_df['TreatmentCost'].mean():,.0f}" if len(filtered_df) > 0 else "—")
with m4:
    st.metric("Avg Stay", f"{filtered_df['LengthOfStay'].mean():.0f} days" if len(filtered_df) > 0 else "—")
with m5:
    if len(filtered_df) > 0:
        mortality = (filtered_df["OutcomeName"] == "Deceased").sum() / len(filtered_df) * 100
        st.metric("Mortality Rate", f"{mortality:.1f}%")
    else:
        st.metric("Mortality Rate", "—")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "👥  Demographics",
    "🏥  Clinical Outcomes",
    "🔬  Lab Analysis",
    "💰  Cost Analysis"
])

# ── Demographics Tab ─────────────────────────────────────────────────────────
with tab1:
    d1, d2 = st.columns(2)

    with d1:
        fig = px.histogram(
            filtered_df, x="Age", nbins=15,
            color_discrete_sequence=["#1A73E8"],
            title="Age Distribution"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            title=dict(font=dict(size=16, weight=700)),
            xaxis=dict(title="Age (years)", gridcolor="#E8EAED"),
            yaxis=dict(title="Count", gridcolor="#E8EAED"),
            height=380, margin=dict(t=50, b=50, l=50, r=30), bargap=0.05
        )
        fig.update_traces(marker=dict(cornerradius=4))
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        gender_counts = filtered_df.groupby("Gender").size().reset_index(name="Count")
        gender_counts["Gender"] = gender_counts["Gender"].map({"M": "Male", "F": "Female"})
        fig = px.pie(
            gender_counts, names="Gender", values="Count", hole=0.5,
            color="Gender",
            color_discrete_map={"Male": "#1A73E8", "Female": "#E040FB"},
            title="Gender Distribution"
        )
        fig.update_layout(
            font=dict(family="Inter"),
            title=dict(font=dict(size=16, weight=700)),
            height=380, margin=dict(t=50, b=50, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        fig.update_traces(textinfo="percent+label", textfont=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

    # Age by Diagnosis box plot
    fig_box = px.box(
        filtered_df, x="DiagnosisName", y="Age",
        color="DiagnosisName",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Age Distribution by Diagnosis"
    )
    fig_box.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        title=dict(font=dict(size=16, weight=700)),
        xaxis=dict(title="", tickangle=-25, gridcolor="#E8EAED"),
        yaxis=dict(title="Age (years)", gridcolor="#E8EAED"),
        height=400, margin=dict(t=50, b=80, l=50, r=30),
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ── Clinical Outcomes Tab ────────────────────────────────────────────────────
with tab2:
    o1, o2 = st.columns(2)

    with o1:
        outcome_diag = filtered_df.groupby(["DiagnosisName", "OutcomeName"]).size().reset_index(name="Count")
        fig = px.bar(
            outcome_diag, x="DiagnosisName", y="Count", color="OutcomeName",
            barmode="group",
            color_discrete_map={"Recovered": "#34A853", "Complicated": "#FBBC04", "Deceased": "#EA4335"},
            title="Outcomes by Diagnosis"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            title=dict(font=dict(size=16, weight=700)),
            xaxis=dict(title="", tickangle=-30, gridcolor="#E8EAED"),
            yaxis=dict(title="Count", gridcolor="#E8EAED"),
            height=400, margin=dict(t=50, b=80, l=50, r=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            bargap=0.2
        )
        fig.update_traces(marker=dict(cornerradius=4))
        st.plotly_chart(fig, use_container_width=True)

    with o2:
        outcome_counts = filtered_df.groupby("OutcomeName").size().reset_index(name="Count")
        fig = px.pie(
            outcome_counts, names="OutcomeName", values="Count", hole=0.55,
            color="OutcomeName",
            color_discrete_map={"Recovered": "#34A853", "Complicated": "#FBBC04", "Deceased": "#EA4335"},
            title="Overall Outcome Distribution"
        )
        fig.update_layout(
            font=dict(family="Inter"),
            title=dict(font=dict(size=16, weight=700)),
            height=400, margin=dict(t=50, b=50, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        fig.update_traces(textinfo="percent+label", textfont=dict(family="Inter"))
        st.plotly_chart(fig, use_container_width=True)

    # Mortality rate by diagnosis
    mort_by_diag = filtered_df.groupby("DiagnosisName").apply(
        lambda x: (x["OutcomeName"] == "Deceased").sum() / len(x) * 100 if len(x) > 0 else 0
    ).reset_index(name="MortalityRate")
    mort_by_diag = mort_by_diag.sort_values("MortalityRate", ascending=True)

    fig_mort = px.bar(
        mort_by_diag, y="DiagnosisName", x="MortalityRate", orientation="h",
        color="MortalityRate",
        color_continuous_scale=["#E6F4EA", "#FBBC04", "#EA4335"],
        title="Mortality Rate by Diagnosis (%)"
    )
    fig_mort.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        title=dict(font=dict(size=16, weight=700)),
        xaxis=dict(title="Mortality Rate (%)", gridcolor="#E8EAED"),
        yaxis=dict(title=""),
        height=400, margin=dict(t=50, b=50, l=130, r=30),
        coloraxis_showscale=False
    )
    fig_mort.update_traces(
        marker=dict(cornerradius=4),
        text=[f"{v:.1f}%" for v in mort_by_diag["MortalityRate"]],
        textposition="outside",
        textfont=dict(family="Inter", weight=600)
    )
    st.plotly_chart(fig_mort, use_container_width=True)


# ── Lab Analysis Tab ─────────────────────────────────────────────────────────
with tab3:
    l1, l2 = st.columns(2)

    with l1:
        test_counts = labs_df.groupby("TestName").size().reset_index(name="Count")
        fig = px.bar(
            test_counts.sort_values("Count", ascending=True),
            y="TestName", x="Count", orientation="h",
            color="Count",
            color_continuous_scale=["#E8F0FE", "#1A73E8", "#1A237E"],
            title="Lab Test Frequency"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            title=dict(font=dict(size=16, weight=700)),
            xaxis=dict(title="Test Count", gridcolor="#E8EAED"),
            yaxis=dict(title=""),
            height=380, margin=dict(t=50, b=50, l=130, r=30),
            coloraxis_showscale=False
        )
        fig.update_traces(marker=dict(cornerradius=4))
        st.plotly_chart(fig, use_container_width=True)

    with l2:
        avg_results = labs_df.groupby("TestName")["Result"].mean().reset_index()
        fig = px.bar(
            avg_results.sort_values("Result", ascending=True),
            y="TestName", x="Result", orientation="h",
            color="TestName",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Average Lab Results by Test"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            title=dict(font=dict(size=16, weight=700)),
            xaxis=dict(title="Average Result Value", gridcolor="#E8EAED"),
            yaxis=dict(title=""),
            height=380, margin=dict(t=50, b=50, l=130, r=30),
            showlegend=False
        )
        fig.update_traces(marker=dict(cornerradius=4))
        st.plotly_chart(fig, use_container_width=True)

    # Lab results distribution
    fig_violin = px.violin(
        labs_df, x="TestName", y="Result",
        color="TestName",
        box=True, points="all",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Lab Result Distribution by Test Type"
    )
    fig_violin.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        title=dict(font=dict(size=16, weight=700)),
        xaxis=dict(title="", gridcolor="#E8EAED"),
        yaxis=dict(title="Result Value", gridcolor="#E8EAED"),
        height=420, margin=dict(t=50, b=50, l=60, r=30),
        showlegend=False
    )
    st.plotly_chart(fig_violin, use_container_width=True)


# ── Cost Analysis Tab ────────────────────────────────────────────────────────
with tab4:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            filtered_df, x="TreatmentCost", nbins=20,
            color_discrete_sequence=["#34A853"],
            title="Treatment Cost Distribution"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            title=dict(font=dict(size=16, weight=700)),
            xaxis=dict(title="Cost ($)", gridcolor="#E8EAED"),
            yaxis=dict(title="Count", gridcolor="#E8EAED"),
            height=380, margin=dict(t=50, b=50, l=50, r=30), bargap=0.05
        )
        fig.update_traces(marker=dict(cornerradius=4))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cost_outcome = filtered_df.groupby("OutcomeName")["TreatmentCost"].mean().reset_index()
        fig = px.bar(
            cost_outcome, x="OutcomeName", y="TreatmentCost",
            color="OutcomeName",
            color_discrete_map={"Recovered": "#34A853", "Complicated": "#FBBC04", "Deceased": "#EA4335"},
            title="Average Cost by Outcome"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            title=dict(font=dict(size=16, weight=700)),
            xaxis=dict(title="", gridcolor="#E8EAED"),
            yaxis=dict(title="Average Cost ($)", gridcolor="#E8EAED"),
            height=380, margin=dict(t=50, b=50, l=60, r=30),
            showlegend=False
        )
        fig.update_traces(
            marker=dict(cornerradius=6),
            text=[f"${v:,.0f}" for v in cost_outcome["TreatmentCost"]],
            textposition="outside",
            textfont=dict(family="Inter", weight=700)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cost vs Age with trend line
    fig_trend = px.scatter(
        filtered_df, x="Age", y="TreatmentCost",
        color="OutcomeName",
        color_discrete_map={"Recovered": "#34A853", "Complicated": "#FBBC04", "Deceased": "#EA4335"},
        trendline="ols",
        opacity=0.6,
        title="Treatment Cost vs Age (with Trend Line)"
    )
    fig_trend.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        title=dict(font=dict(size=16, weight=700)),
        xaxis=dict(title="Patient Age", gridcolor="#E8EAED"),
        yaxis=dict(title="Treatment Cost ($)", gridcolor="#E8EAED"),
        height=420, margin=dict(t=50, b=50, l=70, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_trend, use_container_width=True)
