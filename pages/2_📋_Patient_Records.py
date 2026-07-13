import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Records — HealthGuard AI",
    page_icon="📋",
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
        --text-tertiary: #80868B;                
        --border: #DADCE0;                       
        --border-light: #E8EAED;    
        --shadow-sm: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
        --shadow-md: 0 1px 3px 0 rgba(60,64,67,0.3), 0 4px 8px 3px rgba(60,64,67,0.15);
        --shadow-lg: 0 1px 3px 0 rgba(60,64,67,0.3), 0 8px 16px 6px rgba(60,64,67,0.15);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-full: 9999px;
        --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        --font-sans: 'Inter', 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: var(--font-sans);
    }

    h1, h2, h3, h4, h5, h6, p, label, .stMetric, .stSelectbox, .stNumberInput, .stTextInput, .stButton button {
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
    
    section[data-testid="stSidebar"] .stButton > button {
        background: #1A73E8 !important; /* Solid Blue */
        color: #FFFFFF !important;
        border: none !important;
        border-radius: var(--radius-full) !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        box-shadow: var(--shadow-sm) !important;
        transition: var(--transition) !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-md) !important;
        background: #1557B0 !important;
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

    /* DataFrame */
    .stDataFrame {
        border: 1px solid var(--border-light);
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
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

    /* Patient Profile Card */
    .profile-card {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-lg);
        padding: 2rem;
        box-shadow: var(--shadow-md);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    .profile-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background-color: var(--primary) !important; /* Solid color stripe */
    }
    .profile-card:hover { box-shadow: var(--shadow-lg); }

    .profile-name {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text-primary);
        margin: 0.5rem 0 0.25rem;
    }
    .profile-id {
        font-size: 0.8rem;
        color: var(--text-tertiary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1rem;
        margin-top: 1.25rem;
    }
    .info-item {
        padding: 0.75rem;
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-sm);
    }
    .info-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-tertiary);
        margin-bottom: 4px;
    }
    .info-value {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.75rem;
        border-radius: var(--radius-full);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .badge-recovered { background: var(--success-light); color: var(--success); }
    .badge-complicated { background: var(--warning-light); color: #E37400; }
    .badge-deceased { background: var(--danger-light); color: var(--danger); }

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
        .profile-name { font-size: 1.2rem; }
        .info-grid { grid-template-columns: repeat(2, 1fr); }
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

    df["AdmissionDate"] = pd.to_datetime(df["AdmissionDate"], errors="coerce")
    df["DischargeDate"] = pd.to_datetime(df["DischargeDate"], errors="coerce")
    df["LengthOfStay"] = (df["DischargeDate"] - df["AdmissionDate"]).dt.days.abs()

    return patients, diagnoses, labs, outcomes, df


@st.cache_resource
def load_model():
    return joblib.load("Risk_Model1.ipynb")


patients_df, diagnoses_df, labs_df, outcomes_df, merged_df = load_data()
model = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 1rem 0.5rem;">
        <h2 style="font-size:1.4rem;font-weight:800;margin:0;">🛡️ HealthGuard AI</h2>
        <p style="font-size:0.75rem;opacity:0.7;text-transform:uppercase;letter-spacing:1.5px;margin:0.25rem 0 0;">Patient Records</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<p style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;opacity:0.5;margin:0 0 0.5rem 0;">🔍 Search & Filter</p>', unsafe_allow_html=True)

    search_name = st.text_input("Search by Name", placeholder="Type patient name...")

    filter_diagnosis = st.selectbox(
        "Diagnosis",
        ["All Diagnoses"] + diagnoses_df["DiagnosisName"].tolist()
    )

    filter_outcome = st.selectbox(
        "Outcome",
        ["All Outcomes"] + outcomes_df["OutcomeName"].tolist()
    )

    filter_gender = st.selectbox(
        "Gender",
        ["All", "Male (M)", "Female (F)"]
    )

    age_range = st.slider(
        "Age Range",
        min_value=int(patients_df["Age"].min()),
        max_value=int(patients_df["Age"].max()),
        value=(int(patients_df["Age"].min()), int(patients_df["Age"].max()))
    )

    st.divider()

    # Select patient for detail view
    st.markdown('<p style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;opacity:0.5;margin:0 0 0.5rem 0;">👤 Patient Profile</p>', unsafe_allow_html=True)

    patient_names = merged_df["Name"].unique().tolist()
    selected_patient_name = st.selectbox(
        "Select Patient",
        ["— Select —"] + sorted(patient_names)
    )


# ─────────────────────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────────────────────
display_df = merged_df.copy()

if search_name:
    display_df = display_df[display_df["Name"].str.contains(search_name, case=False, na=False)]
if filter_diagnosis != "All Diagnoses":
    display_df = display_df[display_df["DiagnosisName"] == filter_diagnosis]
if filter_outcome != "All Outcomes":
    display_df = display_df[display_df["OutcomeName"] == filter_outcome]
if filter_gender == "Male (M)":
    display_df = display_df[display_df["Gender"] == "M"]
elif filter_gender == "Female (F)":
    display_df = display_df[display_df["Gender"] == "F"]

display_df = display_df[
    (display_df["Age"] >= age_range[0]) &
    (display_df["Age"] <= age_range[1])
]


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>📋 Patient Records</h1>
    <p>Browse, search, and analyze individual patient records. Select a patient from the sidebar to view their detailed profile and risk assessment.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KEY METRICS
# ─────────────────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Records Found", f"{len(display_df):,}")
with m2:
    if len(display_df) > 0:
        st.metric("Age Range", f"{display_df['Age'].min()} – {display_df['Age'].max()}")
    else:
        st.metric("Age Range", "—")
with m3:
    if len(display_df) > 0:
        st.metric("Avg Treatment Cost", f"${display_df['TreatmentCost'].mean():,.0f}")
    else:
        st.metric("Avg Treatment Cost", "—")
with m4:
    if len(display_df) > 0:
        st.metric("Avg Length of Stay", f"{display_df['LengthOfStay'].mean():.0f} days")
    else:
        st.metric("Avg Length of Stay", "—")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT PROFILE (if selected)
# ─────────────────────────────────────────────────────────────────────────────
if selected_patient_name != "— Select —":
    patient_data = merged_df[merged_df["Name"] == selected_patient_name].iloc[0]
    patient_labs = labs_df[labs_df["PatientID"] == patient_data["PatientID"]]

    # Outcome badge class
    outcome = patient_data.get("OutcomeName", "Unknown")
    badge_class = "badge-recovered" if outcome == "Recovered" else ("badge-deceased" if outcome == "Deceased" else "badge-complicated")

    # Run risk prediction for this patient
    input_data = pd.DataFrame(
        [[patient_data["Age"], patient_data["LengthOfStay"], patient_data["TreatmentCost"]]],
        columns=["Age", "LengthOfStay", "TreatmentCost"]
    )
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    risk_color = "#EA4335" if prediction == 1 else "#34A853"

    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-id">Patient ID: {patient_data['PatientID']}</div>
        <div class="profile-name">{patient_data['Name']}</div>
        <span class="badge {badge_class}" style="margin-top:0.5rem">{outcome}</span>
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Age</div>
                <div class="info-value">{patient_data['Age']} years</div>
            </div>
            <div class="info-item">
                <div class="info-label">Gender</div>
                <div class="info-value">{'Male' if patient_data['Gender'] == 'M' else 'Female'}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Diagnosis</div>
                <div class="info-value">{patient_data.get('DiagnosisName', 'N/A')}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Treatment Cost</div>
                <div class="info-value">${patient_data['TreatmentCost']:,.0f}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Length of Stay</div>
                <div class="info-value">{patient_data['LengthOfStay']:.0f} days</div>
            </div>
            <div class="info-item">
                <div class="info-label">Risk Assessment</div>
                <div class="info-value" style="color:{risk_color}">{risk_label} ({probability:.0%})</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Patient charts
    pc1, pc2 = st.columns(2)

    with pc1:
        # Risk gauge for this patient
        radius = 75
        circumference = 2 * 3.14159 * radius
        dash_offset = circumference * (1 - probability)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            title=dict(text="Risk Score", font=dict(size=16, family="Inter")),
            number=dict(suffix="%", font=dict(size=36, family="Inter", weight=700)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#5F6368"),
                bar=dict(color=risk_color),
                bgcolor="#E8EAED",
                borderwidth=0,
                steps=[
                    dict(range=[0, 30], color="#E6F4EA"),
                    dict(range=[30, 70], color="#FEF7E0"),
                    dict(range=[70, 100], color="#FCE8E6")
                ],
                threshold=dict(
                    line=dict(color=risk_color, width=3),
                    thickness=0.8,
                    value=probability * 100
                )
            )
        ))
        fig_gauge.update_layout(
            height=300,
            margin=dict(t=60, b=30, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with pc2:
        # Lab results for this patient
        if len(patient_labs) > 0:
            fig_labs = px.bar(
                patient_labs, x="TestName", y="Result",
                color="TestName",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title=f"Lab Results for {selected_patient_name}"
            )
            fig_labs.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                title=dict(font=dict(size=16, weight=700)),
                xaxis=dict(title="", gridcolor="#E8EAED"),
                yaxis=dict(title="Result Value", gridcolor="#E8EAED"),
                height=300, margin=dict(t=50, b=40, l=50, r=30),
                showlegend=False
            )
            fig_labs.update_traces(marker=dict(cornerradius=6))
            st.plotly_chart(fig_labs, use_container_width=True)
        else:
            st.info("No lab results available for this patient.")

    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h3 style="font-weight:700;color:#202124;margin-bottom:0.75rem;">📊 All Patient Records</h3>', unsafe_allow_html=True)

if len(display_df) > 0:
    show_cols = ["PatientID", "Name", "Age", "Gender", "DiagnosisName", "OutcomeName",
                 "TreatmentCost", "LengthOfStay", "AdmissionDate", "DischargeDate"]
    available_cols = [c for c in show_cols if c in display_df.columns]
    table_df = display_df[available_cols].copy()

    rename_map = {
        "PatientID": "ID",
        "Name": "Patient Name",
        "Age": "Age",
        "Gender": "Gender",
        "DiagnosisName": "Diagnosis",
        "OutcomeName": "Outcome",
        "TreatmentCost": "Cost ($)",
        "LengthOfStay": "Stay (days)",
        "AdmissionDate": "Admitted",
        "DischargeDate": "Discharged"
    }
    table_df.columns = [rename_map.get(c, c) for c in table_df.columns]

    st.dataframe(
        table_df,
        use_container_width=True,
        height=600,
        column_config={
            "ID": st.column_config.NumberColumn("ID", width="small"),
            "Cost ($)": st.column_config.NumberColumn("Cost ($)", format="$%d"),
        }
    )

    # Download button
    csv = table_df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="patient_records.csv",
        mime="text/csv"
    )
else:
    st.info("No patients found matching your filters. Try adjusting the search criteria in the sidebar.")
