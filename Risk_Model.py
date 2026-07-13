import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthGuard AI — Risk Prediction System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "### HealthGuard AI\nEnterprise Healthcare Risk Prediction System\n\nPowered by Machine Learning"
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — PRODUCTION GRADE DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts Import ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap');

    /* ── CSS Custom Properties (Design Tokens) ───────────────────────── */
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
        --primary-dark: #E68A00;                 /* Slightly darker shade for active button states */
        --success: #34A853;                      /* Keep Recovered / Low Risk green */
        --success-light: #E6F4EA;   
        --danger: #EA4335;                       /* Keep Deceased / High Risk red */
        --danger-light: #FCE8E6;    
        --warning: #FBBC04;         
        --warning-light: #FEF7E0;   
        --bg-primary: #F8F9FA;                   /* Page body background (Light Grey) */
        --bg-card: var(--theme-white);           /* Surface/Card background (White) */
        --text-primary: #202124;                 /* Heading & Title text (Dark Grey) */
        --text-secondary: #5F6368;               /* Body text (Grey) */
        --text-tertiary: #80868B;                
        --border: #DADCE0;                       
        --border-light: #E8EAED;    
        --shadow-sm: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
        --shadow-md: 0 1px 3px 0 rgba(60,64,67,0.3), 0 4px 8px 3px rgba(60,64,67,0.15);
        --shadow-lg: 0 1px 3px 0 rgba(60,64,67,0.3), 0 8px 16px 6px rgba(60,64,67,0.15);
        --shadow-xl: 0 4px 8px 0 rgba(60,64,67,0.3), 0 16px 32px 8px rgba(60,64,67,0.15);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 24px;
        --radius-full: 9999px;
        --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        --font-sans: 'Inter', 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-mono: 'Roboto Mono', 'SF Mono', 'Consolas', monospace;
    }

    /* ── Global Reset & Base ─────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: var(--font-sans);
    }

    /* Custom font for headings, body text, inputs, buttons, and labels, while preserving Streamlit close icons */
    h1, h2, h3, h4, h5, h6, p, label, .stMetric, .stSelectbox, .stNumberInput, .stTextInput, .stButton button {
        font-family: var(--font-sans) !important;
    }

    .stApp {
        background: var(--bg-primary);
    }

    /* ── Sidebar Styling ─────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--theme-teal) !important; /* Sidebar Background: Teal (#2EC4B6) */
        box-shadow: var(--shadow-lg);
    }

    /* Collapse button icon color */
    section[data-testid="stSidebar"] button svg,
    section[data-testid="stSidebar"] button svg * {
        fill: var(--theme-white) !important;      /* Color: White (#FFFFFF) */
        color: var(--theme-white) !important;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .sidebar-brand h2,
    section[data-testid="stSidebar"] .sidebar-brand p,
    section[data-testid="stSidebar"] .sidebar-section-title {
        color: var(--theme-white) !important;     /* Text Color: White (#FFFFFF) */
    }

    section[data-testid="stSidebar"] label {
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.3px;
        text-transform: uppercase;
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
        background-color: rgba(255, 255, 255, 0.25) !important; /* Semi-transparent white active indicator */
        font-weight: 600 !important;
    }

    /* Sidebar inputs: High-contrast white background with dark slate text */
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: var(--theme-white) !important; /* Input background: White */
        color: #1E293B !important;                       /* Input text: Dark Slate */
        border: 1px solid #DADCE0 !important;
        border-radius: var(--radius-sm) !important;
        transition: var(--transition);
    }

    section[data-testid="stSidebar"] .stNumberInput input:focus,
    section[data-testid="stSidebar"] .stTextInput input:focus,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(255, 159, 28, 0.15) !important;
    }

    /* Force dropdown options list to have dark text on light background */
    div[data-baseweb="popover"] *,
    div[role="listbox"] * {
        color: #1E293B !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
        margin: 1.25rem 0 !important;
    }

    /* ── Sidebar Button ──────────────────────────────────────────────── */
    section[data-testid="stSidebar"] .stButton > button {
        background: #1A73E8 !important; /* Solid Blue */
        color: #FFFFFF !important;
        border: none !important;
        border-radius: var(--radius-full) !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: var(--shadow-md) !important;
        transition: var(--transition) !important;
        width: 100% !important;
        cursor: pointer;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-xl) !important;
        background: #1557B0 !important;
    }

    section[data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Main Content Buttons ────────────────────────────────────────── */
    .stMainBlockContainer .stButton > button {
        background: #1A73E8 !important; /* Solid Blue */
        color: #FFFFFF !important;
        border: none !important;
        border-radius: var(--radius-full) !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        box-shadow: var(--shadow-sm) !important;
        transition: var(--transition) !important;
    }

    .stMainBlockContainer .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-md) !important;
        background: #1557B0 !important;
    }

    /* ── Metric Cards ────────────────────────────────────────────────── */
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
        border-color: var(--primary);
    }

    div[data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 800 !important;
        font-size: 1.75rem !important;
    }

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

    /* ── Expander ────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    /* ── DataFrames / Tables ─────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid var(--border-light);
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }

    /* ── Toast & Alerts ──────────────────────────────────────────────── */
    .stAlert {
        border-radius: var(--radius-md) !important;
        border-left-width: 4px !important;
        font-weight: 500;
    }

    /* ── Plotly Chart Container ───────────────────────────────────────── */
    .stPlotlyChart {
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-md);
        padding: 0.5rem;
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }

    .stPlotlyChart:hover {
        box-shadow: var(--shadow-md);
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

    /* ── Responsive Media Queries ────────────────────────────────────── */
    @media (max-width: 640px) {
        .hero-banner h1 { font-size: 1.5rem !important; }
        .hero-banner p { font-size: 0.9rem !important; }
        div[data-testid="stMetric"] { padding: 0.75rem 1rem; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.25rem !important; }
        .result-card { padding: 1.25rem !important; }
    }

    @media (min-width: 641px) and (max-width: 1024px) {
        .hero-banner h1 { font-size: 1.75rem !important; }
        .result-card { padding: 1.5rem !important; }
    }

    @media (min-width: 1441px) {
        .stMainBlockContainer { max-width: 1400px; margin: 0 auto; }
    }

    /* ── Animations ──────────────────────────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(30px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out forwards;
    }

    .animate-slide-in {
        animation: slideInRight 0.5s ease-out forwards;
    }

    /* ── Hero Banner ─────────────────────────────────────────────────── */
    .hero-banner {
        background: var(--theme-orange-lt) !important; /* Hero Background: Light Orange (#FFBF69) */
        border-radius: var(--radius-lg);
        padding: 2.5rem 3rem;
        color: var(--theme-white) !important;          /* Hero Text: White (#FFFFFF) */
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.6s ease-out;
    }

    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(255,255,255,0.02) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-banner h1 {
        font-family: var(--font-sans);
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }

    .hero-banner p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
        font-weight: 400;
        line-height: 1.6;
        position: relative;
        z-index: 1;
        max-width: 600px;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: var(--radius-full);
        padding: 0.35rem 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }

    .hero-badge .pulse-dot {
        width: 8px;
        height: 8px;
        background: #34A853;
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
        box-shadow: 0 0 6px rgba(52,168,83,0.4);
    }

    /* ── Result Cards ────────────────────────────────────────────────── */
    .result-card {
        background: var(--bg-card);
        border-radius: var(--radius-lg);
        padding: 2rem;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-md);
        animation: fadeInUp 0.5s ease-out;
        transition: var(--transition);
    }

    .result-card:hover {
        box-shadow: var(--shadow-lg);
    }

    .result-card.high-risk {
        border-left: 4px solid var(--danger);
        background-color: var(--danger-light) !important;
    }

    .result-card.low-risk {
        border-left: 4px solid var(--success);
        background-color: var(--success-light) !important;
    }

    .result-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-secondary);
        margin-bottom: 0.75rem;
    }

    .result-value {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.1;
    }

    .result-value.high { color: var(--danger); }
    .result-value.low { color: var(--success); }

    .result-subtitle {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* ── Risk Gauge ──────────────────────────────────────────────────── */
    .gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1.5rem;
        animation: fadeInUp 0.7s ease-out;
    }

    .gauge-ring {
        position: relative;
        width: 200px;
        height: 200px;
    }

    .gauge-ring svg {
        transform: rotate(-90deg);
    }

    .gauge-ring .gauge-bg {
        fill: none;
        stroke: #E8EAED;
        stroke-width: 12;
    }

    .gauge-ring .gauge-fill {
        fill: none;
        stroke-width: 12;
        stroke-linecap: round;
        transition: stroke-dashoffset 1.5s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.3s ease;
    }

    .gauge-center {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
    }

    .gauge-percentage {
        font-size: 2.5rem;
        font-weight: 800;
        font-family: var(--font-sans);
        letter-spacing: -1px;
    }

    .gauge-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
        margin-top: 2px;
    }

    /* ── Progress Bar ────────────────────────────────────────────────── */
    .risk-bar-container {
        background: #E8EAED;
        border-radius: var(--radius-full);
        height: 10px;
        overflow: hidden;
        margin: 0.5rem 0;
    }

    .risk-bar-fill {
        height: 100%;
        border-radius: var(--radius-full);
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Section Headers ─────────────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border-light);
        animation: fadeInUp 0.5s ease-out;
    }

    .section-header h2 {
        font-family: var(--font-sans);
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.3px;
    }

    .section-header .section-icon {
        font-size: 1.5rem;
    }

    /* ── Info Chips ───────────────────────────────────────────────────── */
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 0.25rem 0.75rem;
        border-radius: var(--radius-full);
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .chip-primary { background: var(--primary-light); color: var(--primary); }
    .chip-success { background: var(--success-light); color: var(--success); }
    .chip-danger { background: var(--danger-light); color: var(--danger); }
    .chip-warning { background: var(--warning-light); color: #E37400; }

    /* ── Footer ──────────────────────────────────────────────────────── */
    .footer {
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border-light);
        color: var(--text-tertiary);
        font-size: 0.8rem;
    }

    .footer a {
        color: var(--primary);
        text-decoration: none;
        font-weight: 500;
    }

    /* ── Sidebar Brand ───────────────────────────────────────────────── */
    .sidebar-brand {
        text-align: center;
        padding: 1.5rem 1rem 1rem 1rem;
    }

    .sidebar-brand h2 {
        font-family: var(--font-sans);
        font-size: 1.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.3px;
    }

    .sidebar-brand p {
        font-size: 0.75rem;
        opacity: 0.7;
        margin: 0.25rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
    }

    .sidebar-section-title {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.5;
        margin: 1.5rem 0 0.5rem 0;
    }

    /* ── Quick Stat Boxes ────────────────────────────────────────────── */
    .quick-stat {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: var(--radius-sm);
        padding: 0.75rem;
        text-align: center;
        backdrop-filter: blur(5px);
        transition: var(--transition);
        color: #FFFFFF !important;
    }

    .quick-stat:hover {
        background: rgba(255,255,255,0.14);
    }

    .quick-stat .stat-value {
        font-size: 1.3rem;
        font-weight: 800;
        display: block;
        color: #FFFFFF !important;
    }

    .quick-stat .stat-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        opacity: 0.7;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.8) !important;
    }

    /* ── Welcome state ───────────────────────────────────────────────── */
    .welcome-state {
        text-align: center;
        padding: 4rem 2rem;
        animation: fadeInUp 0.8s ease-out;
    }

    .welcome-state .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }

    .welcome-state h3 {
        font-family: var(--font-sans);
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.75rem 0;
    }

    .welcome-state p {
        color: var(--text-secondary);
        font-size: 1rem;
        max-width: 480px;
        margin: 0 auto;
        line-height: 1.6;
    }

    .step-list {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        max-width: 400px;
        margin: 1.5rem auto 0 auto;
        text-align: left;
    }

    .step-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.75rem 1rem;
        background: var(--bg-card);
        border: 1px solid var(--border-light);
        border-radius: var(--radius-sm);
        font-size: 0.9rem;
        color: var(--text-primary);
        transition: var(--transition);
    }

    .step-item:hover {
        border-color: var(--primary);
        box-shadow: var(--shadow-sm);
    }

    .step-number {
        width: 28px;
        height: 28px;
        background: var(--primary);
        color: #FFFFFF;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        flex-shrink: 0;
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

    # Merge for enriched dataset
    df = patients.merge(diagnoses, on="DiagnosisID", how="left")
    df = df.merge(outcomes, on="OutcomeID", how="left")
    df = df.merge(labs, left_on="PatientID", right_on="PatientID", how="left")

    # Calculate length of stay
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
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🛡️ HealthGuard AI</h2>
        <p>Risk Prediction Engine</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Patient Information Section
    st.markdown('<div class="sidebar-section-title">📋 Patient Information</div>', unsafe_allow_html=True)

    age = st.number_input(
        "Age (years)",
        min_value=0,
        max_value=120,
        value=45,
        step=1,
        help="Patient's age in years"
    )

    length_of_stay = st.number_input(
        "Length of Stay (days)",
        min_value=0,
        max_value=365,
        value=7,
        step=1,
        help="Expected hospital stay duration"
    )

    treatment_cost = st.number_input(
        "Treatment Cost ($)",
        min_value=0.0,
        max_value=100000.0,
        value=8000.0,
        step=100.0,
        format="%.0f",
        help="Estimated total treatment cost in USD"
    )

    diagnosis = st.selectbox(
        "Primary Diagnosis",
        options=diagnoses_df["DiagnosisName"].tolist(),
        index=0,
        help="Patient's primary diagnosis"
    )

    gender = st.selectbox(
        "Gender",
        options=["Male", "Female"],
        index=0
    )

    st.divider()

    # Predict Button
    predict_clicked = st.button("🔍  Analyze Patient Risk", use_container_width=True)

    st.divider()

    # Quick Stats
    st.markdown('<div class="sidebar-section-title">📊 System Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="quick-stat">
            <span class="stat-value">{len(patients_df)}</span>
            <span class="stat-label">Patients</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="quick-stat">
            <span class="stat-value">{len(diagnoses_df)}</span>
            <span class="stat-label">Diagnoses</span>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"""
        <div class="quick-stat">
            <span class="stat-value">{len(labs_df)}</span>
            <span class="stat-label">Lab Tests</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_age = int(patients_df["Age"].mean())
        st.markdown(f"""
        <div class="quick-stat">
            <span class="stat-value">{avg_age}</span>
            <span class="stat-label">Avg Age</span>
        </div>
        """, unsafe_allow_html=True)

    # Footer in sidebar
    st.markdown("""
    <div style="text-align:center; margin-top:2rem; opacity:0.6; font-size:0.7rem; color: #FFFFFF !important;">
        v2.0.0 · ML Engine Active<br/>
        Scikit-learn Logistic Regression
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT — HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">
        <span class="pulse-dot"></span>
        System Online · ML Model Active
    </div>
    <h1>Healthcare Risk Prediction System</h1>
    <p>Enterprise-grade patient risk stratification powered by Machine Learning.
       Enter patient details in the sidebar and click <strong>Analyze Patient Risk</strong> to generate
       a comprehensive risk assessment.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL METRICS
# ─────────────────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        label="Total Patients",
        value=f"{len(patients_df):,}",
        delta="Active Records"
    )
with m2:
    avg_cost = patients_df["TreatmentCost"].mean()
    st.metric(
        label="Avg Treatment Cost",
        value=f"${avg_cost:,.0f}",
        delta=f"Range ${patients_df['TreatmentCost'].min():,.0f} – ${patients_df['TreatmentCost'].max():,.0f}"
    )
with m3:
    # Count high-risk predictions from our dataset
    deceased_count = len(patients_df[patients_df["OutcomeID"] == 3])
    high_risk_pct = (deceased_count / len(patients_df)) * 100
    st.metric(
        label="High-Risk Rate",
        value=f"{high_risk_pct:.1f}%",
        delta=f"{deceased_count} patients",
        delta_color="inverse"
    )
with m4:
    diagnosis_count = len(diagnoses_df)
    st.metric(
        label="Diagnoses Tracked",
        value=f"{diagnosis_count}",
        delta="Categories"
    )

st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_dashboard, tab_patients, tab_model = st.tabs([
    "🔍  Risk Analysis",
    "📊  Dashboard",
    "👥  Patient Records",
    "🧠  Model Insights"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RISK PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    if predict_clicked:
        # Run prediction
        input_data = pd.DataFrame(
            [[age, length_of_stay, treatment_cost]],
            columns=["Age", "LengthOfStay", "TreatmentCost"]
        )
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        risk_label = "High Risk" if prediction == 1 else "Low Risk"
        risk_class = "high-risk" if prediction == 1 else "low-risk"
        risk_color = "#EA4335" if prediction == 1 else "#34A853"
        prob_pct = probability * 100

        # Calculate circumference for gauge
        radius = 85
        circumference = 2 * 3.14159 * radius
        dash_offset = circumference * (1 - probability)

        # Results Row
        st.markdown('<div class="section-header"><span class="section-icon">📋</span><h2>Risk Assessment Results</h2></div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns([1, 1, 1])

        with r1:
            st.markdown(f"""
            <div class="result-card {risk_class}">
                <div class="result-title">Risk Classification</div>
                <div class="result-value {'high' if prediction == 1 else 'low'}">{risk_label}</div>
                <div class="result-subtitle">Based on Logistic Regression model analysis</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="gauge-container">
                <div class="gauge-ring">
                    <svg width="200" height="200" viewBox="0 0 200 200">
                        <circle class="gauge-bg" cx="100" cy="100" r="{radius}"/>
                        <circle class="gauge-fill" cx="100" cy="100" r="{radius}"
                            stroke="{risk_color}"
                            stroke-dasharray="{circumference}"
                            stroke-dashoffset="{dash_offset}"/>
                    </svg>
                    <div class="gauge-center">
                        <div class="gauge-percentage" style="color:{risk_color}">{prob_pct:.1f}%</div>
                        <div class="gauge-label">Risk Score</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            safe_prob = (1 - probability) * 100
            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">Risk Breakdown</div>
                <div style="margin-bottom:1rem">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="font-size:0.85rem;font-weight:500;color:var(--text-secondary)">High Risk Probability</span>
                        <span style="font-size:0.85rem;font-weight:700;color:var(--danger)">{prob_pct:.1f}%</span>
                    </div>
                    <div class="risk-bar-container">
                        <div class="risk-bar-fill" style="width:{prob_pct}%;background:linear-gradient(90deg,#FBBC04,#EA4335)"></div>
                    </div>
                </div>
                <div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="font-size:0.85rem;font-weight:500;color:var(--text-secondary)">Low Risk Probability</span>
                        <span style="font-size:0.85rem;font-weight:700;color:var(--success)">{safe_prob:.1f}%</span>
                    </div>
                    <div class="risk-bar-container">
                        <div class="risk-bar-fill" style="width:{safe_prob}%;background:linear-gradient(90deg,#34A853,#0F9D58)"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Patient Summary Card
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="section-icon">🏥</span><h2>Patient Summary</h2></div>', unsafe_allow_html=True)

        ps1, ps2, ps3, ps4 = st.columns(4)
        with ps1:
            st.metric("Age", f"{age} years")
        with ps2:
            st.metric("Length of Stay", f"{length_of_stay} days")
        with ps3:
            st.metric("Treatment Cost", f"${treatment_cost:,.0f}")
        with ps4:
            st.metric("Diagnosis", diagnosis)

        # Risk Factors Visualization
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        factor_col1, factor_col2 = st.columns(2)

        with factor_col1:
            # Probability comparison chart
            fig_prob = go.Figure(go.Bar(
                x=["High Risk", "Low Risk"],
                y=[probability * 100, (1 - probability) * 100],
                marker=dict(
                    color=["#EA4335", "#34A853"],
                    cornerradius=8
                ),
                text=[f"{probability*100:.1f}%", f"{(1-probability)*100:.1f}%"],
                textposition="outside",
                textfont=dict(size=14, family="Inter", weight=700)
            ))
            fig_prob.update_layout(
                title=dict(text="Risk Probability Distribution", font=dict(size=16, family="Inter", weight=700)),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title="Probability (%)", range=[0, 110], gridcolor="#E8EAED"),
                xaxis=dict(title=""),
                height=350,
                margin=dict(t=50, b=40, l=50, r=30),
                font=dict(family="Inter")
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        with factor_col2:
            # Patient vs Average comparison
            avg_age = patients_df["Age"].mean()
            avg_los = merged_df["LengthOfStay"].mean() if "LengthOfStay" in merged_df.columns else 7
            avg_cost = patients_df["TreatmentCost"].mean()

            categories = ["Age", "Length of Stay", "Treatment Cost"]
            patient_vals = [age, length_of_stay, treatment_cost]
            avg_vals = [avg_age, avg_los, avg_cost]

            # Normalize for radar
            max_vals = [max(age, avg_age, 1), max(length_of_stay, avg_los, 1), max(treatment_cost, avg_cost, 1)]
            patient_norm = [v / m * 100 for v, m in zip(patient_vals, max_vals)]
            avg_norm = [v / m * 100 for v, m in zip(avg_vals, max_vals)]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=patient_norm + [patient_norm[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="This Patient",
                line=dict(color="#1A73E8", width=2),
                fillcolor="rgba(26,115,232,0.15)"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_norm + [avg_norm[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Population Average",
                line=dict(color="#FBBC04", width=2, dash="dash"),
                fillcolor="rgba(251,188,4,0.1)"
            ))
            fig_radar.update_layout(
                title=dict(text="Patient vs Population Average", font=dict(size=16, family="Inter", weight=700)),
                polar=dict(radialaxis=dict(visible=True, range=[0, 110], gridcolor="#E8EAED")),
                height=350,
                margin=dict(t=60, b=40, l=60, r=60),
                font=dict(family="Inter"),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Clinical Recommendation
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if prediction == 1:
            st.error("⚠️ **HIGH RISK DETECTED** — This patient shows elevated risk indicators. Recommend immediate clinical review, enhanced monitoring protocols, and specialist consultation.")
        else:
            st.success("✅ **LOW RISK** — This patient's risk profile is within normal parameters. Standard care protocols are recommended with routine follow-up scheduling.")

    else:
        # Welcome state
        st.markdown("""
        <div class="welcome-state">
            <span class="welcome-icon">🩺</span>
            <h3>Ready to Analyze Patient Risk</h3>
            <p>Enter patient details in the sidebar and click <strong>Analyze Patient Risk</strong> to generate a comprehensive risk assessment report.</p>
            <div class="step-list">
                <div class="step-item">
                    <span class="step-number">1</span>
                    Enter patient age, stay duration, and treatment cost
                </div>
                <div class="step-item">
                    <span class="step-number">2</span>
                    Select primary diagnosis and gender
                </div>
                <div class="step-item">
                    <span class="step-number">3</span>
                    Click <strong>Analyze Patient Risk</strong> button
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_dashboard:
    st.markdown('<div class="section-header"><span class="section-icon">📊</span><h2>Healthcare Analytics Dashboard</h2></div>', unsafe_allow_html=True)

    chart1, chart2 = st.columns(2)

    with chart1:
        # Age Distribution
        fig_age = px.histogram(
            patients_df,
            x="Age",
            nbins=20,
            color_discrete_sequence=["#1A73E8"],
            title="Patient Age Distribution"
        )
        fig_age.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            title=dict(font=dict(size=16, weight=700)),
            xaxis=dict(title="Age (years)", gridcolor="#E8EAED"),
            yaxis=dict(title="Count", gridcolor="#E8EAED"),
            height=380,
            margin=dict(t=50, b=50, l=50, r=30),
            bargap=0.05
        )
        fig_age.update_traces(
            marker=dict(cornerradius=4, line=dict(width=0)),
            hovertemplate="Age: %{x}<br>Count: %{y}<extra></extra>"
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with chart2:
        # Treatment Cost by Diagnosis
        diag_cost = merged_df.groupby("DiagnosisName")["TreatmentCost"].mean().reset_index()
        diag_cost = diag_cost.sort_values("TreatmentCost", ascending=True)

        fig_cost = px.bar(
            diag_cost,
            y="DiagnosisName",
            x="TreatmentCost",
            orientation="h",
            color="TreatmentCost",
            color_continuous_scale=["#E8F0FE", "#1A73E8", "#1A237E"],
            title="Average Treatment Cost by Diagnosis"
        )
        fig_cost.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            title=dict(font=dict(size=16, weight=700)),
            xaxis=dict(title="Average Cost ($)", gridcolor="#E8EAED"),
            yaxis=dict(title=""),
            height=380,
            margin=dict(t=50, b=50, l=120, r=30),
            coloraxis_showscale=False
        )
        fig_cost.update_traces(
            marker=dict(cornerradius=4),
            hovertemplate="Diagnosis: %{y}<br>Avg Cost: $%{x:,.0f}<extra></extra>"
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    chart3, chart4 = st.columns(2)

    with chart3:
        # Outcome Distribution — Donut Chart
        outcome_counts = merged_df.groupby("OutcomeName").size().reset_index(name="Count")
        fig_outcome = px.pie(
            outcome_counts,
            names="OutcomeName",
            values="Count",
            hole=0.55,
            color="OutcomeName",
            color_discrete_map={
                "Recovered": "#34A853",
                "Complicated": "#FBBC04",
                "Deceased": "#EA4335"
            },
            title="Patient Outcome Distribution"
        )
        fig_outcome.update_layout(
            font=dict(family="Inter"),
            title=dict(font=dict(size=16, weight=700)),
            height=380,
            margin=dict(t=50, b=50, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        fig_outcome.update_traces(
            textinfo="percent+label",
            textfont=dict(size=12, family="Inter"),
            hovertemplate="Outcome: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
        )
        st.plotly_chart(fig_outcome, use_container_width=True)

    with chart4:
        # Age vs Treatment Cost Scatter
        scatter_df = merged_df.copy()
        scatter_df["Outcome"] = scatter_df["OutcomeName"]

        fig_scatter = px.scatter(
            scatter_df,
            x="Age",
            y="TreatmentCost",
            color="Outcome",
            color_discrete_map={
                "Recovered": "#34A853",
                "Complicated": "#FBBC04",
                "Deceased": "#EA4335"
            },
            size="LengthOfStay",
            size_max=18,
            opacity=0.7,
            title="Age vs Treatment Cost (by Outcome)"
        )
        fig_scatter.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            title=dict(font=dict(size=16, weight=700)),
            xaxis=dict(title="Patient Age", gridcolor="#E8EAED"),
            yaxis=dict(title="Treatment Cost ($)", gridcolor="#E8EAED"),
            height=380,
            margin=dict(t=50, b=50, l=60, r=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        fig_scatter.update_traces(
            hovertemplate="Age: %{x}<br>Cost: $%{y:,.0f}<extra></extra>"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Diagnosis Breakdown — Full Width
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    diag_counts = merged_df.groupby("DiagnosisName").size().reset_index(name="Count")
    diag_counts = diag_counts.sort_values("Count", ascending=False)

    fig_diag = px.bar(
        diag_counts,
        x="DiagnosisName",
        y="Count",
        color="Count",
        color_continuous_scale=["#E8F0FE", "#4285F4", "#1A73E8", "#1557B0", "#1A237E"],
        title="Diagnosis Frequency Distribution"
    )
    fig_diag.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        title=dict(font=dict(size=16, weight=700)),
        xaxis=dict(title="Diagnosis", gridcolor="#E8EAED", tickangle=-30),
        yaxis=dict(title="Patient Count", gridcolor="#E8EAED"),
        height=380,
        margin=dict(t=50, b=80, l=50, r=30),
        coloraxis_showscale=False,
        bargap=0.2
    )
    fig_diag.update_traces(
        marker=dict(cornerradius=6),
        hovertemplate="Diagnosis: %{x}<br>Count: %{y}<extra></extra>"
    )
    st.plotly_chart(fig_diag, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PATIENT RECORDS
# ─────────────────────────────────────────────────────────────────────────────
with tab_patients:
    st.markdown('<div class="section-header"><span class="section-icon">👥</span><h2>Patient Records</h2></div>', unsafe_allow_html=True)

    # Search and Filter
    filter1, filter2, filter3 = st.columns([2, 1, 1])

    with filter1:
        search_term = st.text_input("🔍 Search by Name", placeholder="Type patient name...", label_visibility="collapsed")
    with filter2:
        diag_filter = st.selectbox("Filter by Diagnosis", ["All Diagnoses"] + diagnoses_df["DiagnosisName"].tolist())
    with filter3:
        outcome_filter = st.selectbox("Filter by Outcome", ["All Outcomes"] + outcomes_df["OutcomeName"].tolist())

    # Apply filters
    display_df = merged_df.copy()

    if search_term:
        display_df = display_df[display_df["Name"].str.contains(search_term, case=False, na=False)]
    if diag_filter != "All Diagnoses":
        display_df = display_df[display_df["DiagnosisName"] == diag_filter]
    if outcome_filter != "All Outcomes":
        display_df = display_df[display_df["OutcomeName"] == outcome_filter]

    # Summary metrics
    pm1, pm2, pm3, pm4 = st.columns(4)
    with pm1:
        st.metric("Results Found", f"{len(display_df):,}")
    with pm2:
        if len(display_df) > 0:
            st.metric("Average Age", f"{display_df['Age'].mean():.0f} yrs")
        else:
            st.metric("Average Age", "—")
    with pm3:
        if len(display_df) > 0:
            st.metric("Avg Cost", f"${display_df['TreatmentCost'].mean():,.0f}")
        else:
            st.metric("Avg Cost", "—")
    with pm4:
        if len(display_df) > 0:
            st.metric("Avg Stay", f"{display_df['LengthOfStay'].mean():.0f} days")
        else:
            st.metric("Avg Stay", "—")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Display table
    show_cols = ["PatientID", "Name", "Age", "Gender", "DiagnosisName", "OutcomeName",
                 "TreatmentCost", "LengthOfStay", "AdmissionDate", "DischargeDate"]
    available_cols = [c for c in show_cols if c in display_df.columns]

    if len(display_df) > 0:
        styled_df = display_df[available_cols].copy()
        styled_df.columns = [
            "ID", "Patient Name", "Age", "Gender", "Diagnosis", "Outcome",
            "Treatment Cost ($)", "Stay (days)", "Admitted", "Discharged"
        ][:len(available_cols)]

        st.dataframe(
            styled_df,
            use_container_width=True,
            height=500,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "Treatment Cost ($)": st.column_config.NumberColumn("Treatment Cost ($)", format="$%d"),
                "Outcome": st.column_config.TextColumn("Outcome", width="medium"),
            }
        )
    else:
        st.info("No patients found matching your criteria. Try adjusting the filters.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_model:
    st.markdown('<div class="section-header"><span class="section-icon">🧠</span><h2>Model Performance & Insights</h2></div>', unsafe_allow_html=True)

    mi1, mi2, mi3 = st.columns(3)

    with mi1:
        st.markdown("""
        <div class="result-card">
            <div class="result-title">Algorithm</div>
            <div style="font-size:1.5rem;font-weight:800;color:var(--primary)">Logistic Regression</div>
            <div class="result-subtitle">Scikit-learn v1.7+ · Binary Classification</div>
        </div>
        """, unsafe_allow_html=True)

    with mi2:
        st.markdown("""
        <div class="result-card">
            <div class="result-title">Features Used</div>
            <div style="font-size:1.5rem;font-weight:800;color:var(--primary)">3 Features</div>
            <div class="result-subtitle">Age · Length of Stay · Treatment Cost</div>
        </div>
        """, unsafe_allow_html=True)

    with mi3:
        st.markdown("""
        <div class="result-card">
            <div class="result-title">Model Type</div>
            <div style="font-size:1.5rem;font-weight:800;color:var(--primary)">Binary Classifier</div>
            <div class="result-subtitle">High Risk vs Low Risk · Probability Output</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    model_col1, model_col2 = st.columns(2)

    with model_col1:
        # Feature Importance (from model coefficients)
        feature_names = ["Age", "Length of Stay", "Treatment Cost"]
        coefficients = model.coef_[0]
        abs_coeff = np.abs(coefficients)
        importance = abs_coeff / abs_coeff.sum() * 100

        fig_importance = go.Figure(go.Bar(
            x=importance,
            y=feature_names,
            orientation="h",
            marker=dict(
                color=["#1A73E8", "#4285F4", "#8AB4F8"],
                cornerradius=6
            ),
            text=[f"{v:.1f}%" for v in importance],
            textposition="outside",
            textfont=dict(size=13, family="Inter", weight=700)
        ))
        fig_importance.update_layout(
            title=dict(text="Feature Importance (|Coefficients|)", font=dict(size=16, family="Inter", weight=700)),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Relative Importance (%)", gridcolor="#E8EAED", range=[0, max(importance) * 1.3]),
            yaxis=dict(title=""),
            height=350,
            margin=dict(t=50, b=50, l=120, r=60),
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    with model_col2:
        # Model coefficient values
        coeff_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Direction": ["Increases Risk" if c > 0 else "Decreases Risk" for c in coefficients]
        })

        fig_coeff = go.Figure(go.Bar(
            x=coeff_df["Feature"],
            y=coeff_df["Coefficient"],
            marker=dict(
                color=[("#EA4335" if c > 0 else "#34A853") for c in coefficients],
                cornerradius=6
            ),
            text=[f"{v:.4f}" for v in coefficients],
            textposition="outside",
            textfont=dict(size=12, family="Inter", weight=600)
        ))
        fig_coeff.update_layout(
            title=dict(text="Model Coefficients (Direction of Effect)", font=dict(size=16, family="Inter", weight=700)),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Feature", gridcolor="#E8EAED"),
            yaxis=dict(title="Coefficient Value", gridcolor="#E8EAED"),
            height=350,
            margin=dict(t=50, b=50, l=60, r=30),
            font=dict(family="Inter")
        )
        st.plotly_chart(fig_coeff, use_container_width=True)

    # Decision Boundary Visualization
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Generate prediction surface for Age vs Treatment Cost
    age_range = np.linspace(patients_df["Age"].min(), patients_df["Age"].max(), 50)
    cost_range = np.linspace(patients_df["TreatmentCost"].min(), patients_df["TreatmentCost"].max(), 50)
    age_grid, cost_grid = np.meshgrid(age_range, cost_range)
    median_los = int(merged_df["LengthOfStay"].median()) if "LengthOfStay" in merged_df.columns else 7

    grid_data = pd.DataFrame({
        "Age": age_grid.ravel(),
        "LengthOfStay": np.full(age_grid.ravel().shape, median_los),
        "TreatmentCost": cost_grid.ravel()
    })

    grid_probs = model.predict_proba(grid_data)[:, 1].reshape(age_grid.shape)

    fig_surface = go.Figure(go.Contour(
        x=age_range,
        y=cost_range,
        z=grid_probs,
        colorscale=[[0, "#34A853"], [0.5, "#FBBC04"], [1, "#EA4335"]],
        contours=dict(showlabels=True, labelfont=dict(size=11, family="Inter")),
        colorbar=dict(title=dict(text="Risk Probability", font=dict(family="Inter"))),
        hovertemplate="Age: %{x:.0f}<br>Cost: $%{y:,.0f}<br>Risk: %{z:.2%}<extra></extra>"
    ))
    fig_surface.update_layout(
        title=dict(
            text=f"Risk Prediction Surface (Age vs Treatment Cost, LOS={median_los} days)",
            font=dict(size=16, family="Inter", weight=700)
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Patient Age", gridcolor="#E8EAED"),
        yaxis=dict(title="Treatment Cost ($)", gridcolor="#E8EAED"),
        height=450,
        margin=dict(t=60, b=50, l=70, r=30),
        font=dict(family="Inter")
    )
    st.plotly_chart(fig_surface, use_container_width=True)

    # Model Details Expander
    with st.expander("📄 Technical Model Details"):
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | **Algorithm** | Logistic Regression |
        | **Solver** | LBFGS |
        | **Regularization** | L2 (C=5.0) |
        | **Max Iterations** | 200 |
        | **Features** | Age, LengthOfStay, TreatmentCost |
        | **Classes** | 0 (Low Risk), 1 (High Risk) |
        | **Intercept** | {model.intercept_[0]:.6f} |
        | **Coefficients** | Age={coefficients[0]:.6f}, LOS={coefficients[1]:.6f}, Cost={coefficients[2]:.6f} |
        """)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    <strong>HealthGuard AI</strong> · Enterprise Healthcare Risk Prediction System<br/>
    Built with Streamlit · Scikit-learn · Plotly · Python<br/>
    <span style="opacity:0.6">© {datetime.now().year} Healthcare Risk Prediction System. For authorized clinical use only.</span>
</div>
""", unsafe_allow_html=True)