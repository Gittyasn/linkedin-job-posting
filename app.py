import streamlit as st
import matplotlib
matplotlib.use('Agg') # Headless mode

import pandas as pd
import sys
import os
import plotly.express as px
import requests
from streamlit_lottie import st_lottie

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import load_data, clean_data
from eda_plots import plot_general_insights, plot_salary_distribution
from ml_models import train_salary_model, predict_salary, load_saved_model
from nlp_analysis import generate_wordcloud, get_ngrams

# -----------------------------------------------------------------------------
# Page Config & Styling
# -----------------------------------------------------------------------------
st.set_page_config(page_title="LinkedIn Market Intelligence", layout="wide", page_icon="💼", initial_sidebar_state="expanded")

# CUSTOM CSS: Glassmorphism & Dark Theme
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    
    /* Metrics Cards (Glassmorphism) */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0px 0px;
        color: #ddd;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        font-weight: bold;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.3);
    }
    
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

# Load Assets (Animations)
lottie_analytics = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_robot = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_m9zragwd.json")
lottie_search = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_2Ldd0o.json")

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    try:
        df = load_data('postings.csv')
        if df is not None:
            df = clean_data(df)
        return df
    except Exception:
        return None

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/174/174857.png", width=50)
    st.title("Intelligence Hub")
    st.markdown("---")
    
    # Load Data indicator
    data_load_state = st.empty()
    data_load_state.info("Initializing...")
    
    df = get_data()
    
    if df is not None:
        data_load_state.success(f"Live Data: {len(df):,} jobs")
    else:
        data_load_state.error("Data Failed")
        st.stop()

    # Filters
    st.subheader("Global Filters")
    all_titles = ["All"] + sorted(df['clean_title'].unique().tolist())
    selected_title = st.selectbox("Job Title Analysis", all_titles)
    
    if 'location' in df.columns:
        all_locs = ["All"] + sorted(df['location'].unique().tolist())
        selected_loc = st.selectbox("Location Filter", all_locs)
    else:
        selected_loc = "All"
        
    st.markdown("---")
    st.caption("v4.0 Premium Edition")

# Filtering Matches
filtered_df = df.copy()
if selected_title != "All":
    filtered_df = filtered_df[filtered_df['clean_title'] == selected_title]
if selected_loc != "All":
    filtered_df = filtered_df[filtered_df['location'] == selected_loc]

# Load Model
if 'model' not in st.session_state:
    saved_model, saved_encoders, saved_metrics = load_saved_model()
    if saved_model:
        st.session_state['model'] = saved_model
        st.session_state['encoders'] = saved_encoders
        st.session_state['metrics'] = saved_metrics

# -----------------------------------------------------------------------------
# Main Content
# -----------------------------------------------------------------------------
c1, c2 = st.columns([3, 1])
with c1:
    st.title("Market Intelligence Suite")
    st.markdown("### Next-Gen Job Market Analysis & Prediction")
with c2:
    if lottie_analytics:
        st_lottie(lottie_analytics, height=150, key="header_anim")

# Tabs
tab_overview, tab_ml_explain, tab_nlp, tab_predict = st.tabs([
    "📈 Executive Dashboard", 
    "🧠 Explainable AI", 
    "🗣️ Skill Intelligence", 
    "💰 Career Simulator"
])

# --- TAB 1: EXECUTIVE ---
with tab_overview:
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jobs Analyzed", f"{len(filtered_df):,}")
    k2.metric("Market Depth (Titles)", f"{filtered_df['clean_title'].nunique():,}" if not filtered_df.empty else "0")
    
    # Safe Dominant Region Calculation
    if not filtered_df.empty and 'location' in filtered_df.columns:
        mode_val = filtered_df['location'].mode()
        dom_reg = mode_val[0] if not mode_val.empty else "N/A"
    else:
        dom_reg = "N/A"
    k3.metric("Dominant Region", dom_reg)
    
    # Safe Salary Average
    if not filtered_df.empty and 'normalized_salary' in filtered_df.columns:
        avg_sal = filtered_df['normalized_salary'].mean()
    else:
        avg_sal = 0
    k4.metric("Avg Annual Comp", f"${avg_sal:,.0f}" if avg_sal > 0 else "N/A")
    
    st.markdown("---")
    
    # Grid Layout
    r1c1, r1c2 = st.columns([2, 1])
    
    plots = plot_general_insights(filtered_df, top_n=15)
    
    with r1c1:
        st.markdown("#### Deployment Trends")
        if 'titles' in plots: st.plotly_chart(plots['titles'], width='stretch')
 
    with r1c2:
        st.markdown("#### Work Structure")
        if 'work_type' in plots: st.plotly_chart(plots['work_type'], width='stretch')

    # Salary Curve
    st.markdown("#### Compensation Distribution")
    sal_fig = plot_salary_distribution(filtered_df)
    if sal_fig: st.plotly_chart(sal_fig, width='stretch')

# --- TAB 2: AI ---
with tab_ml_explain:
    col_ai_text, col_ai_anim = st.columns([3, 1])
    with col_ai_text:
        st.header("🧠 Model Internals")
        st.markdown("We use a **Random Forest Regressor** to determine salary. This chart explains *how* it thinks.")
    with col_ai_anim:
        if lottie_robot: st_lottie(lottie_robot, height=120)

    if 'metrics' in st.session_state:
        metrics = st.session_state['metrics']
        
        # Feature Importance
        if 'importance' in metrics:
            imp_df = metrics['importance']
            fig = px.bar(imp_df, x='importance', y='feature', orientation='h', 
                         title="Feature Permutation Importance",
                         template='plotly_dark',
                         color='importance', color_continuous_scale='Bluered')
            fig.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')

            with st.expander("Technical Model Metrics"):
                c1, c2 = st.columns(2)
                c1.metric("R² Accuracy", f"{metrics['r2']:.1%}")
                c2.metric("MAE Error", f"${metrics['mae']:,.0f}")
        else:
            st.warning("Model explanation data missing.")
    else:
        st.error("AI Model Offline.")

# --- TAB 3: SKILLS ---
with tab_nlp:
    c_nlp_1, c_nlp_2 = st.columns([3, 1])
    with c_nlp_1:
         st.header("🗣️ Linguistic Analysis")
         st.markdown("Extract patterns from descriptions with customizable depth.")
         n_gram_sel = st.slider("Phrase Intensity", 1, 3, 2)
    with c_nlp_2:
        if lottie_search: st_lottie(lottie_search, height=100)
    
    # N-Grams
    if len(filtered_df) > 0:
        ngrams = get_ngrams(filtered_df, n=n_gram_sel, top_k=15)
        if ngrams is not None:
            fig = px.bar(ngrams, x='Count', y='Phrase', orientation='h', 
                        title=f"{n_gram_sel}-gram Skill Extraction in {selected_title}",
                        template='plotly_dark',
                        color='Count', color_continuous_scale='Viridis')
            fig.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, width='stretch')
        else:
             st.info("Insufficient text data for this depth.")
    
    with st.expander("Generate AI Word Cloud"):
        if st.button("Generate Cloud"):
            with st.spinner("Processing text..."):
                fig_cloud = generate_wordcloud(filtered_df, title_filter=selected_title if selected_title != 'All' else None)
                if fig_cloud: st.pyplot(fig_cloud)

# --- TAB 4: SIMULATOR ---
with tab_predict:
    st.header("💰 Compensation Simulator")
    
    if 'model' in st.session_state:
        # PRE-FILTERED LISTS (Only show what the model knows!)
        trained_roles = sorted(st.session_state['encoders']['clean_title'].classes_.tolist())
        trained_locs = sorted(st.session_state['encoders']['location'].classes_.tolist())
        trained_formats = sorted(st.session_state['encoders']['formatted_work_type'].classes_.tolist())

        # Glassmorphism Card for Inputs
        with st.container():
            st.markdown('<div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1);">', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Current Path")
                role_a = st.selectbox("Role", trained_roles, key="rolesim_a")
                loc_a = st.selectbox("Location", trained_locs, key="locsim_a")
                type_a = st.selectbox("Format", trained_formats, key="typesim_a")
                
                val_a = predict_salary(st.session_state['model'], st.session_state['encoders'], role_a, loc_a, type_a)
                if val_a: 
                    st.metric("Projected", f"${val_a:,.0f}")
                else:
                    st.warning("Prediction Offline for this selection")

            with c2:
                st.subheader("Future Path")
                role_b = st.selectbox("Role", trained_roles, key="rolesim_b")
                loc_b = st.selectbox("Location", trained_locs, key="locsim_b")
                type_b = st.selectbox("Format", trained_formats, key="typesim_b")
                
                val_b = predict_salary(st.session_state['model'], st.session_state['encoders'], role_b, loc_b, type_b)
                if val_b: 
                    st.metric("Projected", f"${val_b:,.0f}", delta=f"${val_b - (val_a if val_a else 0):,.0f}")
                else:
                    st.warning("Prediction Offline for this selection")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("ML Simulator requires model to be loaded. Please ensure src/model.pkl exists.")
