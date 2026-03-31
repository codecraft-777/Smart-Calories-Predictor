import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCalories",
    page_icon="🔥",
    layout="centered"
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #0d0d0d; }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 3rem;
        max-width: 860px;
    }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #0f2027, #1a3a2a, #0f2027);
        border-radius: 0 0 28px 28px;
        padding: 3rem 2rem 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; left: -60px;
        width: 240px; height: 240px;
        background: radial-gradient(circle, rgba(34,197,94,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -40px; right: -40px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(34,197,94,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(34,197,94,0.15);
        border: 1px solid rgba(34,197,94,0.4);
        color: #4ade80;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0.3rem 0.9rem;
        border-radius: 99px;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-family: 'Syne', sans-serif !important;
        font-size: 3.2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em;
        margin: 0 0 0.4rem !important;
        line-height: 1.1;
        text-align: center !important;
    }
    .hero h1 span { color: #4ade80; }
    .hero-sub {
        color: #9ca3af;
        font-size: 0.95rem;
        font-weight: 300;
        margin: 0;
    }

    /* ── Cards ── */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .card-dark {
        background: #111827;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        border: 1px solid #1f2937;
    }
    .card-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card-title-dark {
        font-family: 'Syne', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #4ade80;
        margin-bottom: 1.1rem;
    }

    /* ── Result ── */
    .result-hero {
        background: linear-gradient(135deg, #052e16, #14532d);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.2rem 0;
        border: 1px solid #166534;
        position: relative;
        overflow: hidden;
    }
    .result-hero::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 150px; height: 150px;
        background: radial-gradient(circle, rgba(74,222,128,0.2) 0%, transparent 70%);
    }
    .result-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #86efac;
        margin-bottom: 0.3rem;
    }
    .result-number {
        font-family: 'Syne', sans-serif;
        font-size: 3.4rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
    }
    .result-unit {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        color: #86efac;
        margin-top: 0.2rem;
    }

    /* ── BMI pills ── */
    .bmi-grid {
        display: flex;
        gap: 0.8rem;
        margin-top: 0.5rem;
    }
    .bmi-pill {
        flex: 1;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 0.9rem;
        text-align: center;
    }
    .bmi-pill-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
    }
    .bmi-pill-lbl {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.1rem;
    }
    .bmi-pill-cat {
        display: inline-block;
        margin-top: 0.3rem;
        padding: 0.15rem 0.6rem;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 600;
        background: #dcfce7;
        color: #166534;
    }

    /* ── Insights stat cards ── */
    .stat-row {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1.2rem;
    }
    .stat-card {
        flex: 1;
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 1.1rem 1rem;
        text-align: center;
    }
    .stat-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #4ade80;
    }
    .stat-lbl {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.2rem;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #f3f4f6;
        border-radius: 12px;
        padding: 4px;
        gap: 2px;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px !important;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        color: #6b7280 !important;
        padding: 0.45rem 1.2rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #111827 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ── Inputs ── */
    .stNumberInput input, .stSelectbox select {
        border-radius: 8px !important;
        border: 1px solid #d1d5db !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    label { font-family: 'DM Sans', sans-serif !important; font-size: 0.88rem !important; }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #15803d, #166534) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em;
        width: 100% !important;
        border: none !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    /* ── Alerts ── */
    .stInfo, .stSuccess {
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Divider ── */
    hr { border-color: #e5e7eb !important; margin: 1.5rem 0 !important; }

    /* ── Table ── */
    .stTable table {
        border-radius: 10px;
        overflow: hidden;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Train model ───────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    exercise = pd.read_csv("exercise.csv")
    calories = pd.read_csv("calories.csv")
    df = pd.merge(exercise, calories, on="User_ID", how="inner")
    df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})

    X = df[['Gender', 'Age', 'Weight', 'Height', 'Duration', 'Heart_Rate', 'Body_Temp']]
    y = df['Calories']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, df

model, scaler, df = train_model()
FEATURE_NAMES = ['Gender', 'Age', 'Weight', 'Height', 'Duration', 'Heart Rate', 'Body Temp']

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🔥 Random Forest ML</div>
    <h1>Smart<span>Calories</span></h1>
    <p class="hero-sub">Predict your calorie burn with machine learning precision</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["✦  Manual Prediction", "⊞  Bulk Scanner", "◈  Model Insights"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Manual Prediction
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">👤 Personal Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        gender     = st.selectbox("Gender", ["Male", "Female"])
        age        = st.number_input("Age (years)",   min_value=10,    max_value=100,   value=25,    step=1)
        weight     = st.number_input("Weight (kg)",   min_value=30.0,  max_value=200.0, value=70.0,  step=0.5)
        height     = st.number_input("Height (cm)",   min_value=100.0, max_value=250.0, value=170.0, step=0.5)
    with col2:
        duration   = st.number_input("Duration (min)",min_value=1,     max_value=300,   value=30,    step=1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=220,   value=100,   step=1)
        body_temp  = st.number_input("Body Temp (°C)", min_value=35.0, max_value=43.0,  value=37.5,  step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    # BMI card
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:   bmi_label = "Underweight"
    elif bmi < 25:   bmi_label = "Normal"
    elif bmi < 30:   bmi_label = "Overweight"
    else:            bmi_label = "Obese"

    st.markdown(f"""
    <div class="card">
        <div class="card-title">⚖️ BMI Calculator</div>
        <div class="bmi-grid">
            <div class="bmi-pill">
                <div class="bmi-pill-val">{bmi:.1f}</div>
                <div class="bmi-pill-lbl">BMI Score</div>
                <div class="bmi-pill-cat">{bmi_label}</div>
            </div>
            <div class="bmi-pill" style="flex:2; text-align:left; display:flex; flex-direction:column; justify-content:center; padding-left:1.2rem;">
                <div style="font-size:0.83rem; color:#374151; line-height:1.6;">
                    <span style="color:#6b7280;">Underweight</span> &nbsp;< 18.5<br>
                    <span style="color:#6b7280;">Normal</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;18.5 – 24.9<br>
                    <span style="color:#6b7280;">Overweight</span> &nbsp;25.0 – 29.9<br>
                    <span style="color:#6b7280;">Obese</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;≥ 30.0
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    if st.button("🔥 Predict Calories Burned", key="predict_manual"):
        gender_encoded = 0 if gender == "Male" else 1
        input_data = np.array([[gender_encoded, age, weight, height, duration, heart_rate, body_temp]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.markdown(f"""
        <div class="result-hero">
            <div class="result-label">Estimated Calories Burned</div>
            <div class="result-number">{prediction:.1f}</div>
            <div class="result-unit">kcal</div>
        </div>
        """, unsafe_allow_html=True)

        if prediction < 100:
            st.info("💧 Light activity — suitable for warm-up sessions.")
        elif prediction < 250:
            st.info("🚶 Moderate workout — good effort.")
        elif prediction < 400:
            st.success("🏃 Solid training session — well done!")
        else:
            st.success("🔥 High intensity workout — excellent performance!")

        # Comparison chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Calorie Comparison</div>', unsafe_allow_html=True)

        avg_cal  = df['Calories'].mean()
        low_cal  = df['Calories'].quantile(0.25)
        high_cal = df['Calories'].quantile(0.75)

        labels  = ['Low (25th %)', 'Average', 'High (75th %)', 'Your Burn']
        values  = [low_cal, avg_cal, high_cal, prediction]
        colors  = ['#bfdbfe', '#bbf7d0', '#fde68a', '#4ade80']

        fig, ax = plt.subplots(figsize=(7, 2.8))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f9fafb')
        bars = ax.barh(labels, values, color=colors, height=0.52, edgecolor='none')
        ax.set_xlabel("Calories (kcal)", fontsize=9, color='#6b7280')
        ax.tick_params(colors='#374151', labelsize=9)
        ax.spines[['top','right','left']].set_visible(False)
        ax.spines['bottom'].set_color('#e5e7eb')
        ax.grid(axis='x', color='#e5e7eb', linewidth=0.6)
        for bar in bars:
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.0f}', va='center', fontsize=9, color='#374151')
        fig.tight_layout(pad=1.2)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Summary table
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 Input Summary</div>', unsafe_allow_html=True)
        summary_df = pd.DataFrame({
            "Feature": ["Gender", "Age", "Weight (kg)", "Height (cm)", "Duration (min)",
                        "Heart Rate (bpm)", "Body Temp (°C)", "BMI"],
            "Value":   [gender, age, weight, height, duration, heart_rate, body_temp, f"{bmi:.1f}"]
        })
        st.table(summary_df)
        st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bulk Scanner
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<div class="card-title">⬇️ Sample File</div>', unsafe_allow_html=True)
        sample = pd.DataFrame({
            'Gender':     ['male', 'female'],
            'Age':        [25, 30],
            'Weight':     [70.0, 60.0],
            'Height':     [175.0, 162.0],
            'Duration':   [30, 45],
            'Heart_Rate': [100, 110],
            'Body_Temp':  [37.5, 37.8]
        })
        csv_sample = sample.to_csv(index=False).encode('utf-8')
        st.download_button("Download Sample CSV", csv_sample, "sample_input.csv", "text/csv", use_container_width=True)

    with col_b:
        st.markdown('<div class="card-title">⬆️ Upload File</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    with col_c:
        st.markdown('<div class="card-title">📥 Download Results</div>', unsafe_allow_html=True)
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                input_df['Gender'] = input_df['Gender'].map({'male': 0, 'female': 1})
                X_bulk = input_df[['Gender', 'Age', 'Weight', 'Height', 'Duration', 'Heart_Rate', 'Body_Temp']]
                X_bulk_scaled = scaler.transform(X_bulk)
                predictions = model.predict(X_bulk_scaled)
                input_df['Gender'] = input_df['Gender'].map({0: 'male', 1: 'female'})
                input_df['Predicted_Calories'] = predictions.round(1)
                result_csv = input_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results CSV", result_csv, "predicted_calories.csv", "text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Upload a file first.")

    if uploaded_file is not None:
        try:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Prediction Results</div>', unsafe_allow_html=True)
            st.dataframe(input_df, use_container_width=True)
            st.success(f"✅ Predictions complete for {len(input_df)} records.")
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            pass

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights (dark theme)
# ═════════════════════════════════════════════════════════════════════════════
with tab3:

    # Stat cards
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-val">{len(df):,}</div>
            <div class="stat-lbl">Total Records</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{df['Calories'].mean():.1f}</div>
            <div class="stat-lbl">Avg kcal Burned</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{df['Calories'].max():.0f}</div>
            <div class="stat-lbl">Max kcal Burned</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance chart
    st.markdown('<div class="card-title" style="color:#2e7d32;">◈ Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem; color:#6b7280; margin-bottom:1rem;">Which inputs drive calorie predictions most?</p>', unsafe_allow_html=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)

    exercise_features = ['Duration', 'Heart Rate', 'Body Temp']
    colors = ['#4ade80' if FEATURE_NAMES[i] in exercise_features else '#93c5fd' for i in indices]

    fig2, ax2 = plt.subplots(figsize=(7, 3.8))
    fig2.patch.set_facecolor('#ffffff')
    ax2.set_facecolor('#f9fafb')
    bars2 = ax2.barh([FEATURE_NAMES[i] for i in indices], importances[indices],
                     color=colors, height=0.52, edgecolor='none')
    ax2.set_xlabel("Importance Score", fontsize=9, color='#6b7280')
    ax2.tick_params(colors='#374151', labelsize=9)
    ax2.spines[['top','right','left']].set_visible(False)
    ax2.spines['bottom'].set_color('#e5e7eb')
    ax2.grid(axis='x', color='#e5e7eb', linewidth=0.6)
    for bar in bars2:
        ax2.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.3f}', va='center', fontsize=9, color='#374151')

    green_patch = mpatches.Patch(color='#4ade80', label='Exercise factors (controllable)')
    blue_patch  = mpatches.Patch(color='#93c5fd', label='Personal factors (demographic)')
    ax2.legend(handles=[green_patch, blue_patch], fontsize=8.5,
               framealpha=0, labelcolor='#374151', loc='lower right')

    fig2.tight_layout(pad=1.2)
    st.pyplot(fig2)
