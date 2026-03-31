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
    page_icon="",
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main { background-color: #f8f8f6; }

    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 820px;
    }

    /* ── Header ── */
    .app-header {
        margin-bottom: 2.2rem;
    }
    .app-header h1 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: -0.01em;
        margin: 0 0 0.25rem 0;
    }
    .app-header p {
        font-size: 0.875rem;
        color: #777;
        font-weight: 300;
        margin: 0;
    }
    .accent-line {
        width: 36px;
        height: 3px;
        background: #2d6a4f;
        border-radius: 2px;
        margin-bottom: 0.75rem;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #999;
        margin-bottom: 0.85rem;
        margin-top: 1.8rem;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid #e8e8e4;
        margin: 1.6rem 0;
    }

    /* ── Result block ── */
    .result-block {
        background: #1a1a1a;
        border-radius: 10px;
        padding: 1.8rem 2rem;
        margin: 1.4rem 0;
    }
    .result-number {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.8rem;
        font-weight: 500;
        color: #ffffff;
        line-height: 1;
    }
    .result-unit {
        font-size: 1rem;
        color: #777;
        font-weight: 300;
    }
    .result-label {
        font-size: 0.72rem;
        color: #888;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    /* ── BMI row ── */
    .bmi-row {
        display: flex;
        gap: 1rem;
        margin-top: 0.4rem;
    }
    .bmi-item {
        background: #ffffff;
        border: 1px solid #e8e8e4;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        flex: 1;
    }
    .bmi-item-val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 500;
        color: #1a1a1a;
    }
    .bmi-item-lbl {
        font-size: 0.75rem;
        color: #999;
        margin-top: 0.1rem;
    }
    .bmi-tag {
        display: inline-block;
        margin-top: 0.35rem;
        padding: 0.15rem 0.55rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 500;
        background: #e8f5e9;
        color: #2d6a4f;
    }

    /* ── Stat row ── */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.8rem;
    }
    .stat-item {
        background: #ffffff;
        border: 1px solid #e8e8e4;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        flex: 1;
    }
    .stat-val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 500;
        color: #1a1a1a;
    }
    .stat-lbl {
        font-size: 0.75rem;
        color: #999;
        margin-top: 0.15rem;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #e8e8e4;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.875rem;
        font-weight: 400;
        color: #999 !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1a1a1a !important;
        border-bottom: 2px solid #2d6a4f !important;
        font-weight: 500 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ── Inputs ── */
    label {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.84rem !important;
        color: #444 !important;
        font-weight: 400 !important;
    }
    .stNumberInput input, .stSelectbox > div > div {
        border-radius: 6px !important;
        border: 1px solid #ddd !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.9rem !important;
        background: #ffffff !important;
    }

    /* ── Button ── */
    .stButton > button {
        background: #1a1a1a !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.4rem !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        width: 100% !important;
        letter-spacing: 0.01em;
        transition: background 0.2s !important;
    }
    .stButton > button:hover {
        background: #2d6a4f !important;
    }

    /* ── Alerts ── */
    .stInfo > div, .stSuccess > div {
        border-radius: 6px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.875rem !important;
    }

    /* ── Table ── */
    .stTable table {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.85rem !important;
    }
    .stTable th {
        color: #999 !important;
        font-weight: 500 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
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

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="accent-line"></div>
    <h1>SmartCalories</h1>
    <p>Calorie burn prediction powered by Random Forest</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Prediction", "Bulk Scanner", "Model Insights"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Personal Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        gender     = st.selectbox("Gender", ["Male", "Female"])
        age        = st.number_input("Age (years)",      min_value=10,    max_value=100,   value=25,    step=1)
        weight     = st.number_input("Weight (kg)",      min_value=30.0,  max_value=200.0, value=70.0,  step=0.5)
        height     = st.number_input("Height (cm)",      min_value=100.0, max_value=250.0, value=170.0, step=0.5)
    with col2:
        duration   = st.number_input("Duration (min)",   min_value=1,     max_value=300,   value=30,    step=1)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40,    max_value=220,   value=100,   step=1)
        body_temp  = st.number_input("Body Temp (°C)",   min_value=35.0,  max_value=43.0,  value=37.5,  step=0.1)

    # BMI
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:   bmi_label = "Underweight"
    elif bmi < 25:   bmi_label = "Normal"
    elif bmi < 30:   bmi_label = "Overweight"
    else:            bmi_label = "Obese"

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">BMI</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="bmi-row">
        <div class="bmi-item">
            <div class="bmi-item-val">{bmi:.1f}</div>
            <div class="bmi-item-lbl">BMI Score</div>
            <div class="bmi-tag">{bmi_label}</div>
        </div>
        <div class="bmi-item" style="flex:2;">
            <div style="font-size:0.8rem; color:#555; line-height:2.1;">
                Underweight &nbsp;&lt; 18.5 &nbsp;&nbsp;&nbsp;&nbsp;
                Normal &nbsp; 18.5 – 24.9<br>
                Overweight &nbsp; 25.0 – 29.9 &nbsp;&nbsp;&nbsp;&nbsp;
                Obese &nbsp; ≥ 30.0
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if st.button("Predict Calories Burned"):
        gender_encoded = 0 if gender == "Male" else 1
        input_data     = np.array([[gender_encoded, age, weight, height, duration, heart_rate, body_temp]])
        input_scaled   = scaler.transform(input_data)
        prediction     = model.predict(input_scaled)[0]

        st.markdown(f"""
        <div class="result-block">
            <div class="result-label">Estimated Calories Burned</div>
            <div style="display:flex; align-items:baseline; gap:0.5rem;">
                <div class="result-number">{prediction:.1f}</div>
                <div class="result-unit">kcal</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if prediction < 100:
            st.info("Light activity — suitable for warm-up sessions.")
        elif prediction < 250:
            st.info("Moderate workout — good effort.")
        elif prediction < 400:
            st.success("Solid training session — well done.")
        else:
            st.success("High intensity workout — excellent performance.")

        # Comparison chart
        st.markdown('<div class="section-label">Comparison</div>', unsafe_allow_html=True)
        avg_cal  = df['Calories'].mean()
        low_cal  = df['Calories'].quantile(0.25)
        high_cal = df['Calories'].quantile(0.75)

        labels = ['25th Percentile', 'Average', '75th Percentile', 'Your Result']
        values = [low_cal, avg_cal, high_cal, prediction]
        colors = ['#d1e8d5', '#a8d5b5', '#7fbe93', '#2d6a4f']

        fig, ax = plt.subplots(figsize=(7, 2.6))
        fig.patch.set_facecolor('#f8f8f6')
        ax.set_facecolor('#f8f8f6')
        bars = ax.barh(labels, values, color=colors, height=0.45, edgecolor='none')
        ax.set_xlabel("Calories (kcal)", fontsize=8.5, color='#888')
        ax.tick_params(colors='#444', labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis='x', color='#e8e8e4', linewidth=0.7)
        ax.tick_params(axis='y', length=0)
        for bar in bars:
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    f'{bar.get_width():.0f}', va='center', fontsize=8.5, color='#444')
        fig.tight_layout(pad=1.0)
        st.pyplot(fig)

        # Input summary
        st.markdown('<div class="section-label">Input Summary</div>', unsafe_allow_html=True)
        st.table(pd.DataFrame({
            "Feature": ["Gender", "Age", "Weight (kg)", "Height (cm)",
                        "Duration (min)", "Heart Rate (bpm)", "Body Temp (°C)", "BMI"],
            "Value":   [gender, age, weight, height, duration, heart_rate, body_temp, f"{bmi:.1f}"]
        }))

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bulk Scanner
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<div class="section-label">Sample File</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-label">Upload File</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    with col_c:
        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
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
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)
            st.dataframe(input_df, use_container_width=True)
            st.success(f"Predictions complete for {len(input_df)} records.")
        except:
            pass

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-item">
            <div class="stat-val">{len(df):,}</div>
            <div class="stat-lbl">Training Records</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">{df['Calories'].mean():.1f}</div>
            <div class="stat-lbl">Avg Calories Burned</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">{df['Calories'].max():.0f}</div>
            <div class="stat-lbl">Max Calories Burned</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.83rem; color:#888; margin-bottom:1rem; font-weight:300;">Which inputs the model relies on most when predicting calorie burn.</p>', unsafe_allow_html=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)
    exercise_features = ['Duration', 'Heart Rate', 'Body Temp']
    colors = ['#2d6a4f' if FEATURE_NAMES[i] in exercise_features else '#b7d4c8' for i in indices]

    fig2, ax2 = plt.subplots(figsize=(7, 3.6))
    fig2.patch.set_facecolor('#f8f8f6')
    ax2.set_facecolor('#f8f8f6')
    bars2 = ax2.barh([FEATURE_NAMES[i] for i in indices], importances[indices],
                     color=colors, height=0.45, edgecolor='none')
    ax2.set_xlabel("Importance Score", fontsize=8.5, color='#888')
    ax2.tick_params(colors='#444', labelsize=8.5)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.grid(axis='x', color='#e8e8e4', linewidth=0.7)
    ax2.tick_params(axis='y', length=0)
    for bar in bars2:
        ax2.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.3f}', va='center', fontsize=8.5, color='#444')

    dark_patch  = mpatches.Patch(color='#2d6a4f', label='Exercise factors')
    light_patch = mpatches.Patch(color='#b7d4c8', label='Personal factors')
    ax2.legend(handles=[dark_patch, light_patch], fontsize=8.5,
               framealpha=0, labelcolor='#555', loc='lower right')

    fig2.tight_layout(pad=1.0)
    st.pyplot(fig2)
