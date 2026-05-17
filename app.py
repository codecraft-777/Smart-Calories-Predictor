import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import requests

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SmartCalories", page_icon="🔥", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #0b0e17 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
        background-color: #0b0e17 !important;
    }
    section[data-testid="stSidebar"] { display: none; }
    #MainMenu, header, footer { visibility: hidden; }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(135deg, #0d1f2d 0%, #1a3a4a 50%, #0f2535 100%);
        border: 0.5px solid #1e3a4a;
        border-radius: 18px;
        padding: 1.1rem 1.6rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-left h1 { font-size: 20px; font-weight: 700; color: #ffffff; margin: 0; letter-spacing: -0.3px; }
    .hero-left p  { font-size: 12px; color: rgba(255,255,255,0.4); margin: 3px 0 0 0; }
    .hero-badge {
        background: rgba(255,255,255,0.06);
        border: 0.5px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11px;
        color: rgba(255,255,255,0.4);
    }

    /* ── Section label ── */
    .section-label {
        font-size: 10px;
        font-weight: 600;
        color: #3d4560;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 0.5px solid #181d2e;
    }

    /* ── Card ── */
    .card {
        background: #131826;
        border: 0.5px solid #1e2438;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .card-title {
        font-size: 10px;
        font-weight: 600;
        color: #3d4560;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.8rem;
    }

    /* ── Metric boxes ── */
    .metric-row { display: flex; gap: 8px; }
    .metric-box {
        flex: 1;
        background: #0e1120;
        border-radius: 10px;
        padding: 10px 12px;
        border: 0.5px solid #1a1f30;
    }
    .metric-lbl { font-size: 10px; color: #3d4560; margin-bottom: 3px; }
    .metric-val { font-size: 18px; font-weight: 600; color: #dde1f0; }
    .metric-val.green { color: #34d399; }
    .metric-val.amber { color: #fbbf24; }
    .metric-val.red   { color: #f87171; }

    /* ── Result card ── */
    .result-hero {
        background: linear-gradient(135deg, #0d1f2d, #162f3e);
        border: 0.5px solid #1e3a4a;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-top: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .result-label { font-size: 11px; color: #4a6070; margin-bottom: 2px; }
    .result-value { font-size: 38px; font-weight: 700; color: #ffffff; letter-spacing: -1px; }
    .result-tag   { font-size: 12px; color: #34d399; margin-top: 4px; }

    /* ── Summary rows ── */
    .summary-row {
        display: flex;
        justify-content: space-between;
        padding: 5px 0;
        border-bottom: 0.5px solid #181d2e;
        font-size: 12px;
    }
    .summary-row:last-child { border-bottom: none; }
    .s-key { color: #3d4560; }
    .s-val { color: #dde1f0; font-weight: 500; }

    /* ── Workout plan ── */
    .plan-header {
        background: linear-gradient(135deg, #0d1f2d, #162f3e);
        border: 0.5px solid #1e3a4a;
        border-radius: 14px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.8rem;
    }
    .plan-title { font-size: 16px; font-weight: 700; color: #fff; margin: 0 0 2px 0; }
    .plan-sub   { font-size: 12px; color: rgba(255,255,255,0.35); }
    .goal-chip  {
        display: inline-block;
        background: rgba(52,211,153,0.08);
        border: 0.5px solid rgba(52,211,153,0.25);
        color: #34d399;
        font-size: 11px;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 6px;
    }
    .plan-body {
        background: #131826;
        border: 0.5px solid #1e2438;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        font-size: 13px;
        color: #c0c6dc;
        line-height: 1.8;
        white-space: pre-wrap;
    }

    /* ── Streamlit widget overrides ── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: #0e1120 !important;
        color: #dde1f0 !important;
        border-color: #1e2438 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
    }
    label[data-testid="stWidgetLabel"] p {
        color: #5a6080 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    .stTextInput > div > div > input {
        background-color: #0e1120 !important;
        color: #dde1f0 !important;
        border-color: #1e2438 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #162f3e, #1e4a5e);
        color: #e0e8f0;
        border: 0.5px solid #254a5e;
        border-radius: 10px;
        height: 42px;
        font-size: 13px;
        font-weight: 600;
        width: 100%;
        letter-spacing: 0.2px;
        transition: all 0.15s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e4a5e, #2a5e72);
        border-color: #2a5a6e;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #131826;
        border-radius: 10px;
        padding: 3px;
        gap: 3px;
        border: 0.5px solid #1e2438;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        font-size: 12px;
        font-weight: 500;
        color: #3d4560;
        padding: 6px 14px;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #0e1120 !important;
        color: #dde1f0 !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stDataFrame { background: #131826 !important; border-radius: 10px; }
    .stAlert { border-radius: 10px; }
    p, span, div { color: #dde1f0; }
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

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <h1>🔥 SmartCalories</h1>
        <p>AI-powered calorie prediction & personalized workout planning</p>
    </div>
    <div class="hero-badge">ML · Random Forest · Claude AI</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Manual Prediction", "📁 Bulk Scanner", "🧠 Model Insights", "🏋️ Workout Planner"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — Manual Prediction
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Personal Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age    = st.number_input("Age (years)",  min_value=10,    max_value=100,   value=25,    step=1)
    with col2:
        weight = st.number_input("Weight (kg)",  min_value=30.0,  max_value=200.0, value=70.0,  step=0.5)
        height = st.number_input("Height (cm)",  min_value=100.0, max_value=250.0, value=170.0, step=0.5)

    st.markdown('<div class="section-label">Exercise Details</div>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        duration   = st.number_input("Duration (min)",   min_value=1,   max_value=300, value=30,   step=1)
    with col4:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40,  max_value=220, value=100,  step=1)
    with col5:
        body_temp  = st.number_input("Body Temp (°C)",   min_value=35.0, max_value=43.0, value=37.5, step=0.1)

    # BMI
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:   bmi_label, bmi_color = "Underweight", "amber"
    elif bmi < 25:   bmi_label, bmi_color = "Normal",      "green"
    elif bmi < 30:   bmi_label, bmi_color = "Overweight",  "amber"
    else:            bmi_label, bmi_color = "Obese",        "red"

    st.markdown(f"""
    <div class="card" style="margin-top:0.4rem;">
        <div class="card-title">BMI Calculator</div>
        <div class="metric-row">
            <div class="metric-box"><div class="metric-lbl">BMI Score</div><div class="metric-val">{bmi:.1f}</div></div>
            <div class="metric-box"><div class="metric-lbl">Category</div><div class="metric-val {bmi_color}">{bmi_label}</div></div>
            <div class="metric-box"><div class="metric-lbl">Ideal Range</div><div class="metric-val">18.5–25</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
    if st.button("Predict Calories Burned", key="predict"):
        gender_encoded = 0 if gender == "Male" else 1
        input_data   = np.array([[gender_encoded, age, weight, height, duration, heart_rate, body_temp]])
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]

        if prediction < 100:   msg = "Light activity — suitable for warm-up sessions."
        elif prediction < 250: msg = "Moderate workout — good effort."
        elif prediction < 400: msg = "Solid training session — well done."
        else:                  msg = "High intensity workout — excellent performance."

        st.markdown(f"""
        <div class="result-hero">
            <div>
                <div class="result-label">Estimated calories burned</div>
                <div class="result-value">{prediction:.1f} <span style="font-size:15px;font-weight:400;color:#3d5060">kcal</span></div>
                <div class="result-tag">{msg}</div>
            </div>
            <div style="font-size:44px;">🔥</div>
        </div>
        """, unsafe_allow_html=True)

        avg_cal  = df['Calories'].mean()
        low_cal  = df['Calories'].quantile(0.25)
        high_cal = df['Calories'].quantile(0.75)
        max_val  = max(prediction, high_cal) * 1.15

        st.markdown('<div class="card" style="margin-top:0.8rem;"><div class="card-title">Calorie Comparison</div>', unsafe_allow_html=True)
        labels = ['Low (25th %)', 'Average', 'High (75th %)', 'Your Burn']
        values = [low_cal, avg_cal, high_cal, prediction]
        colors = ['#1e4a7a', '#1e6a4e', '#7a5e1e', '#7a1e1e']

        fig, ax = plt.subplots(figsize=(8, 2.2))
        fig.patch.set_facecolor('#131826')
        ax.set_facecolor('#131826')
        bars = ax.barh(labels, values, color=colors, height=0.32, edgecolor='none')
        ax.set_xlabel("Calories (kcal)", fontsize=8, color='#3d4560')
        ax.set_xlim(0, max_val)
        for bar in bars:
            ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.1f}', va='center', fontsize=8, color='#6b7494')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(colors='#3d4560', labelsize=8)
        ax.xaxis.label.set_color('#3d4560')
        ax.tick_params(axis='y', colors='#6b7494', labelsize=8)
        fig.tight_layout(pad=0.8)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Input Summary</div>
            <div class="summary-row"><span class="s-key">Gender</span><span class="s-val">{gender}</span></div>
            <div class="summary-row"><span class="s-key">Age</span><span class="s-val">{age} years</span></div>
            <div class="summary-row"><span class="s-key">Weight</span><span class="s-val">{weight} kg</span></div>
            <div class="summary-row"><span class="s-key">Height</span><span class="s-val">{height} cm</span></div>
            <div class="summary-row"><span class="s-key">Duration</span><span class="s-val">{duration} min</span></div>
            <div class="summary-row"><span class="s-key">Heart Rate</span><span class="s-val">{heart_rate} bpm</span></div>
            <div class="summary-row"><span class="s-key">Body Temperature</span><span class="s-val">{body_temp} °C</span></div>
            <div class="summary-row"><span class="s-key">BMI</span><span class="s-val">{bmi:.1f} ({bmi_label})</span></div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — Bulk Scanner
# ════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="card"><div class="card-title">Download Sample File</div>', unsafe_allow_html=True)
        sample = pd.DataFrame({
            'Gender': ['male', 'female'], 'Age': [25, 30],
            'Weight': [70.0, 60.0], 'Height': [175.0, 162.0],
            'Duration': [30, 45], 'Heart_Rate': [100, 110], 'Body_Temp': [37.5, 37.8]
        })
        fmt = st.selectbox("Format", ["CSV", "XLSX", "JSON"], label_visibility="collapsed")
        if fmt == "CSV":
            st.download_button("Download Sample", sample.to_csv(index=False).encode(), "sample_input.csv", "text/csv", use_container_width=True)
        elif fmt == "XLSX":
            import io
            buf = io.BytesIO(); sample.to_excel(buf, index=False)
            st.download_button("Download Sample", buf.getvalue(), "sample_input.xlsx", use_container_width=True)
        else:
            st.download_button("Download Sample", sample.to_json(orient="records").encode(), "sample_input.json", "application/json", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card"><div class="card-title">Upload File to Scan</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drag and drop file here", type=["csv", "xlsx", "xls", "json"],
                                          label_visibility="visible", help="Supported: CSV, XLSX, JSON")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="card"><div class="card-title">Download Results</div>', unsafe_allow_html=True)
        if uploaded_file is not None:
            try:
                fname = uploaded_file.name.lower()
                if fname.endswith(".csv"):             input_df = pd.read_csv(uploaded_file)
                elif fname.endswith((".xlsx", ".xls")): input_df = pd.read_excel(uploaded_file)
                elif fname.endswith(".json"):           input_df = pd.read_json(uploaded_file)
                else:                                   input_df = pd.read_csv(uploaded_file)
                display_df = input_df.copy()
                input_df['Gender'] = input_df['Gender'].map({'male': 0, 'female': 1})
                X_bulk = input_df[['Gender','Age','Weight','Height','Duration','Heart_Rate','Body_Temp']]
                preds  = model.predict(scaler.transform(X_bulk))
                display_df['Predicted_Calories'] = preds.round(1)
                st.download_button("Download Results CSV", display_df.to_csv(index=False).encode(), "results.csv", "text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.caption("Upload a file first.")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            st.markdown('<div class="card"><div class="card-title">Prediction Results</div>', unsafe_allow_html=True)
            st.dataframe(display_df, use_container_width=True)
            st.success(f"✅ Predictions complete for {len(display_df)} records.")
            st.markdown('</div>', unsafe_allow_html=True)
        except: pass

# ════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Dataset Overview</div>
        <div class="metric-row">
            <div class="metric-box"><div class="metric-lbl">Total Records</div><div class="metric-val">{len(df):,}</div></div>
            <div class="metric-box"><div class="metric-lbl">Avg Calories</div><div class="metric-val">{df['Calories'].mean():.1f}</div></div>
            <div class="metric-box"><div class="metric-lbl">Max Calories</div><div class="metric-val">{df['Calories'].max():.1f}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    fi_colors   = ['#7a1e1e' if FEATURE_NAMES[i] in ['Duration', 'Heart Rate', 'Body Temp']
                   else '#1e4a7a' for i in indices]

    fig2, ax2 = plt.subplots(figsize=(8, 2.4))
    fig2.patch.set_facecolor('#131826')
    ax2.set_facecolor('#131826')
    bars2 = ax2.barh([FEATURE_NAMES[i] for i in indices], importances[indices],
                     color=fi_colors, height=0.32, edgecolor='none')
    ax2.set_xlabel("Importance Score", fontsize=8, color='#3d4560')
    for bar in bars2:
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.3f}', va='center', fontsize=8, color='#6b7494')
    ax2.spines[['top','right','left','bottom']].set_visible(False)
    ax2.tick_params(colors='#6b7494', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.xaxis.label.set_color('#3d4560')
    fig2.tight_layout(pad=0.8)
    st.pyplot(fig2)

    col_l, col_r = st.columns(2)
    col_l.caption("🔴 Red — Exercise factors (controllable during workout)")
    col_r.caption("🔵 Blue — Personal factors (demographic / physical)")
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — Workout Planner (AI-powered via Claude)
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">Your Profile</div>', unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p_gender = st.selectbox("Gender", ["Male", "Female"], key="p_gender")
        p_age    = st.number_input("Age (years)", min_value=10, max_value=100, value=25, step=1, key="p_age")
        p_weight = st.number_input("Current Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5, key="p_weight")
    with col_p2:
        p_height  = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5, key="p_height")
        p_fitness = st.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"], key="p_fitness")
        p_days    = st.selectbox("Days Available per Week", ["3 days", "4 days", "5 days", "6 days"], key="p_days")

    st.markdown('<div class="section-label">Goal Settings</div>', unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        p_goal = st.selectbox("Primary Goal", [
            "🔥 Weight Loss",
            "💪 Muscle Gain / Weight Gain",
            "⚖️ Maintenance / Stay Fit",
            "🏃 Improve Endurance & Stamina"
        ], key="p_goal")
    with col_g2:
        p_target = st.number_input("Target Weight (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.5, key="p_target")

    p_notes = st.text_input("Any injuries or preferences? (optional)",
                             placeholder="e.g. bad knees, prefer home workouts, no equipment...", key="p_notes")

    # Profile summary
    p_bmi       = p_weight / ((p_height / 100) ** 2)
    bmi_cat     = "Underweight" if p_bmi < 18.5 else "Normal" if p_bmi < 25 else "Overweight" if p_bmi < 30 else "Obese"
    bmi_col     = "green" if p_bmi < 25 else "amber" if p_bmi < 30 else "red"
    weight_diff = p_weight - p_target
    diff_label  = f"{'Lose' if weight_diff > 0 else 'Gain'} {abs(weight_diff):.1f} kg"
    diff_col    = "green" if abs(weight_diff) <= 5 else "amber" if abs(weight_diff) <= 15 else "red"

    st.markdown(f"""
    <div class="card" style="margin-top:0.4rem; margin-bottom:0.8rem;">
        <div class="card-title">Profile Summary</div>
        <div class="metric-row">
            <div class="metric-box"><div class="metric-lbl">Current BMI</div><div class="metric-val {bmi_col}">{p_bmi:.1f}</div></div>
            <div class="metric-box"><div class="metric-lbl">Category</div><div class="metric-val {bmi_col}">{bmi_cat}</div></div>
            <div class="metric-box"><div class="metric-lbl">Weight Target</div><div class="metric-val {diff_col}">{diff_label}</div></div>
            <div class="metric-box"><div class="metric-lbl">Training Days</div><div class="metric-val">{p_days[0]}/wk</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🏋️ Generate My Workout Plan", key="gen_plan"):
        prompt = f"""You are a certified personal trainer and nutritionist. Create a detailed, personalized workout and nutrition plan.

User Profile:
- Gender: {p_gender}
- Age: {p_age} years
- Current Weight: {p_weight} kg | Height: {p_height} cm | BMI: {p_bmi:.1f} ({bmi_cat})
- Fitness Level: {p_fitness}
- Primary Goal: {p_goal}
- Target Weight: {p_target} kg (needs to {('lose' if weight_diff > 0 else 'gain')} {abs(weight_diff):.1f} kg)
- Available Days per Week: {p_days}
- Notes/Preferences/Injuries: {p_notes if p_notes else 'None'}

Provide a structured plan with these sections (use the exact emoji headers):

🎯 GOAL ANALYSIS
Analyze their stats. What's realistic? Give a timeframe estimate.

🔥 DAILY CALORIE TARGET
Calories to burn per session. Daily intake recommendation. Simple macro split.

📅 WEEKLY SCHEDULE
Day-by-day plan for all {p_days}. Include workout type, focus area, and duration. Mark rest days.

🏋️ RECOMMENDED EXERCISES
8 specific exercises suited to their goal and level. For each: name, sets/reps/duration, and one line on why it helps.

🥗 NUTRITION TIPS
4 specific food/nutrition tips aligned to their goal.

💡 PRO TIPS
3 practical tips specific to their profile (recovery, progression, mindset).

📈 PROGRESS MILESTONES
Expected results at 2 weeks, 1 month, and 3 months.

Be specific with numbers. Keep it motivating and practical."""

        with st.spinner("✨ Generating your personalized plan with AI..."):
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1800,
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=45
                )
                data = response.json()
                plan_text = data["content"][0]["text"]

                st.markdown(f"""
                <div class="plan-header">
                    <div class="plan-title">Your Personalized Workout Plan</div>
                    <div class="plan-sub">{p_gender} · {p_age} yrs · {p_weight} kg · {p_fitness} · {p_days}</div>
                    <div class="goal-chip">{p_goal}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f'<div class="plan-body">{plan_text}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Could not generate plan. Error: {e}")
