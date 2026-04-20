import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCalories",
    page_icon="",
    layout="wide"
)

# Custom CSS 
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #0f1117 !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0f1117 !important;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        background-color: #0f1117 !important;
    }
    section[data-testid="stSidebar"] { display: none; }
    #MainMenu, header, footer { visibility: hidden; }

    .hero {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 16px;
        padding: 2rem 2rem 1.6rem;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 28px;
        font-weight: 600;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin: 0;
    }
    .hero p { font-size: 13px; color: rgba(255,255,255,0.6); margin-top: 6px; }
    .hero-tags { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
    .tag {
        background: rgba(255,255,255,0.08);
        border: 0.5px solid rgba(255,255,255,0.18);
        color: rgba(255,255,255,0.8);
        font-size: 11px;
        padding: 4px 12px;
        border-radius: 20px;
    }

    .card {
        background: #1a1f2e;
        border: 0.5px solid #2a2f3f;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 10px;
        font-weight: 600;
        color: #5a6080;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }

    .metric-row { display: flex; gap: 12px; }
    .metric-box {
        flex: 1;
        background: #12151f;
        border-radius: 10px;
        padding: 12px 14px;
        border: 0.5px solid #2a2f3f;
    }
    .metric-lbl { font-size: 11px; color: #5a6080; margin-bottom: 4px; }
    .metric-val { font-size: 20px; font-weight: 500; color: #e8eaf0; }
    .metric-val.green { color: #4ade80; }
    .metric-val.amber { color: #fbbf24; }
    .metric-val.red   { color: #f87171; }

    .result-hero {
        background: #1a1f2e;
        border: 0.5px solid #2a2f3f;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .result-label { font-size: 12px; color: #5a6080; margin-bottom: 4px; }
    .result-value { font-size: 32px; font-weight: 600; color: #e8eaf0; }
    .result-tag   { font-size: 13px; color: #4ade80; margin-top: 4px; }

    .summary-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 0.5px solid #2a2f3f;
        font-size: 13px;
    }
    .summary-row:last-child { border-bottom: none; }
    .s-key { color: #5a6080; }
    .s-val { color: #e8eaf0; font-weight: 500; }

    /* Streamlit input overrides for dark */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: #12151f !important;
        color: #e8eaf0 !important;
        border-color: #2a2f3f !important;
        border-radius: 8px !important;
    }
    label[data-testid="stWidgetLabel"] p { color: #9aa0b8 !important; font-size: 13px !important; }

    .stButton > button {
        background: linear-gradient(135deg, #203a43, #2c5364);
        color: white;
        border: 0.5px solid #3a5060;
        border-radius: 10px;
        height: 44px;
        font-size: 14px;
        font-weight: 500;
        width: 100%;
        letter-spacing: 0.2px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2c5364, #3a6070);
        border-color: #4a7080;
        color: white;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        border: 0.5px solid #2a2f3f;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px;
        font-size: 13px;
        color: #5a6080;
        padding: 6px 16px;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #12151f !important;
        color: #e8eaf0 !important;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"]    { display: none; }

    .stDataFrame { background: #1a1f2e !important; border-radius: 10px; }
    .stAlert { border-radius: 10px; }
    p, span, div { color: #e8eaf0; }
</style>
""", unsafe_allow_html=True)

# Train model 
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

# Hero 
st.markdown("""
<div class="hero">
    <h1>SmartCalories</h1>
    <p>Enter your body stats and exercise details to get an instant calorie burn estimate.</p>
    
</div>
""", unsafe_allow_html=True)

# Tabs 
tab1, tab2, tab3 = st.tabs(["Manual Prediction", "Bulk Scanner", "Model Insights"])

# TAB 1 — Manual Prediction
with tab1:
    st.markdown('<div class="card"><div class="card-title">Personal Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age    = st.number_input("Age (years)",  min_value=10,    max_value=100,   value=25,    step=1)
    with col2:
        weight = st.number_input("Weight (kg)",  min_value=30.0,  max_value=200.0, value=70.0,  step=0.5)
        height = st.number_input("Height (cm)",  min_value=100.0, max_value=250.0, value=170.0, step=0.5)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">Exercise Details</div>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        duration   = st.number_input("Duration (min)",   min_value=1,    max_value=300,  value=30,   step=1)
    with col4:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40,   max_value=220,  value=100,  step=1)
    with col5:
        body_temp  = st.number_input("Body Temp (°C)",   min_value=35.0, max_value=43.0, value=37.5, step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    # BMI
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:   bmi_label, bmi_color = "Underweight", "amber"
    elif bmi < 25:   bmi_label, bmi_color = "Normal",      "green"
    elif bmi < 30:   bmi_label, bmi_color = "Overweight",  "amber"
    else:            bmi_label, bmi_color = "Obese",        "red"

    st.markdown(f"""
    <div class="card">
        <div class="card-title">BMI Calculator</div>
        <div class="metric-row">
            <div class="metric-box"><div class="metric-lbl">BMI score</div><div class="metric-val">{bmi:.1f}</div></div>
            <div class="metric-box"><div class="metric-lbl">Category</div><div class="metric-val {bmi_color}">{bmi_label}</div></div>
            <div class="metric-box"><div class="metric-lbl">Ideal range</div><div class="metric-val">18.5–25</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
                <div class="result-value">{prediction:.1f} kcal</div>
                <div class="result-tag">{msg}</div>
            </div>
            <div style="font-size:36px;">🔥</div>
        </div>
        """, unsafe_allow_html=True)

        # Comparison chart — dark
        avg_cal  = df['Calories'].mean()
        low_cal  = df['Calories'].quantile(0.25)
        high_cal = df['Calories'].quantile(0.75)
        max_val  = max(prediction, high_cal) * 1.15

        st.markdown('<div class="card" style="margin-top:1rem;"><div class="card-title">Calorie Comparison</div>', unsafe_allow_html=True)
        labels = ['Low (25th %)', 'Average', 'High (75th %)', 'Your Burn']
        values = [low_cal, avg_cal, high_cal, prediction]
        colors = ['#2d5a8e', '#2d7a5e', '#8e6e2d', '#8e2d2d']

        fig, ax = plt.subplots(figsize=(7, 2.8))
        fig.patch.set_facecolor('#1a1f2e')
        ax.set_facecolor('#1a1f2e')
        bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor='none')
        ax.set_xlabel("Calories (kcal)", fontsize=11, color='#5a6080')
        ax.set_xlim(0, max_val)
        for bar in bars:
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.1f}', va='center', fontsize=10, color='#9aa0b8')
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.tick_params(colors='#5a6080', labelsize=10)
        ax.xaxis.label.set_color('#5a6080')
        ax.tick_params(axis='y', colors='#9aa0b8')
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Summary
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

# TAB 2 — Bulk Scanner
with tab2:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="card"><div class="card-title">Download Sample File</div>', unsafe_allow_html=True)
        sample = pd.DataFrame({
            'Gender':     ['male', 'female'],
            'Age':        [25, 30],
            'Weight':     [70.0, 60.0],
            'Height':     [175.0, 162.0],
            'Duration':   [30, 45],
            'Heart_Rate': [100, 110],
            'Body_Temp':  [37.5, 37.8]
        })
        fmt = st.selectbox("Format", ["CSV", "XLSX", "JSON"], label_visibility="collapsed")
        if fmt == "CSV":
            st.download_button("Download Sample", sample.to_csv(index=False).encode(), "sample_input.csv", "text/csv", use_container_width=True)
        elif fmt == "XLSX":
            import io
            buf = io.BytesIO()
            sample.to_excel(buf, index=False)
            st.download_button("Download Sample", buf.getvalue(), "sample_input.xlsx", use_container_width=True)
        else:
            st.download_button("Download Sample", sample.to_json(orient="records").encode(), "sample_input.json", "application/json", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card"><div class="card-title">Upload File to Scan</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type=["csv", "xlsx", "xls", "json"],
            label_visibility="visible",
            help="Supported: CSV, XLSX, JSON"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="card"><div class="card-title">Download Results</div>', unsafe_allow_html=True)
        if uploaded_file is not None:
            try:
                fname = uploaded_file.name.lower()
                if fname.endswith(".csv"):
                    input_df = pd.read_csv(uploaded_file)
                elif fname.endswith((".xlsx", ".xls")):
                    input_df = pd.read_excel(uploaded_file)
                elif fname.endswith(".json"):
                    input_df = pd.read_json(uploaded_file)
                else:
                    input_df = pd.read_csv(uploaded_file)

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
            st.success(f"Predictions complete for {len(display_df)} records.")
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            pass

# TAB 3 — Model Insights
with tab3:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Dataset Overview</div>
        <div class="metric-row">
            <div class="metric-box"><div class="metric-lbl">Total records</div><div class="metric-val">{len(df):,}</div></div>
            <div class="metric-box"><div class="metric-lbl">Avg calories</div><div class="metric-val">{df['Calories'].mean():.1f}</div></div>
            <div class="metric-box"><div class="metric-lbl">Max calories</div><div class="metric-val">{df['Calories'].max():.1f}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    importances = model.feature_importances_
    indices = np.argsort(importances)
    fi_colors = ['#8e2d2d' if FEATURE_NAMES[i] in ['Duration', 'Heart Rate', 'Body Temp']
                 else '#2d5a8e' for i in indices]

    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    fig2.patch.set_facecolor('#1a1f2e')
    ax2.set_facecolor('#1a1f2e')
    bars2 = ax2.barh([FEATURE_NAMES[i] for i in indices], importances[indices],
                     color=fi_colors, height=0.5, edgecolor='none')
    ax2.set_xlabel("Importance Score", fontsize=11, color='#5a6080')
    for bar in bars2:
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.3f}', va='center', fontsize=9, color='#9aa0b8')
    ax2.spines[['top','right','left','bottom']].set_visible(False)
    ax2.tick_params(colors='#9aa0b8', labelsize=10)
    ax2.xaxis.label.set_color('#5a6080')
    fig2.tight_layout()
    st.pyplot(fig2)

    col_l, col_r = st.columns(2)
    col_l.caption("Red — Exercise factors (controllable during workout)")
    col_r.caption("Blue — Personal factors (demographic / physical)")
    st.markdown('</div>', unsafe_allow_html=True)
