import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCalories",
    page_icon="🔥",
    layout="centered"
)

# ── Custom CSS (FraudGuard style) ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .block-container { padding-top: 2rem; }

    h1 {
        color: #2e7d32;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .section-box {
        border-left: 4px solid #2e7d32;
        padding: 0.4rem 1rem;
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1rem;
        color: #2e7d32;
    }
    .stButton > button {
        background-color: #1a1a2e;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        width: 100%;
        border: none;
    }
    .stButton > button:hover {
        background-color: #2e7d32;
        color: white;
    }
    .result-box {
        background-color: #f1f8e9;
        border: 1px solid #a5d6a7;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        color: #1b5e20;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 500;
        color: #444;
    }
    .stTabs [aria-selected="true"] {
        color: #2e7d32 !important;
        border-bottom: 2px solid #2e7d32 !important;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Train model ──────────────────────────────────────────────────────────────
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

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("<h1>SmartCalories</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Calorie Burn Prediction — Powered by Random Forest Machine Learning</p>', unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Manual Prediction", "Bulk Scanner", "Model Insights"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Manual Prediction
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-box">Enter Your Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        gender     = st.selectbox("Gender", ["Male", "Female"])
        age        = st.number_input("Age (years)",            min_value=10,    max_value=100,   value=25,    step=1)
        weight     = st.number_input("Weight (kg)",            min_value=30.0,  max_value=200.0, value=70.0,  step=0.5)
        height     = st.number_input("Height (cm)",            min_value=100.0, max_value=250.0, value=170.0, step=0.5)
    with col2:
        duration   = st.number_input("Exercise Duration (min)",min_value=1,     max_value=300,   value=30,    step=1)
        heart_rate = st.number_input("Heart Rate (bpm)",       min_value=40,    max_value=220,   value=100,   step=1)
        body_temp  = st.number_input("Body Temperature (C)",   min_value=35.0,  max_value=43.0,  value=37.5,  step=0.1)

    # BMI
    st.markdown('<div class="section-box">BMI Calculator</div>', unsafe_allow_html=True)
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:   bmi_label = "Underweight"
    elif bmi < 25:   bmi_label = "Normal"
    elif bmi < 30:   bmi_label = "Overweight"
    else:            bmi_label = "Obese"

    c1, c2 = st.columns(2)
    c1.metric("BMI Score", f"{bmi:.1f}")
    c2.metric("Category", bmi_label)

    st.markdown("")
    if st.button("Predict Calories Burned", key="predict_manual"):
        gender_encoded = 0 if gender == "Male" else 1
        input_data = np.array([[gender_encoded, age, weight, height, duration, heart_rate, body_temp]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.markdown(f'<div class="result-box">Estimated Calories Burned: {prediction:.1f} kcal</div>', unsafe_allow_html=True)

        if prediction < 100:     st.info("Light activity — suitable for warm-up sessions.")
        elif prediction < 250:   st.info("Moderate workout — good effort.")
        elif prediction < 400:   st.success("Solid training session — well done.")
        else:                    st.success("High intensity workout — excellent performance.")

        # Comparison chart
        st.markdown('<div class="section-box">Calorie Comparison</div>', unsafe_allow_html=True)
        avg_cal  = df['Calories'].mean()
        low_cal  = df['Calories'].quantile(0.25)
        high_cal = df['Calories'].quantile(0.75)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(['Low (25th %)', 'Average', 'High (75th %)', 'Your Burn'],
                [low_cal, avg_cal, high_cal, prediction],
                color=['#90CAF9', '#A5D6A7', '#FFCC80', '#EF9A9A'])
        ax.set_xlabel("Calories (kcal)")
        ax.set_title("Your Calorie Burn vs Dataset")
        for bar in ax.patches:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.1f}', va='center', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)

        # Summary table
        st.markdown('<div class="section-box">Input Summary</div>', unsafe_allow_html=True)
        st.table(pd.DataFrame({
            "Feature": ["Gender", "Age", "Weight (kg)", "Height (cm)", "Duration (min)", "Heart Rate (bpm)", "Body Temp (C)", "BMI"],
            "Value":   [gender,   age,   weight,         height,         duration,          heart_rate,          body_temp,       f"{bmi:.1f}"]
        }))

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bulk Scanner
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<div class="section-box">Download Sample File</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-box">Upload File to Scan</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    with col_c:
        st.markdown('<div class="section-box">Download Results</div>', unsafe_allow_html=True)
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
            st.info("Upload a file first to download results.")

    if uploaded_file is not None:
        try:
            st.markdown('<div class="section-box">Prediction Results</div>', unsafe_allow_html=True)
            st.dataframe(input_df, use_container_width=True)
            st.success(f"Predictions complete for {len(input_df)} records.")
        except:
            pass

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-box">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    st.markdown("This chart shows which features the model relies on most when predicting calorie burn.")

    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    colors = ['#EF9A9A' if FEATURE_NAMES[i] in ['Duration', 'Heart Rate', 'Body Temp']
              else '#90CAF9' for i in indices]
    bars2 = ax2.barh([FEATURE_NAMES[i] for i in indices], importances[indices], color=colors)
    ax2.set_xlabel("Importance Score")
    ax2.set_title("Feature Importance")
    for bar in bars2:
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.3f}', va='center', fontsize=9)
    fig2.tight_layout()
    st.pyplot(fig2)

    c1, c2 = st.columns(2)
    c1.markdown("**Red — Exercise factors** (controllable during workout)")
    c2.markdown("**Blue — Personal factors** (demographic / physical)")

    st.markdown('<div class="section-box">Dataset Overview</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    c3.metric("Total Records", len(df))
    c4.metric("Avg Calories Burned", f"{df['Calories'].mean():.1f} kcal")
    c5.metric("Max Calories Burned", f"{df['Calories'].max():.1f} kcal")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("SmartCalories  |  Random Forest Model  |  Built with Streamlit")
