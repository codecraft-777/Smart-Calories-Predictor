import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCalories 🔥",
    page_icon="🔥",
    layout="centered"
)

# ── Train model from CSVs ────────────────────────────────────────────────────
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

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🔥 SmartCalories Predictor")
st.markdown("Predict how many **calories** you burn during exercise using a **Random Forest** model.")
st.divider()

st.subheader("📋 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    gender     = st.selectbox("👤 Gender", ["Male", "Female"])
    age        = st.number_input("🎂 Age (years)",               min_value=10,    max_value=100,   value=25,    step=1)
    weight     = st.number_input("⚖️ Weight (kg)",               min_value=30.0,  max_value=200.0, value=70.0,  step=0.5)
    height     = st.number_input("📏 Height (cm)",               min_value=100.0, max_value=250.0, value=170.0, step=0.5)

with col2:
    duration   = st.number_input("⏱️ Exercise Duration (min)",   min_value=1,     max_value=300,   value=30,    step=1)
    heart_rate = st.number_input("❤️ Heart Rate (bpm)",          min_value=40,    max_value=220,   value=100,   step=1)
    body_temp  = st.number_input("🌡️ Body Temperature (°C)",     min_value=35.0,  max_value=43.0,  value=37.5,  step=0.1)

# ── BMI Calculator ───────────────────────────────────────────────────────────
st.divider()
st.subheader("📐 Your BMI")

bmi = weight / ((height / 100) ** 2)

if bmi < 18.5:
    bmi_label = "Underweight 🟡"
elif bmi < 25:
    bmi_label = "Normal ✅"
elif bmi < 30:
    bmi_label = "Overweight 🟠"
else:
    bmi_label = "Obese 🔴"

col3, col4 = st.columns(2)
with col3:
    st.metric("BMI Score", f"{bmi:.1f}")
with col4:
    st.metric("Category", bmi_label)

st.divider()

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("🔥 Predict Calories Burned", use_container_width=True, type="primary"):
    gender_encoded = 0 if gender == "Male" else 1
    input_data = np.array([[gender_encoded, age, weight, height, duration, heart_rate, body_temp]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"### 🏃 Estimated Calories Burned: **{prediction:.1f} kcal**")

    if prediction < 100:
        st.info("💡 Light activity — great for a warm-up!")
    elif prediction < 250:
        st.info("💡 Moderate workout — keep it up!")
    elif prediction < 400:
        st.info("💪 Solid session — nice effort!")
    else:
        st.info("🔥 Intense workout — outstanding performance!")

    st.divider()

    # ── Calorie Comparison Graph ─────────────────────────────────────────────
    st.subheader("📊 How You Compare")

    avg_calories = df['Calories'].mean()
    low_calories = df['Calories'].quantile(0.25)
    high_calories = df['Calories'].quantile(0.75)

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(
        ['Low (25th %)', 'Average', 'High (75th %)', 'Your Burn'],
        [low_calories, avg_calories, high_calories, prediction],
        color=['#4FC3F7', '#81C784', '#FFB74D', '#E57373']
    )
    ax.set_xlabel("Calories (kcal)")
    ax.set_title("Your Calorie Burn vs Dataset")
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.1f}', va='center', fontsize=9)
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')
    st.pyplot(fig)

    st.divider()
    st.markdown("#### 📋 Your Input Summary")
    st.table({
        "Feature": ["Gender", "Age", "Weight (kg)", "Height (cm)", "Duration (min)", "Heart Rate (bpm)", "Body Temp (°C)", "BMI"],
        "Value":   [gender,   age,   weight,         height,         duration,          heart_rate,          body_temp,        f"{bmi:.1f}"]
    })

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("SmartCalories | Random Forest Model | Built with Streamlit 🚀")
