import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartCalories", page_icon="", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #0b0e17 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .block-container { padding-top:1.2rem; padding-bottom:2rem; max-width:1100px; background-color:#0b0e17 !important; }
    section[data-testid="stSidebar"] { display:none; }
    #MainMenu, header, footer { visibility:hidden; }

    .hero {
        background: linear-gradient(135deg,#0d1f2d 0%,#1a3a4a 50%,#0f2535 100%);
        border:0.5px solid #1e3a4a; border-radius:18px;
        padding:1.1rem 1.6rem; margin-bottom:1.2rem;
        display:flex; align-items:center; justify-content:space-between;
    }
    .hero-left h1 { font-size:20px; font-weight:700; color:#fff; margin:0; letter-spacing:-0.3px; }
    .hero-left p  { font-size:12px; color:rgba(255,255,255,0.4); margin:3px 0 0 0; }
    .hero-badge   { background:rgba(255,255,255,0.06); border:0.5px solid rgba(255,255,255,0.1); border-radius:20px; padding:4px 12px; font-size:11px; color:rgba(255,255,255,0.4); }

    .section-label { font-size:10px; font-weight:600; color:#3d4560; text-transform:uppercase; letter-spacing:1.2px; margin:1rem 0 0.5rem 0; padding-bottom:0.4rem; border-bottom:0.5px solid #181d2e; }

    .card { background:#131826; border:0.5px solid #1e2438; border-radius:14px; padding:1rem 1.2rem; margin-bottom:0.8rem; }
    .card-title { font-size:10px; font-weight:600; color:#3d4560; text-transform:uppercase; letter-spacing:1.2px; margin-bottom:0.8rem; }

    .metric-row { display:flex; gap:8px; }
    .metric-box { flex:1; background:#0e1120; border-radius:10px; padding:10px 12px; border:0.5px solid #1a1f30; }
    .metric-lbl { font-size:10px; color:#3d4560; margin-bottom:3px; }
    .metric-val { font-size:18px; font-weight:600; color:#dde1f0; }
    .metric-val.green { color:#34d399; }
    .metric-val.amber { color:#fbbf24; }
    .metric-val.red   { color:#f87171; }

    .result-hero { background:linear-gradient(135deg,#0d1f2d,#162f3e); border:0.5px solid #1e3a4a; border-radius:14px; padding:1.2rem 1.4rem; margin-top:0.8rem; display:flex; align-items:center; justify-content:space-between; }
    .result-label { font-size:11px; color:#4a6070; margin-bottom:2px; }
    .result-value { font-size:38px; font-weight:700; color:#fff; letter-spacing:-1px; }
    .result-tag   { font-size:12px; color:#34d399; margin-top:4px; }

    .summary-row { display:flex; justify-content:space-between; padding:5px 0; border-bottom:0.5px solid #181d2e; font-size:12px; }
    .summary-row:last-child { border-bottom:none; }
    .s-key { color:#3d4560; }
    .s-val { color:#dde1f0; font-weight:500; }

    .plan-header { background:linear-gradient(135deg,#0d1f2d,#162f3e); border:0.5px solid #1e3a4a; border-radius:14px; padding:1rem 1.4rem; margin-bottom:0.8rem; }
    .plan-title  { font-size:16px; font-weight:700; color:#fff; margin:0 0 2px 0; }
    .plan-sub    { font-size:12px; color:rgba(255,255,255,0.35); }
    .goal-chip   { display:inline-block; background:rgba(52,211,153,0.08); border:0.5px solid rgba(52,211,153,0.25); color:#34d399; font-size:11px; padding:3px 10px; border-radius:20px; margin-top:6px; }

    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background-color:#0e1120 !important; color:#dde1f0 !important;
        border-color:#1e2438 !important; border-radius:8px !important; font-size:13px !important;
    }
    label[data-testid="stWidgetLabel"] p { color:#5a6080 !important; font-size:12px !important; font-weight:500 !important; }
    .stTextInput > div > div > input { background-color:#0e1120 !important; color:#dde1f0 !important; border-color:#1e2438 !important; border-radius:8px !important; font-size:13px !important; }
    .stButton > button { background:linear-gradient(135deg,#162f3e,#1e4a5e); color:#e0e8f0; border:0.5px solid #254a5e; border-radius:10px; height:42px; font-size:13px; font-weight:600; width:100%; transition:all 0.15s; }
    .stButton > button:hover { background:linear-gradient(135deg,#1e4a5e,#2a5e72); color:white; }
    .stTabs [data-baseweb="tab-list"] { background:#131826; border-radius:10px; padding:3px; gap:3px; border:0.5px solid #1e2438; }
    .stTabs [data-baseweb="tab"] { border-radius:7px; font-size:12px; font-weight:500; color:#3d4560; padding:6px 14px; background:transparent; }
    .stTabs [aria-selected="true"] { background:#0e1120 !important; color:#dde1f0 !important; font-weight:600; }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
    p, span, div { color:#dde1f0; }
</style>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    exercise = pd.read_csv("exercise.csv")
    calories = pd.read_csv("calories.csv")
    df = pd.merge(exercise, calories, on="User_ID", how="inner")
    df["Gender"] = df["Gender"].map({"male": 0, "female": 1})
    X = df[["Gender","Age","Weight","Height","Duration","Heart_Rate","Body_Temp"]]
    y = df["Calories"]
    scaler = StandardScaler()
    model  = RandomForestRegressor(random_state=42)
    model.fit(scaler.fit_transform(X), y)
    return model, scaler, df

model, scaler, df = train_model()
FEATURE_NAMES = ["Gender","Age","Weight","Height","Duration","Heart Rate","Body Temp"]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-left">
    <h1>SmartCalories</h1>
    <p>ML-powered calorie prediction and personalized workout planning</p>
  </div>
  <div class="hero-badge">Random Forest · Rule-Based Planner</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Manual Prediction", "Bulk Scanner", "Model Insights", "Workout Planner"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Personal Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male","Female"])
        age    = st.number_input("Age (years)", min_value=10, max_value=100, value=25, step=1)
    with c2:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)

    st.markdown('<div class="section-label">Exercise Details</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3: duration   = st.number_input("Duration (min)",   min_value=1,    max_value=300,  value=30,   step=1)
    with c4: heart_rate = st.number_input("Heart Rate (bpm)", min_value=40,   max_value=220,  value=100,  step=1)
    with c5: body_temp  = st.number_input("Body Temp (C)",    min_value=35.0, max_value=43.0, value=37.5, step=0.1)

    bmi = weight / ((height / 100) ** 2)
    if   bmi < 18.5: bmi_label, bmi_color = "Underweight", "amber"
    elif bmi < 25:   bmi_label, bmi_color = "Normal",      "green"
    elif bmi < 30:   bmi_label, bmi_color = "Overweight",  "amber"
    else:            bmi_label, bmi_color = "Obese",        "red"

    st.markdown(
        '<div class="card" style="margin-top:0.4rem;">'
        '<div class="card-title">BMI Calculator</div>'
        '<div class="metric-row">'
        f'<div class="metric-box"><div class="metric-lbl">BMI Score</div><div class="metric-val">{bmi:.1f}</div></div>'
        f'<div class="metric-box"><div class="metric-lbl">Category</div><div class="metric-val {bmi_color}">{bmi_label}</div></div>'
        '<div class="metric-box"><div class="metric-lbl">Ideal Range</div><div class="metric-val">18.5-25</div></div>'
        '</div></div>',
        unsafe_allow_html=True)

    if st.button("Predict Calories Burned", key="predict"):
        enc   = 0 if gender == "Male" else 1
        pred  = model.predict(scaler.transform([[enc, age, weight, height, duration, heart_rate, body_temp]]))[0]
        msg   = ("Light activity — suitable for warm-up sessions." if pred < 100
                 else "Moderate workout — good effort." if pred < 250
                 else "Solid training session — well done." if pred < 400
                 else "High intensity workout — excellent performance.")

        st.markdown(
            '<div class="result-hero">'
            '<div>'
            '<div class="result-label">Estimated calories burned</div>'
            f'<div class="result-value">{pred:.1f} <span style="font-size:15px;font-weight:400;color:#3d5060">kcal</span></div>'
            f'<div class="result-tag">{msg}</div>'
            '</div>'
           # '<div style="font-size:44px;color:#e05030">&#9632;</div>'
            '</div>',
            unsafe_allow_html=True)

        avg_cal  = df["Calories"].mean()
        low_cal  = df["Calories"].quantile(0.25)
        high_cal = df["Calories"].quantile(0.75)
        max_val  = max(pred, high_cal) * 1.15

        labels = ["Low (25th %)", "Average", "High (75th %)", "Your Burn"]
        values = [low_cal, avg_cal, high_cal, pred]
        colors = ["#1e4a7a","#1e6a4e","#7a5e1e","#7a1e1e"]

        st.markdown('<div class="card" style="margin-top:0.8rem;"><div class="card-title">Calorie Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 2.2))
        fig.patch.set_facecolor("#131826"); ax.set_facecolor("#131826")
        bars = ax.barh(labels, values, color=colors, height=0.32, edgecolor="none")
        ax.set_xlabel("Calories (kcal)", fontsize=8, color="#3d4560")
        ax.set_xlim(0, max_val)
        for bar in bars:
            ax.text(bar.get_width()+1.5, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():.1f}", va="center", fontsize=8, color="#6b7494")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.tick_params(colors="#3d4560", labelsize=8)
        ax.xaxis.label.set_color("#3d4560")
        ax.tick_params(axis="y", colors="#6b7494", labelsize=8)
        fig.tight_layout(pad=0.8)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="card"><div class="card-title">Input Summary</div>'
            f'<div class="summary-row"><span class="s-key">Gender</span><span class="s-val">{gender}</span></div>'
            f'<div class="summary-row"><span class="s-key">Age</span><span class="s-val">{age} years</span></div>'
            f'<div class="summary-row"><span class="s-key">Weight</span><span class="s-val">{weight} kg</span></div>'
            f'<div class="summary-row"><span class="s-key">Height</span><span class="s-val">{height} cm</span></div>'
            f'<div class="summary-row"><span class="s-key">Duration</span><span class="s-val">{duration} min</span></div>'
            f'<div class="summary-row"><span class="s-key">Heart Rate</span><span class="s-val">{heart_rate} bpm</span></div>'
            f'<div class="summary-row"><span class="s-key">Body Temperature</span><span class="s-val">{body_temp} C</span></div>'
            f'<div class="summary-row"><span class="s-key">BMI</span><span class="s-val">{bmi:.1f} ({bmi_label})</span></div>'
            '</div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    ca, cb, cc = st.columns(3)
    sample = pd.DataFrame({
        "Gender":["male","female"],"Age":[25,30],"Weight":[70.0,60.0],
        "Height":[175.0,162.0],"Duration":[30,45],"Heart_Rate":[100,110],"Body_Temp":[37.5,37.8]
    })
    with ca:
        st.markdown('<div class="card"><div class="card-title">Download Sample</div>', unsafe_allow_html=True)
        fmt = st.selectbox("Format", ["CSV","XLSX","JSON"], label_visibility="collapsed")
        if fmt == "CSV":
            st.download_button("Download Sample", sample.to_csv(index=False).encode(), "sample.csv", "text/csv", use_container_width=True)
        elif fmt == "XLSX":
            import io; buf = io.BytesIO(); sample.to_excel(buf, index=False)
            st.download_button("Download Sample", buf.getvalue(), "sample.xlsx", use_container_width=True)
        else:
            st.download_button("Download Sample", sample.to_json(orient="records").encode(), "sample.json", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="card"><div class="card-title">Upload File</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drag file here", type=["csv","xlsx","xls","json"], label_visibility="visible")
        st.markdown('</div>', unsafe_allow_html=True)
    with cc:
        st.markdown('<div class="card"><div class="card-title">Download Results</div>', unsafe_allow_html=True)
        if uploaded:
            try:
                fn = uploaded.name.lower()
                if fn.endswith(".csv"):           inp = pd.read_csv(uploaded)
                elif fn.endswith((".xlsx",".xls")): inp = pd.read_excel(uploaded)
                else:                              inp = pd.read_json(uploaded)
                disp = inp.copy()
                inp["Gender"] = inp["Gender"].map({"male":0,"female":1})
                disp["Predicted_Calories"] = model.predict(scaler.transform(inp[["Gender","Age","Weight","Height","Duration","Heart_Rate","Body_Temp"]])).round(1)
                st.download_button("Download Results", disp.to_csv(index=False).encode(), "results.csv", "text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.caption("Upload a file first.")
        st.markdown('</div>', unsafe_allow_html=True)
    if uploaded:
        try:
            st.markdown('<div class="card"><div class="card-title">Results</div>', unsafe_allow_html=True)
            st.dataframe(disp, use_container_width=True)
            st.success(f"Predictions complete for {len(disp)} records.")
            st.markdown('</div>', unsafe_allow_html=True)
        except: pass

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div class="card"><div class="card-title">Dataset Overview</div>'
        '<div class="metric-row">'
        f'<div class="metric-box"><div class="metric-lbl">Total Records</div><div class="metric-val">{len(df):,}</div></div>'
        f'<div class="metric-box"><div class="metric-lbl">Avg Calories</div><div class="metric-val">{df["Calories"].mean():.1f}</div></div>'
        f'<div class="metric-box"><div class="metric-lbl">Max Calories</div><div class="metric-val">{df["Calories"].max():.1f}</div></div>'
        '</div></div>',
        unsafe_allow_html=True)

    importances = model.feature_importances_
    indices     = np.argsort(importances)
    fi_colors   = ["#7a1e1e" if FEATURE_NAMES[i] in ["Duration","Heart Rate","Body Temp"] else "#1e4a7a" for i in indices]

    st.markdown('<div class="card"><div class="card-title">Feature Importance — Random Forest</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(8, 2.4))
    fig2.patch.set_facecolor("#131826"); ax2.set_facecolor("#131826")
    bars2 = ax2.barh([FEATURE_NAMES[i] for i in indices], importances[indices], color=fi_colors, height=0.32, edgecolor="none")
    ax2.set_xlabel("Importance Score", fontsize=8, color="#3d4560")
    for bar in bars2:
        ax2.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                 f"{bar.get_width():.3f}", va="center", fontsize=8, color="#6b7494")
    ax2.spines[["top","right","left","bottom"]].set_visible(False)
    ax2.tick_params(colors="#6b7494", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.xaxis.label.set_color("#3d4560")
    fig2.tight_layout(pad=0.8)
    st.pyplot(fig2)
    cl, cr = st.columns(2)
    cl.caption("Red — Exercise factors (controllable)")
    cr.caption("Blue — Personal / demographic factors")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Workout Planner (rule-based)
# ══════════════════════════════════════════════════════════════════════════════

WORKOUT_DATA = {
    "Weight Loss": {
        "Beginner": {
            "exercises": [
                ("Brisk Walking",       "30-40 min",        "Burns calories, easy on joints, great starting cardio"),
                ("Bodyweight Squats",   "3 x 15 reps",      "Activates large leg muscles for high calorie burn"),
                ("Push-ups (knee)",     "3 x 10 reps",      "Upper body strength with minimal equipment"),
                ("Jumping Jacks",       "3 x 30 sec",       "Elevates heart rate quickly for fat burn"),
                ("Plank Hold",          "3 x 20 sec",       "Core stability, burns more than crunches"),
                ("Glute Bridges",       "3 x 15 reps",      "Activates glutes and lower back safely"),
                ("Step-ups (chair)",    "3 x 12 each leg",  "Functional cardio, tones legs"),
                ("Mountain Climbers",   "3 x 20 sec",       "Full body cardio, great calorie torcher"),
            ],
            "cal_burn": "250-350 kcal/session",
        },
        "Intermediate": {
            "exercises": [
                ("Running / Jogging",   "25-30 min",            "High calorie burn, improves cardiovascular health"),
                ("Burpees",             "4 x 10 reps",          "Full body explosive movement, max calorie burn"),
                ("Jump Squats",         "4 x 12 reps",          "Plyometric power, burns 30% more than regular squats"),
                ("Push-ups",            "4 x 15 reps",          "Compound upper body, elevates metabolism"),
                ("Dumbbell Lunges",     "3 x 12 each leg",      "Unilateral leg work, improves balance"),
                ("Bicycle Crunches",    "3 x 20 reps",          "Best ab exercise for calorie burn"),
                ("HIIT Intervals",      "20 min (30s on/30s off)", "Afterburn effect keeps burning calories post-workout"),
                ("Plank to Push-up",    "3 x 10 reps",          "Dynamic core and chest combination"),
            ],
            "cal_burn": "400-550 kcal/session",
        },
        "Advanced": {
            "exercises": [
                ("Sprint Intervals",    "10 x 100m",        "Maximum fat burn with high intensity"),
                ("Barbell Squats",      "5 x 8 reps",       "Heaviest compound lift, huge metabolic demand"),
                ("Deadlifts",           "4 x 6 reps",       "Full posterior chain, highest calorie burn of any lift"),
                ("Box Jumps",           "4 x 10 reps",      "Explosive plyometric, shocks metabolism"),
                ("Pull-ups",            "4 x 8 reps",       "Upper body compound, burns more than machines"),
                ("Kettlebell Swings",   "4 x 15 reps",      "Cardiovascular strength combo, 20 kcal/min"),
                ("Battle Ropes",        "5 x 30 sec",       "Upper body HIIT, keeps heart rate elevated"),
                ("Thrusters",           "4 x 10 reps",      "Full body squat-to-press, ultimate calorie burner"),
            ],
            "cal_burn": "600-800 kcal/session",
        },
    },
    "Muscle Gain / Weight Gain": {
        "Beginner": {
            "exercises": [
                ("Goblet Squats",           "3 x 12 reps",      "Teaches squat pattern, builds quad and glute mass"),
                ("Dumbbell Bench Press",    "3 x 12 reps",      "Primary chest builder, safe for beginners"),
                ("Dumbbell Rows",           "3 x 12 each",      "Back thickness, essential for upper body balance"),
                ("Shoulder Press",          "3 x 12 reps",      "Builds shoulder width and mass"),
                ("Romanian Deadlift",       "3 x 12 reps",      "Hamstring and glute development"),
                ("Dumbbell Curls",          "3 x 15 reps",      "Bicep isolation for arm mass"),
                ("Tricep Dips (chair)",     "3 x 12 reps",      "Tricep mass, makes arms look bigger"),
                ("Plank",                   "3 x 30 sec",       "Core stability supports all other lifts"),
            ],
            "cal_burn": "200-300 kcal/session",
        },
        "Intermediate": {
            "exercises": [
                ("Barbell Squats",          "4 x 8 reps",       "King of mass builders — full lower body"),
                ("Bench Press",             "4 x 8 reps",       "Primary chest and tricep mass builder"),
                ("Barbell Rows",            "4 x 8 reps",       "Upper back thickness and width"),
                ("Overhead Press",          "4 x 8 reps",       "Shoulder mass and core strength"),
                ("Deadlifts",               "3 x 6 reps",       "Total body mass, highest anabolic stimulus"),
                ("Pull-ups / Lat Pulldown", "4 x 8 reps",       "Back width, V-taper development"),
                ("Dips",                    "3 x 10 reps",      "Chest and tricep compound mass builder"),
                ("Face Pulls",              "3 x 15 reps",      "Rear delt and rotator cuff health"),
            ],
            "cal_burn": "350-500 kcal/session",
        },
        "Advanced": {
            "exercises": [
                ("Heavy Barbell Squats",    "5 x 5 reps",       "Max strength and mass stimulus"),
                ("Incline Bench Press",     "4 x 6 reps",       "Upper chest development for full look"),
                ("Weighted Pull-ups",       "4 x 6 reps",       "Maximum lat width with progressive overload"),
                ("Romanian Deadlifts",      "4 x 8 reps",       "Posterior chain hypertrophy"),
                ("Cable Flyes",             "4 x 12 reps",      "Chest isolation for definition"),
                ("Barbell Curls",           "4 x 10 reps",      "Bicep peak mass builder"),
                ("Close-grip Bench",        "4 x 8 reps",       "Tricep mass for arm size"),
                ("Farmer Carries",          "4 x 40m",          "Full body strength and grip development"),
            ],
            "cal_burn": "450-650 kcal/session",
        },
    },
    "Maintenance / Stay Fit": {
        "Beginner": {
            "exercises": [
                ("Walking / Light Jog",     "30 min",           "Maintains cardiovascular base"),
                ("Bodyweight Circuit",      "3 rounds",         "Full body conditioning"),
                ("Yoga / Stretching",       "20 min",           "Flexibility, injury prevention"),
                ("Bodyweight Squats",       "3 x 15 reps",      "Maintain leg strength"),
                ("Push-ups",               "3 x 10 reps",      "Upper body maintenance"),
                ("Plank",                  "3 x 30 sec",       "Core stability"),
                ("Glute Bridges",          "3 x 15 reps",      "Hip and lower back health"),
                ("Side Lateral Raises",    "3 x 15 reps",      "Shoulder mobility and tone"),
            ],
            "cal_burn": "200-300 kcal/session",
        },
        "Intermediate": {
            "exercises": [
                ("Cycling / Swimming",      "30-40 min",        "Low impact cardio, joint friendly"),
                ("Dumbbell Full Body Circuit","4 rounds",       "Efficient full body maintenance"),
                ("Push-ups",               "4 x 15 reps",      "Upper body strength maintenance"),
                ("Dumbbell Squats",        "4 x 12 reps",      "Leg and glute maintenance"),
                ("Plank Variations",       "3 x 45 sec",       "Core strength and stability"),
                ("Dumbbell Rows",          "3 x 12 reps",      "Back posture and strength"),
                ("Lateral Lunges",         "3 x 12 reps",      "Hip mobility and leg tone"),
                ("Shoulder Press",         "3 x 12 reps",      "Upper body balance"),
            ],
            "cal_burn": "300-400 kcal/session",
        },
        "Advanced": {
            "exercises": [
                ("CrossFit-style WOD",      "20-30 min",        "High intensity full body conditioning"),
                ("Olympic Lifts (light)",   "4 x 5 reps",       "Power and athleticism maintenance"),
                ("Pull-ups",               "4 x 10 reps",      "Upper body strength baseline"),
                ("Barbell Squats (moderate)","4 x 10 reps",    "Strength maintenance with volume"),
                ("Kettlebell Complex",      "4 rounds",         "Cardio-strength hybrid"),
                ("Ring Dips",              "3 x 10 reps",      "Advanced pushing strength"),
                ("Rope Climbs",            "3 x 2 climbs",     "Full body functional strength"),
                ("Handstand Practice",     "10 min",           "Balance and shoulder stability"),
            ],
            "cal_burn": "450-600 kcal/session",
        },
    },
    "Improve Endurance & Stamina": {
        "Beginner": {
            "exercises": [
                ("Brisk Walking",           "40-50 min",        "Builds aerobic base safely"),
                ("Cycling (easy)",          "30 min",           "Low impact, builds leg endurance"),
                ("Bodyweight Circuit",      "3 rounds",         "Muscular endurance foundation"),
                ("Jump Rope (slow)",        "3 x 1 min",        "Coordination and light cardio"),
                ("Step-ups",               "3 x 15 each",      "Functional leg endurance"),
                ("Standing March",         "3 x 2 min",        "Hip flexor and core endurance"),
                ("Wall Sit",               "3 x 30 sec",       "Isometric leg endurance"),
                ("Bear Crawl",             "3 x 20m",          "Full body coordination and stamina"),
            ],
            "cal_burn": "250-350 kcal/session",
        },
        "Intermediate": {
            "exercises": [
                ("5K Run / Jog",            "25-35 min",            "Core endurance builder, tracks progress"),
                ("Tempo Runs",              "20 min at 70% effort", "Raises lactate threshold"),
                ("Cycling Intervals",       "30 min",               "VO2 max improvement"),
                ("Jump Rope",              "3 x 3 min",            "Coordination and cardiovascular endurance"),
                ("Rowing Machine",         "20 min",               "Full body aerobic conditioning"),
                ("Burpees",                "4 x 10 reps",          "Cardiovascular and muscular endurance"),
                ("Box Step-ups",           "3 x 15 each",          "Leg stamina and power"),
                ("Stair Climbs",           "10 min",               "Cardiovascular and leg endurance"),
            ],
            "cal_burn": "400-550 kcal/session",
        },
        "Advanced": {
            "exercises": [
                ("Long Run",                "45-60 min at easy pace",   "Builds aerobic engine"),
                ("Fartlek Training",        "30 min",                   "Speed play for race-level stamina"),
                ("VO2 Max Intervals",       "6 x 800m",                 "Maximum oxygen uptake improvement"),
                ("Threshold Run",           "20 min at 85% effort",     "Raises lactate threshold significantly"),
                ("Triathlon Training",      "Swim/Bike/Run combo",      "Complete endurance conditioning"),
                ("Hill Repeats",           "8 x 100m hill",            "Power endurance and leg drive"),
                ("Plyometric Circuit",     "4 rounds",                 "Explosive endurance"),
                ("Long Cycle Ride",        "60-90 min",                "Aerobic base and fat adaptation"),
            ],
            "cal_burn": "600-900 kcal/session",
        },
    },
}

SCHEDULES = {
    "3 days": {
        "Weight Loss":                   ["Cardio + Full Body","REST","HIIT + Core","REST","Cardio + Strength","REST","REST"],
        "Muscle Gain / Weight Gain":     ["Push (Chest/Shoulders/Triceps)","REST","Pull (Back/Biceps)","REST","Legs + Core","REST","REST"],
        "Maintenance / Stay Fit":        ["Full Body Circuit","REST","Cardio + Core","REST","Full Body Strength","REST","REST"],
        "Improve Endurance & Stamina":   ["Long Cardio","REST","Intervals","REST","Tempo + Core","REST","REST"],
    },
    "4 days": {
        "Weight Loss":                   ["Cardio + Upper Body","HIIT","REST","Cardio + Lower Body","Core + Cardio","REST","REST"],
        "Muscle Gain / Weight Gain":     ["Chest + Triceps","Back + Biceps","REST","Legs + Core","Shoulders + Arms","REST","REST"],
        "Maintenance / Stay Fit":        ["Upper Body","Cardio","Lower Body","REST","Full Body + Core","REST","REST"],
        "Improve Endurance & Stamina":   ["Long Run","Cross-train","REST","Intervals","Tempo Run","REST","REST"],
    },
    "5 days": {
        "Weight Loss":                   ["Cardio","Upper HIIT","Lower Strength","Core + Cardio","Full Body HIIT","REST","REST"],
        "Muscle Gain / Weight Gain":     ["Chest + Triceps","Back + Biceps","Legs","Shoulders + Core","Arms + Weak Points","REST","REST"],
        "Maintenance / Stay Fit":        ["Upper Body","Cardio","Lower Body","Full Body","Core + Flexibility","REST","REST"],
        "Improve Endurance & Stamina":   ["Easy Run","Intervals","Cross-train","Tempo Run","Long Run","REST","REST"],
    },
    "6 days": {
        "Weight Loss":                   ["Cardio","Upper HIIT","Lower HIIT","Core + Cardio","Full Body","Active Recovery","REST"],
        "Muscle Gain / Weight Gain":     ["Chest","Back","Legs","Shoulders","Arms","Full Body / Weak Points","REST"],
        "Maintenance / Stay Fit":        ["Upper Body","Cardio","Lower Body","Full Body","Cardio + Core","Flexibility","REST"],
        "Improve Endurance & Stamina":   ["Easy Run","Intervals","Swim/Bike","Tempo Run","Long Run","Active Recovery","REST"],
    },
}

NUTRITION = {
    "Weight Loss": [
        "Aim for a 300-500 kcal daily deficit — enough to lose 0.5 kg/week without muscle loss",
        "Prioritize protein: 1.6-2g per kg of bodyweight to preserve muscle while losing fat",
        "Fill half your plate with vegetables — high volume, low calorie, keeps you full",
        "Cut liquid calories (soda, juice, alcohol) — easiest way to create a painless deficit",
        "Eat your largest meal post-workout when insulin sensitivity is highest",
    ],
    "Muscle Gain / Weight Gain": [
        "Eat in a 300-500 kcal surplus — enough to build muscle without excessive fat gain",
        "Target 2g protein per kg of bodyweight, spread across 4-5 meals for best absorption",
        "Prioritize complex carbs (rice, oats, sweet potato) around workouts for energy",
        "Do not skip healthy fats — avocado, nuts, eggs support testosterone and muscle growth",
        "Post-workout: fast protein and carbs within 30 minutes (banana + protein shake is ideal)",
    ],
    "Maintenance / Stay Fit": [
        "Eat at maintenance calories — use your weight x 33 as a rough daily kcal estimate",
        "Keep protein moderate at 1.4-1.6g per kg to maintain muscle mass",
        "Focus on whole foods 80% of the time — 20% flexibility keeps it sustainable",
        "Stay hydrated — even 2% dehydration reduces performance and focus significantly",
        "Meal prep 2-3 days ahead to avoid defaulting to unhealthy convenience foods",
    ],
    "Improve Endurance & Stamina": [
        "Carbs are your fuel — 55-65% of calories should come from quality carbohydrates",
        "Fuel long sessions (60+ min) with 30-60g carbs per hour (banana, dates, sports drink)",
        "Post-workout recovery meal within 45 min: 3:1 carb-to-protein ratio",
        "Replenish sodium and electrolytes after sweaty sessions to avoid cramping",
        "Eat iron-rich foods (spinach, red meat, lentils) — iron deficiency kills endurance fast",
    ],
}

MILESTONES = {
    "Weight Loss":               ("Reduced bloating, clothes feel slightly looser",                         "0.5-1 kg lost, improved energy and sleep quality",               "2-4 kg lost, visible body composition change, better stamina"),
    "Muscle Gain / Weight Gain": ("Strength increases on key lifts, better mind-muscle connection",         "1-1.5 kg gained, noticeable pump and muscle fullness",           "3-5 kg gained, visible size difference, significantly stronger"),
    "Maintenance / Stay Fit":    ("Consistent energy levels, better mood and sleep",                        "Maintained weight within 1 kg, improved overall fitness",        "Strong baseline fitness, injury-free, sustainable healthy habits"),
    "Improve Endurance & Stamina":("Same pace feels easier, recovery between sets improves",                "Resting heart rate drops 3-5 bpm, can sustain effort longer",    "VO2 max improvement, pace/distance goals reached, strong cardio base"),
}

with tab4:
    st.markdown('<div class="section-label">Your Profile</div>', unsafe_allow_html=True)
    cp1, cp2 = st.columns(2)
    with cp1:
        p_gender  = st.selectbox("Gender", ["Male","Female"], key="p_gender")
        p_age     = st.number_input("Age (years)", min_value=10, max_value=100, value=25, step=1, key="p_age")
        p_weight  = st.number_input("Current Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5, key="p_weight")
    with cp2:
        p_height  = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5, key="p_height")
        p_fitness = st.selectbox("Fitness Level", ["Beginner","Intermediate","Advanced"], key="p_fitness")
        p_days    = st.selectbox("Days Available per Week", ["3 days","4 days","5 days","6 days"], key="p_days")

    st.markdown('<div class="section-label">Goal Settings</div>', unsafe_allow_html=True)
    cg1, cg2 = st.columns(2)
    with cg1:
        p_goal = st.selectbox("Primary Goal", [
            "Weight Loss","Muscle Gain / Weight Gain",
            "Maintenance / Stay Fit","Improve Endurance & Stamina"
        ], key="p_goal")
    with cg2:
        p_target = st.number_input("Target Weight (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.5, key="p_target")

    p_notes = st.text_input("Any injuries or preferences? (optional)", placeholder="e.g. bad knees, prefer home workouts, no equipment...", key="p_notes")

    p_bmi       = p_weight / ((p_height / 100) ** 2)
    bmi_cat     = "Underweight" if p_bmi < 18.5 else "Normal" if p_bmi < 25 else "Overweight" if p_bmi < 30 else "Obese"
    bmi_col     = "green" if p_bmi < 25 else "amber" if p_bmi < 30 else "red"
    weight_diff = p_weight - p_target
    diff_label  = ("Lose " if weight_diff > 0 else "Gain ") + f"{abs(weight_diff):.1f} kg"
    diff_col    = "green" if abs(weight_diff) <= 5 else "amber" if abs(weight_diff) <= 15 else "red"
    weeks       = abs(weight_diff) / 0.5 if weight_diff != 0 else 0

    st.markdown(
        '<div class="card" style="margin-top:0.4rem;margin-bottom:0.8rem;">'
        '<div class="card-title">Profile Summary</div>'
        '<div class="metric-row">'
        f'<div class="metric-box"><div class="metric-lbl">Current BMI</div><div class="metric-val {bmi_col}">{p_bmi:.1f}</div></div>'
        f'<div class="metric-box"><div class="metric-lbl">Category</div><div class="metric-val {bmi_col}">{bmi_cat}</div></div>'
        f'<div class="metric-box"><div class="metric-lbl">Weight Target</div><div class="metric-val {diff_col}">{diff_label}</div></div>'
        f'<div class="metric-box"><div class="metric-lbl">Training Days</div><div class="metric-val">{p_days[0]}/wk</div></div>'
        '</div></div>',
        unsafe_allow_html=True)

    if st.button("Generate My Workout Plan", key="gen_plan"):
        data      = WORKOUT_DATA[p_goal][p_fitness]
        schedule  = SCHEDULES[p_days][p_goal]
        nutrition = NUTRITION[p_goal]
        m2w, m1m, m3m = MILESTONES[p_goal]
        days_map  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

        # Plan header
        st.markdown(
            '<div class="plan-header">'
            '<div class="plan-title">Your Personalized Workout Plan</div>'
            f'<div class="plan-sub">{p_gender} · {p_age} yrs · {p_weight} kg · {p_fitness} · {p_days}</div>'
            f'<div class="goal-chip">{p_goal}</div>'
            '</div>',
            unsafe_allow_html=True)

        # Pre-compute all values
        timeframe      = f"~{int(weeks)} weeks" if weeks > 0 else "ongoing"
        caloric_intake = int(p_weight * 33)
        if p_goal == "Weight Loss":              caloric_intake -= 400
        elif p_goal == "Muscle Gain / Weight Gain": caloric_intake += 400
        protein_g     = int(p_weight * 1.8)
        achievability = "very achievable" if abs(weight_diff) <= 10 else "ambitious but reachable with consistency"
        fitness_msg   = ("build your base safely before increasing intensity" if p_fitness == "Beginner"
                         else "handle structured progressive overload" if p_fitness == "Intermediate"
                         else "go high volume and high intensity from day one")
        macro_split   = ("40% protein / 35% carbs / 25% fat"    if p_goal == "Weight Loss"
                         else "30% protein / 50% carbs / 20% fat" if p_goal == "Muscle Gain / Weight Gain"
                         else "25% protein / 50% carbs / 25% fat" if p_goal == "Improve Endurance & Stamina"
                         else "30% protein / 45% carbs / 25% fat")
        notes_line    = f" Note: {p_notes} — keep this in mind when selecting exercises." if p_notes else ""
        cal_burn_val  = data["cal_burn"]

        # Goal Analysis
        st.markdown(
            '<div class="card">'
            '<div class="card-title">GOAL ANALYSIS</div>'
            '<div style="font-size:13px;color:#c0c6dc;line-height:1.8;">'
            f'Based on your profile (BMI {p_bmi:.1f} — {bmi_cat}), your goal to '
            f'<b style="color:#dde1f0">{diff_label}</b> is {achievability}. '
            f'At ~0.5 kg/week, expect to reach your target in <b style="color:#34d399">{timeframe}</b>.<br><br>'
            f'Your {p_fitness.lower()} fitness level means you will {fitness_msg}.{notes_line}'
            '</div></div>',
            unsafe_allow_html=True)

        # Calorie Target
        st.markdown(
            '<div class="card">'
            '<div class="card-title">DAILY CALORIE TARGET</div>'
            '<div class="metric-row" style="margin-bottom:0.8rem;">'
            f'<div class="metric-box"><div class="metric-lbl">Daily Intake</div><div class="metric-val">{caloric_intake:,} kcal</div></div>'
            f'<div class="metric-box"><div class="metric-lbl">Burn Per Session</div><div class="metric-val">{cal_burn_val}</div></div>'
            f'<div class="metric-box"><div class="metric-lbl">Protein Target</div><div class="metric-val">{protein_g}g/day</div></div>'
            '</div>'
            f'<div style="font-size:12px;color:#6b7494;">Macro split: {macro_split}</div>'
            '</div>',
            unsafe_allow_html=True)

        # Weekly Schedule
        rows = ""
        for day, workout in zip(days_map, schedule):
            if "REST" in workout:
                badge = '<span style="color:#3d4560;font-size:11px;margin-left:8px;">Rest &amp; Recover</span>'
            else:
                badge = f'<span style="background:rgba(52,211,153,0.1);border:0.5px solid rgba(52,211,153,0.2);color:#34d399;font-size:10px;padding:2px 8px;border-radius:10px;margin-left:8px;">{workout}</span>'
            rows += f'<div style="display:flex;align-items:center;padding:7px 0;border-bottom:0.5px solid #181d2e;"><span style="font-size:11px;color:#3d4560;width:90px;">{day}</span>{badge}</div>'

        st.markdown(
            '<div class="card">'
            '<div class="card-title">WEEKLY SCHEDULE</div>'
            + rows +
            '</div>',
            unsafe_allow_html=True)

        # Exercises
        ex_rows = ""
        for i, (name, sets, why) in enumerate(data["exercises"]):
            ex_rows += (
                '<div style="padding:10px 0;border-bottom:0.5px solid #181d2e;">'
                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">'
                f'<span style="font-size:13px;font-weight:600;color:#dde1f0;">{i+1}. {name}</span>'
                f'<span style="font-size:11px;background:#0e1120;border:0.5px solid #1e2438;padding:2px 10px;border-radius:10px;color:#6b7494;">{sets}</span>'
                '</div>'
                f'<div style="font-size:12px;color:#5a6080;">{why}</div>'
                '</div>'
            )
        st.markdown(
            '<div class="card"><div class="card-title">RECOMMENDED EXERCISES</div>'
            + ex_rows +
            '</div>',
            unsafe_allow_html=True)

        # Nutrition
        nut_rows = ""
        for tip in nutrition:
            nut_rows += f'<div style="padding:6px 0;border-bottom:0.5px solid #181d2e;font-size:12px;color:#c0c6dc;">{tip}</div>'
        st.markdown(
            '<div class="card"><div class="card-title">NUTRITION TIPS</div>'
            + nut_rows +
            '</div>',
            unsafe_allow_html=True)

        # Milestones
        st.markdown(
            '<div class="card"><div class="card-title">PROGRESS MILESTONES</div>'
            '<div style="display:flex;gap:8px;">'
            '<div style="flex:1;background:#0e1120;border-radius:10px;padding:12px;border:0.5px solid #1a1f30;">'
            '<div style="font-size:10px;color:#3d4560;margin-bottom:6px;">2 WEEKS</div>'
            f'<div style="font-size:12px;color:#c0c6dc;line-height:1.6;">{m2w}</div>'
            '</div>'
            '<div style="flex:1;background:#0e1120;border-radius:10px;padding:12px;border:0.5px solid #1a1f30;">'
            '<div style="font-size:10px;color:#3d4560;margin-bottom:6px;">1 MONTH</div>'
            f'<div style="font-size:12px;color:#c0c6dc;line-height:1.6;">{m1m}</div>'
            '</div>'
            '<div style="flex:1;background:#0e1120;border-radius:10px;padding:12px;border:0.5px solid rgba(52,211,153,0.2);">'
            '<div style="font-size:10px;color:#34d399;margin-bottom:6px;">3 MONTHS</div>'
            f'<div style="font-size:12px;color:#c0c6dc;line-height:1.6;">{m3m}</div>'
            '</div>'
            '</div></div>',
            unsafe_allow_html=True)
