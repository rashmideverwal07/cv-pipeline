import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Real-Time CV Pipeline",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d4470;
        margin: 5px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00ff88;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #aaaaaa;
        margin-top: 5px;
    }
    .feature-card {
        background: #1a1d2e;
        border-radius: 10px;
        padding: 15px 20px;
        border-left: 4px solid #00ff88;
        margin: 8px 0;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 2px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00ff88, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00ff88;
        border-bottom: 2px solid #00ff88;
        padding-bottom: 8px;
        margin: 20px 0 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model info ───────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("gesture_model.pkl"):
        with open("gesture_model.pkl", "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_data():
    if os.path.exists("gesture_data.csv"):
        return pd.read_csv("gesture_data.csv")
    return None

model = load_model()
df    = load_data()

# ════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤟 CV Pipeline")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Model Stats",
        "🖐 Gesture Guide",
        "⚙️ How to Run",
        "👩‍💻 About"
    ])
    st.markdown("---")
    st.markdown("**Built with**")
    st.markdown("🐍 Python 3.11")
    st.markdown("👁️ OpenCV")
    st.markdown("🤖 MediaPipe")
    st.markdown("🌲 Scikit-learn")
    st.markdown("---")
    st.markdown("**GitHub**")
    st.markdown("[Rashmi7905](https://github.com/Rashmi7905)")


# ════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ════════════════════════════════════════════════════════
if page == "🏠 Overview":

    st.markdown('<p class="hero-title">Real-Time Computer Vision Pipeline</p>',
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#aaa; font-size:1.1rem;'>"
                "Sign Language Recognition + Posture Monitoring | Live Webcam | 20+ FPS"
                "</p>", unsafe_allow_html=True)

    st.markdown("---")

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>100%</div>
            <div class='metric-label'>Model Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>27</div>
            <div class='metric-label'>Gestures Detected</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>11K+</div>
            <div class='metric-label'>Training Samples</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class='metric-card'>
            <div class='metric-value'>20+ FPS</div>
            <div class='metric-label'>Real-Time Speed</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Two column layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">🤟 Sign Language Mode</p>',
                    unsafe_allow_html=True)
        features = [
            ("24 ASL Letters", "A–Y (excluding J, Z which require motion)"),
            ("Word Builder", "Hold gesture 1.2s to add letter to word"),
            ("Autocorrect", "pyspellchecker fixes spelling automatically"),
            ("Text to Speech", "pyttsx3 speaks each letter and word aloud"),
            ("Two Hand Support", "Both hands tracked simultaneously"),
            ("Custom Gestures", "Map any sign to custom words like HELLO"),
            ("Subtitle Overlay", "YouTube-style captions on the video feed"),
            ("Confidence Filter", "Only accepts predictions above threshold"),
        ]
        for title, desc in features:
            st.markdown(f"""<div class='feature-card'>
                <strong style='color:#00ff88'>✅ {title}</strong><br>
                <span style='color:#aaa; font-size:0.85rem'>{desc}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">🧍 Posture Monitor Mode</p>',
                    unsafe_allow_html=True)
        features2 = [
            ("33 Body Landmarks", "Full upper body skeleton tracked live"),
            ("Neck Angle Detection", "Alerts when head tilts forward"),
            ("Shoulder Monitoring", "Detects uneven shoulder height"),
            ("Visual Alerts", "Red border flashes on bad posture"),
            ("Live Angle Display", "Shows exact neck angle in degrees"),
            ("Instant Switch", "Press P to switch from sign to posture"),
        ]
        for title, desc in features2:
            st.markdown(f"""<div class='feature-card'>
                <strong style='color:#00aaff'>✅ {title}</strong><br>
                <span style='color:#aaa; font-size:0.85rem'>{desc}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-header">🛠️ Tech Stack</p>',
                    unsafe_allow_html=True)
        tech = {
            "Python 3.11"    : "#3776ab",
            "OpenCV"         : "#5C3EE8",
            "MediaPipe"      : "#FF6B35",
            "Scikit-learn"   : "#F7931E",
            "NumPy"          : "#4DABCF",
            "pyttsx3"        : "#00ff88",
            "pyspellchecker" : "#ff6b6b",
            "Streamlit"      : "#FF4B4B",
        }
        badges = " ".join([
            f"<span class='badge' style='background:{c}22; color:{c}; "
            f"border:1px solid {c}'>{t}</span>"
            for t, c in tech.items()
        ])
        st.markdown(badges, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE: MODEL STATS
# ════════════════════════════════════════════════════════
elif page == "📊 Model Stats":
    st.markdown('<p class="section-header">📊 Model Performance</p>',
                unsafe_allow_html=True)

    if df is not None and model is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples",   f"{len(df):,}")
        col2.metric("Total Gestures",  f"{df['label'].nunique()}")
        col3.metric("Model Accuracy",  "100.00%")

        st.markdown("#### Samples per Gesture")
        counts = df['label'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1a1d2e')
        bars = ax.bar(counts.index, counts.values,
                      color='#00ff88', edgecolor='#0e1117', linewidth=0.5)
        ax.set_xlabel("Gesture", color='white', fontsize=11)
        ax.set_ylabel("Samples", color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 3,
                    str(int(bar.get_height())),
                    ha='center', va='bottom',
                    color='white', fontsize=7)
        st.pyplot(fig)

        st.markdown("#### Per-Gesture Precision & Recall")
        # Hardcode from your actual results
        gestures = ['A','B','C','D','E','F','G','H','I','K',
                    'L','M','N','O','P','Q','R','S','T','U',
                    'V','W','X','Y','fist','open_palm','thumbs_up']
        perf_df = pd.DataFrame({
            'Gesture'  : gestures,
            'Precision': [1.0] * len(gestures),
            'Recall'   : [1.0] * len(gestures),
            'F1-Score' : [1.0] * len(gestures),
        })
        st.dataframe(perf_df.set_index('Gesture'),
                     use_container_width=True)

        st.success("🎯 All 27 gestures achieved perfect 1.00 precision, recall and F1-score!")

    else:
        st.warning("gesture_data.csv or gesture_model.pkl not found in project folder.")


# ════════════════════════════════════════════════════════
#  PAGE: GESTURE GUIDE
# ════════════════════════════════════════════════════════
elif page == "🖐 Gesture Guide":
    st.markdown('<p class="section-header">🖐 ASL Gesture Reference</p>',
                unsafe_allow_html=True)

    st.info("💡 J and Z are excluded as they require motion (dynamic gestures). "
            "All others are static and detected in real-time.")

    gestures_info = {
        "A": "Fist with thumb resting on side of fingers",
        "B": "All 4 fingers straight up, thumb tucked across palm",
        "C": "All fingers curved like holding a cylinder",
        "D": "Index up, middle+ring+pinky touch thumb tip",
        "E": "All fingers bent/hooked, thumb tucked under",
        "F": "Index+thumb touch forming circle, other 3 fingers up",
        "G": "Index points sideways, thumb points sideways parallel",
        "H": "Index+middle point sideways together",
        "I": "Only pinky up, others closed",
        "K": "Index up, middle angled out, thumb between them",
        "L": "Index points up, thumb points out (L shape)",
        "M": "Three fingers folded over thumb",
        "N": "Two fingers folded over thumb",
        "O": "All fingers curve to meet thumb (O shape)",
        "P": "Like K but pointing downward",
        "Q": "Like G but pointing downward",
        "R": "Index+middle fingers crossed/twisted together",
        "S": "Fist with thumb over fingers",
        "T": "Thumb between index and middle finger",
        "U": "Index+middle fingers up together",
        "V": "Index+middle fingers up and spread (peace)",
        "W": "Index+middle+ring fingers up and spread",
        "X": "Index finger hooked/bent",
        "Y": "Thumb+pinky out, others closed (shaka)",
    }

    special_info = {
        "open_palm" : "All fingers spread wide → SPACE (new word)",
        "fist"      : "Tight fist → BACKSPACE (delete letter)",
        "thumbs_up" : "Fist + thumb up → GREAT (custom gesture)",
    }

    cols = st.columns(3)
    for i, (letter, desc) in enumerate(gestures_info.items()):
        with cols[i % 3]:
            st.markdown(f"""<div class='feature-card' style='border-left-color:#00aaff'>
                <strong style='color:#00aaff; font-size:1.3rem'>{letter}</strong>
                <span style='color:#aaa; font-size:0.82rem; margin-left:10px'>{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">🎮 Control Gestures</p>',
                unsafe_allow_html=True)
    for gesture, desc in special_info.items():
        st.markdown(f"""<div class='feature-card' style='border-left-color:#ff6b35'>
            <strong style='color:#ff6b35'>{gesture}</strong>
            <span style='color:#aaa; font-size:0.85rem; margin-left:10px'>{desc}</span>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE: HOW TO RUN
# ════════════════════════════════════════════════════════
elif page == "⚙️ How to Run":
    st.markdown('<p class="section-header">⚙️ Setup & Run Instructions</p>',
                unsafe_allow_html=True)

    st.markdown("#### 1️⃣ Clone the repository")
    st.code("git clone https://github.com/Rashmi7905/cv-pipeline.git\ncd cv-pipeline",
            language="bash")

    st.markdown("#### 2️⃣ Create virtual environment")
    st.code("""python -m venv venv
venv\\Scripts\\activate        # Windows
source venv/bin/activate      # Mac/Linux""", language="bash")

    st.markdown("#### 3️⃣ Install dependencies")
    st.code("pip install -r requirements.txt", language="bash")

    st.markdown("#### 4️⃣ Download MediaPipe models")
    st.code(
        "Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task "
        "-OutFile hand_landmarker.task\n"
        "Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task "
        "-OutFile pose_landmarker.task",
        language="bash"
    )

    st.markdown("#### 5️⃣ Run the app")
    st.code("python main.py", language="bash")

    st.markdown("#### 6️⃣ Controls")
    controls = pd.DataFrame({
        "Key / Gesture" : ["S", "P", "T", "Q",
                           "open_palm", "fist", "thumbs_up"],
        "Action"        : ["Switch to Sign Language mode",
                           "Switch to Posture Monitor mode",
                           "Teach a custom gesture",
                           "Quit",
                           "SPACE — complete word",
                           "BACKSPACE — delete last letter",
                           "GREAT — custom gesture"],
    })
    st.table(controls)

    st.markdown("#### 📦 Project Structure")
    st.code("""cv_project/
├── main.py                  # Main application
├── collect_data.py          # Data collection script
├── train_model.py           # Model training script
├── app.py                   # Streamlit dashboard (this page)
├── gesture_model.pkl        # Trained Random Forest model
├── gesture_data.csv         # Collected landmark dataset
├── hand_landmarker.task     # MediaPipe hand model
├── pose_landmarker.task     # MediaPipe pose model
└── requirements.txt         # Python dependencies""", language="bash")


# ════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ════════════════════════════════════════════════════════
elif page == "👩‍💻 About":
    st.markdown('<p class="section-header">👩‍💻 About This Project</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### What is this?
        A real-time computer vision pipeline built entirely in Python that:
        - Translates **ASL sign language into text** using your webcam
        - Monitors **office worker posture** and alerts on bad posture
        - Runs at **20+ FPS on CPU** — no GPU required

        ### Why it stands out
        - End-to-end pipeline: data collection → training → live inference
        - **11,107 training samples** collected from scratch
        - **100% test accuracy** using Random Forest on MediaPipe landmarks
        - Word building, autocorrect, text-to-speech — fully usable system
        - Dual mode: sign language + posture in one unified app

        ### Who built this?
        **Rashmi** — B.Tech CSE student at IGDTUW, Delhi (2024–2028)
        Interested in Computer Vision, ML, and real-time systems.
        """)

        st.markdown("#### 🔗 Links")
        st.markdown("- 💻 [GitHub](https://github.com/Rashmi7905)")
        st.markdown("- 🎓 IGDTUW, Delhi — B.Tech CSE, Batch 2024–2028")

    with col2:
        st.markdown("#### Pipeline Flow")
        steps = [
            "📷 Webcam Input",
            "🔄 OpenCV Pre-process",
            "🤖 MediaPipe Landmarks",
            "🌲 Random Forest Predict",
            "📝 Word Builder",
            "🔊 Text to Speech",
            "📺 Display Output",
        ]
        for i, step in enumerate(steps):
            st.markdown(f"""<div style='background:#1a1d2e; border-radius:8px;
                padding:8px 14px; margin:4px 0; border-left:3px solid #00ff88;
                color:white; font-size:0.85rem'>
                <span style='color:#00ff88'>{i+1}.</span> {step}
            </div>""", unsafe_allow_html=True)