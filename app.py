import os, io, warnings
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import librosa
import joblib
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IDX2CLASS = {0:"CargoShip", 1:"KaiYuan", 2:"noise", 3:"SpeedBoat", 4:"Uuv"}
CLASS_INFO = {
    "CargoShip": {"emoji":"🚢","color":"#4FC3F7","desc":"Large commercial cargo vessel","img":"assets/Cargoship.jpg"},
    "KaiYuan":   {"emoji":"⚓","color":"#81C784","desc":"KaiYuan class naval vessel",   "img":"assets/kaiyuan.jpg"},
    "noise":     {"emoji":"🌊","color":"#FF8A65","desc":"Ambient underwater noise",      "img":None},
    "SpeedBoat": {"emoji":"🛥️","color":"#CE93D8","desc":"High-speed small watercraft",  "img":"assets/SpeedBoat.jpg"},
    "Uuv":       {"emoji":"🤿","color":"#FFD54F","desc":"Unmanned Underwater Vehicle",  "img":"assets/UUV.jpg"},
}
# Classical ML
ML_MODELS = {
    "🌲 Random Forest": "data/classical_ml/best_rf.pkl",
    "⚡ SVM":           "data/classical_ml/best_svm.pkl",
    "🔍 KNN":           "data/classical_ml/best_knn.pkl",
}
SCALER_PATH = "data/classical_ml/scaler.pkl"
# Deep Learning
DL_MODELS = {
    "🧠 Custom CNN":    "data/models/Custom_CNN.pth",
    "🔬 ResNet-18":     "data/models/ResNet18_Transfer.pth",
}
# Mel-spectrogram params (MUST match FeatureExtraction.ipynb exactly)
TARGET_SR   = 22050
WIN_LENGTH  = int(0.025 * TARGET_SR)   # 551
HOP_LENGTH  = int(0.010 * TARGET_SR)   # 220
N_FFT       = 1024
N_MELS      = 128
FMAX        = 8000
DEVICE      = torch.device("cpu")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG + CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ShipSense AI", page_icon="🚢",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
html,body,[data-testid="stAppViewContainer"]{background:#050d1a;color:#e0eaff;font-family:'Outfit',sans-serif}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#071428,#0a1e3d);border-right:1px solid #1a3a6e}
.hero{background:linear-gradient(135deg,#071e3d,#0a2a5e 50%,#071428);border:1px solid #1a3a6e;
      border-radius:20px;padding:36px 48px;margin-bottom:24px;overflow:hidden;position:relative}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:280px;height:280px;
              background:radial-gradient(circle,rgba(79,195,247,.12),transparent 70%);border-radius:50%}
.hero h1{font-size:2.4rem;font-weight:700;background:linear-gradient(90deg,#4FC3F7,#81C784);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 8px}
.hero p{color:#8aaed4;margin:0}
.card{background:linear-gradient(135deg,#0d2244,#0a1e3d);border:1px solid #1a3a6e;
      border-radius:14px;padding:22px;margin-bottom:16px;transition:border-color .3s}
.card:hover{border-color:#4FC3F7}
.card-title{font-size:.72rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#4FC3F7;margin-bottom:10px}
.metric-box{background:rgba(79,195,247,.08);border:1px solid rgba(79,195,247,.2);border-radius:12px;padding:18px;text-align:center}
.metric-val{font-size:1.8rem;font-weight:700;color:#4FC3F7}
.metric-lbl{font-size:.8rem;color:#8aaed4;margin-top:4px}
.pill{display:inline-block;padding:3px 12px;border-radius:50px;font-size:.8rem;font-weight:500;
      background:rgba(79,195,247,.12);color:#4FC3F7;border:1px solid rgba(79,195,247,.3);margin-bottom:14px}
.bar-outer{background:rgba(255,255,255,.08);border-radius:10px;height:10px;overflow:hidden;margin-top:3px}
.bar-inner{height:100%;border-radius:10px}
.stButton>button{width:100%;background:linear-gradient(90deg,#1565C0,#0288D1);color:white;border:none;
                 border-radius:12px;padding:13px 24px;font-size:1rem;font-weight:600;
                 font-family:'Outfit',sans-serif;transition:all .3s}
.stButton>button:hover{background:linear-gradient(90deg,#0288D1,#4FC3F7);transform:translateY(-2px);
                       box-shadow:0 8px 24px rgba(79,195,247,.3)}
h2,h3{color:#c8deff}
.footer{text-align:center;color:#3a5c8e;font-size:.82rem;padding:20px 0 6px;
        border-top:1px solid #0d2244;margin-top:36px}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL ARCHITECTURES (must match training exactly)
# ─────────────────────────────────────────────────────────────────────────────
class CustomShallowCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2,2))
        self.block2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2,2))
        self.block3 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2,2))
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128,256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256,num_classes)
    def forward(self,x):
        x = self.block3(self.block2(self.block1(x)))
        x = self.global_avg_pool(x).view(x.size(0),-1)
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

def build_resnet18(num_classes=5):
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ─────────────────────────────────────────────────────────────────────────────
#  CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_ml_model(path):
    return joblib.load(path)

@st.cache_resource
def get_scaler():
    return joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

@st.cache_resource
def get_cnn():
    m = CustomShallowCNN(5)
    m.load_state_dict(torch.load(DL_MODELS["🧠 Custom CNN"], map_location=DEVICE))
    m.eval(); return m

@st.cache_resource
def get_resnet():
    m = build_resnet18(5)
    m.load_state_dict(torch.load(DL_MODELS["🔬 ResNet-18"], map_location=DEVICE))
    m.eval(); return m

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def wav_to_melspec(wav_bytes: bytes) -> torch.Tensor:
    """WAV bytes → (1,1,128,128) float tensor (matches training pipeline)."""
    y, _ = librosa.load(io.BytesIO(wav_bytes), sr=TARGET_SR, mono=True)
    S    = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_fft=N_FFT,
               hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
               window='hann', n_mels=N_MELS, fmax=FMAX)
    S_dB = librosa.power_to_db(S, ref=np.max)
    t    = torch.tensor(S_dB).unsqueeze(0).unsqueeze(0).float()
    return F.interpolate(t, size=(128,128), mode='bilinear', align_corners=False)  # (1,1,128,128)

def wav_to_ml_features(wav_bytes: bytes) -> np.ndarray:
    """WAV bytes → (1,282) classical-ML feature vector."""
    y, sr = librosa.load(io.BytesIO(wav_bytes), sr=TARGET_SR, mono=True)
    n_mfcc = 40
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_c = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr    = librosa.feature.zero_crossing_rate(y)
    rms    = librosa.feature.rms(y=y)
    feats  = []
    for arr in [mfcc, delta, delta2, chroma, spec_c, zcr, rms]:
        feats += [arr.mean(axis=1), arr.std(axis=1)]
    return np.concatenate(feats).reshape(1,-1)

# ─────────────────────────────────────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def predict_ml(wav_bytes, model_key):
    X      = wav_to_ml_features(wav_bytes)
    scaler = get_scaler()
    X_sc   = scaler.transform(X) if scaler else X
    model  = get_ml_model(ML_MODELS[model_key])
    idx    = int(model.predict(X_sc)[0])
    if hasattr(model,"predict_proba"):
        proba = model.predict_proba(X_sc)[0]
    else:
        raw = model.decision_function(X_sc)[0]
        exp = np.exp(raw - raw.max()); proba = exp/exp.sum()
    return IDX2CLASS[idx], proba

def predict_dl(wav_bytes, model_key):
    spec = wav_to_melspec(wav_bytes)          # (1,1,128,128)
    if model_key == "🔬 ResNet-18":
        spec = spec.repeat(1,3,1,1)           # (1,3,128,128)
        model = get_resnet()
    else:
        model = get_cnn()
    with torch.no_grad():
        logits = model(spec)
        proba  = torch.softmax(logits, dim=1).squeeze().numpy()
    idx = int(np.argmax(proba))
    return IDX2CLASS[idx], proba

# ─────────────────────────────────────────────────────────────────────────────
#  RESULT DISPLAY helper
# ─────────────────────────────────────────────────────────────────────────────
def show_result(name, proba):
    info = CLASS_INFO[name]
    conf = float(proba[list(IDX2CLASS.values()).index(name)]) * 100

    img_path = info.get("img")
    if img_path and os.path.exists(img_path):
        st.markdown(f'<div style="border-radius:14px;overflow:hidden;border:2px solid {info["color"]}66;margin-bottom:4px">', unsafe_allow_html=True)
        st.image(img_path, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="border-radius:14px;border:2px solid {info['color']}66;padding:36px;text-align:center;
                    margin-bottom:10px;background:linear-gradient(135deg,{info['color']}15,{info['color']}05)">
            <div style="font-size:4.5rem">{info['emoji']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;padding:10px 0 16px">
        <span style="font-size:1.6rem;font-weight:700;color:{info['color']}">{info['emoji']} {name}</span><br>
        <span style="font-size:.85rem;color:#8aaed4">{info['desc']}</span>
    </div>
    <div style="text-align:center;margin-bottom:18px">
        <div style="font-size:2.6rem;font-weight:700;color:{info['color']}">{conf:.1f}%</div>
        <div style="font-size:.8rem;color:#8aaed4;letter-spacing:.1em">CONFIDENCE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**All Class Probabilities**")
    for i, cls in IDX2CLASS.items():
        pct = float(proba[i]) * 100
        ci  = CLASS_INFO[cls]
        st.markdown(f"""
        <div style="margin-bottom:9px">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px">
                <span style="font-size:.84rem">{ci['emoji']} {cls}</span>
                <span style="font-size:.84rem;color:{ci['color']};font-weight:600">{pct:.1f}%</span>
            </div>
            <div class="bar-outer"><div class="bar-inner" style="width:{pct}%;background:{ci['color']}"></div></div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align:center;padding:18px 0 6px;font-size:3rem">🚢</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;font-size:1.2rem;font-weight:700;color:#4FC3F7;margin-bottom:4px">ShipSense AI</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;font-size:.8rem;color:#8aaed4;margin-bottom:18px">Underwater Acoustic Classifier</div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("Nav", ["🏠 Home","🔮 Predict","📊 Model Info","ℹ️ About"], label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="card-title">Model Type</div>', unsafe_allow_html=True)
    model_type = st.radio("Type", ["Classical ML","Deep Learning"], horizontal=True, label_visibility="collapsed")
    st.markdown('<div class="card-title" style="margin-top:10px">Select Model</div>', unsafe_allow_html=True)
    if model_type == "Classical ML":
        model_choice = st.selectbox("Model", list(ML_MODELS.keys()), label_visibility="collapsed")
    else:
        model_choice = st.selectbox("Model", list(DL_MODELS.keys()), label_visibility="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
#  HOME
# ─────────────────────────────────────────────────────────────────────────────
if "Home" in page:
    st.markdown("""
    <div class="hero">
        <div class="pill">🔬 AI-Powered Underwater Acoustics</div>
        <h1>ShipSense AI</h1>
        <p>Classify underwater vessels from acoustic signals using Classical ML or Deep Learning.<br>
        Supports KNN, Random Forest, SVM, Custom CNN, and ResNet-18.</p>
    </div>""", unsafe_allow_html=True)

    for col,(v,l) in zip(st.columns(4),[("5","Vessel Classes"),("5","ML+DL Models"),("282","ML Features"),("128×128","DL Spectrogram")]):
        col.markdown(f'<div class="metric-box"><div class="metric-val">{v}</div><div class="metric-lbl">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>### 🎯 Detectable Vessel Classes")
    cols = st.columns(5)
    for col,(name,info) in zip(cols, CLASS_INFO.items()):
        if info.get("img") and os.path.exists(info["img"]):
            col.image(info["img"], use_container_width=True)
        col.markdown(f'<div style="text-align:center"><div style="font-size:1.4rem">{info["emoji"]}</div><div style="font-weight:600;color:{info["color"]};font-size:.9rem">{name}</div><div style="font-size:.72rem;color:#8aaed4">{info["desc"]}</div></div>', unsafe_allow_html=True)

    st.markdown("### 🔗 Pipeline")
    st.markdown("""<div class="card"><div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:.88rem">
    <span style="background:#0d2244;border:1px solid #4FC3F7;border-radius:8px;padding:7px 14px;color:#4FC3F7">📁 WAV</span>
    <span style="color:#8aaed4">→</span><span style="background:#0d2244;border:1px solid #1a3a6e;border-radius:8px;padding:7px 14px">🔧 Load@22kHz</span>
    <span style="color:#8aaed4">→</span><span style="background:#0d2244;border:1px solid #1a3a6e;border-radius:8px;padding:7px 14px">🔬 Features</span>
    <span style="color:#8aaed4">→</span><span style="background:#0d2244;border:1px solid #81C784;border-radius:8px;padding:7px 14px;color:#81C784">🤖 ML/CNN</span>
    <span style="color:#8aaed4">→</span><span style="background:#0d2244;border:1px solid #FFD54F;border-radius:8px;padding:7px 14px;color:#FFD54F">🎯 Result</span>
    </div></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────────────────────────────────────
elif "Predict" in page:
    st.markdown("## 🔮 Vessel Classification")
    st.markdown(f'<div class="pill">{"🤖 "+model_type} &nbsp;|&nbsp; {model_choice}</div>', unsafe_allow_html=True)

    left, right = st.columns([1,1], gap="large")

    with left:
        st.markdown('<div class="card"><div class="card-title">📤 Upload WAV Segment</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop a .wav segment", type=["wav"],
            help="Any .wav from data/processed/segments/")
        if uploaded:
            raw = uploaded.read()
            st.session_state["wav_bytes"] = raw
            st.session_state["wav_name"]  = uploaded.name
            st.success(f"✅ **{uploaded.name}** ({len(raw)/1024:.1f} KB)")
        elif "wav_name" in st.session_state:
            st.info(f"📎 Using: **{st.session_state['wav_name']}**")
        st.markdown('</div>', unsafe_allow_html=True)
        btn = st.button("🚀 Classify Vessel", use_container_width=True)

    with right:
        st.markdown('<div class="card-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
        if btn:
            wav = st.session_state.get("wav_bytes")
            if wav is None:
                st.warning("⚠️ Upload a WAV file first.")
            else:
                with st.spinner("Analyzing…"):
                    try:
                        if model_type == "Classical ML":
                            name, proba = predict_ml(wav, model_choice)
                        else:
                            name, proba = predict_dl(wav, model_choice)
                        st.session_state["result"] = {"name":name,"proba":proba}
                    except Exception as e:
                        import traceback
                        st.error(f"**Error:** {e}")
                        st.code(traceback.format_exc())

        res = st.session_state.get("result")
        if res:
            show_result(res["name"], res["proba"])
        else:
            st.markdown('<div style="text-align:center;padding:60px 20px;color:#3a5c8e"><div style="font-size:4rem;margin-bottom:14px">📡</div><div>Upload a WAV and click Classify Vessel</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────
elif "Model Info" in page:
    st.markdown("## 📊 Model Information")
    t1,t2 = st.tabs(["🤖 Classical ML","🧠 Deep Learning"])

    with t1:
        st.markdown("### Classical ML — 282 Feature Dims")
        rows=[("MFCC (40)","mean+std","80"),("MFCC Δ (40)","mean+std","80"),("MFCC Δ² (40)","mean+std","80"),
              ("Chroma (12)","mean+std","24"),("Spectral Contrast (7)","mean+std","14"),("ZCR","mean+std","2"),("RMS","mean+std","2")]
        st.dataframe(pd.DataFrame(rows,columns=["Feature","Stats","Dims"]),hide_index=True,use_container_width=True)
        for c,(v,l) in zip(st.columns(3),[("2870","Train Samples"),("3","ML Models"),("282","Feature Dims")]):
            c.markdown(f'<div class="metric-box"><div class="metric-val">{v}</div><div class="metric-lbl">{l}</div></div>',unsafe_allow_html=True)

    with t2:
        st.markdown("### Deep Learning — 128×128 Log-Mel Spectrogram")
        st.markdown("""<div class="card">
        <div class="card-title">Input Pipeline</div>
        WAV → resample 22050 Hz → Mel Spectrogram (n_mels=128, n_fft=1024, win=25ms, hop=10ms, fmax=8kHz)
        → power_to_db → bilinear resize (128×128) → CNN/ResNet
        </div>""",unsafe_allow_html=True)
        cols=st.columns(2)
        for col,(name,desc,sz) in zip(cols,[
            ("Custom CNN","3× Conv2d-BN-ReLU-MaxPool → AdaptiveAvgPool → FC(256) → FC(5)","508 KB"),
            ("ResNet-18","Pretrained ImageNet → modified FC(512→5) → 3-channel input","42.7 MB")]):
            col.markdown(f'<div class="card"><div class="card-title">{name}</div><div style="font-size:.85rem;color:#8aaed4;line-height:1.7">{desc}</div><div style="color:#81C784;font-size:.8rem;margin-top:8px">Size: {sz}</div></div>',unsafe_allow_html=True)

    if os.path.exists("notebooks/Model_Testing_History.csv"):
        st.markdown("### 🧪 Experiment History")
        st.dataframe(pd.read_csv("notebooks/Model_Testing_History.csv"), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif "About" in page:
    st.markdown("## ℹ️ About ShipSense AI")
    st.markdown("""<div class="card"><div class="card-title">Overview</div>
    <p style="color:#8aaed4;line-height:1.9">ShipSense AI identifies underwater vessels from passive sonar recordings.
    Two inference paths are available: <b style="color:#4FC3F7">Classical ML</b> (282-dim MFCC features + KNN/RF/SVM) and
    <b style="color:#81C784">Deep Learning</b> (128×128 log-mel spectrograms + Custom CNN or ResNet-18).</p></div>""",
    unsafe_allow_html=True)

    for num,title,desc in [
        ("1","Load WAV","Audio loaded at 22,050 Hz mono"),
        ("2","Classical ML","MFCC+Δ+Δ²+Chroma+SpectralContrast+ZCR+RMS → 282 features → StandardScaler → ML model"),
        ("3","Deep Learning","Mel spectrogram (128 mels) → log dB → resize 128×128 → CNN/ResNet-18"),
        ("4","Output","Predicted class + full probability distribution for all 5 vessels"),
    ]:
        st.markdown(f'<div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:12px"><div style="min-width:32px;height:32px;background:linear-gradient(135deg,#1565C0,#0288D1);border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.9rem;flex-shrink:0">{num}</div><div><div style="font-weight:600;color:#c8deff">{title}</div><div style="color:#8aaed4;font-size:.87rem">{desc}</div></div></div>',unsafe_allow_html=True)

st.markdown('<div class="footer">🚢 ShipSense AI &nbsp;|&nbsp; Powered by <b>Soulware</b> &nbsp;|&nbsp; 🧠 Logic by <b>Mathmind 2.0</b></div>',unsafe_allow_html=True)