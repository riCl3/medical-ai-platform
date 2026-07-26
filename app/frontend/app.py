import base64
import io
import os
import sys
import tempfile
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.db.database import get_all_predictions, get_prediction_by_id, init_db, save_prediction
from app.llm.report_generator import generate_report
from app.xai.gradcam import run_gradcam

# ── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI — Chest X-Ray Diagnostic Assistant",
    layout="wide",
    page_icon="🩻",
    initial_sidebar_state="collapsed",
)

# ── Inject Inter font ─────────────────────────────────────────────────
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link
    href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&display=swap"
    rel="stylesheet"
/>
""",
    unsafe_allow_html=True,
)

# ── Full Dark Theme CSS ───────────────────────────────────────────────
st.markdown(
    """
<style>
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    .stApp {
        background: #080c18;
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(56, 189, 248, 0.06) 0%, transparent 80%),
            radial-gradient(ellipse 60% 40% at 20% 80%, rgba(168, 85, 247, 0.04) 0%, transparent 70%),
            radial-gradient(ellipse 60% 40% at 80% 90%, rgba(34, 211, 238, 0.04) 0%, transparent 70%);
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    p, li { color: #cbd5e1; }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    .app-header {
        text-align: center;
        padding: 32px 16px 8px 16px;
        position: relative;
    }
    .app-header h1 {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px !important;
        letter-spacing: -0.03em;
    }
    .app-header .subtitle {
        color: #64748b;
        font-size: 0.95rem;
        font-weight: 400;
        margin-top: 0;
    }
    .app-header .accent-line {
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #a78bfa);
        border-radius: 2px;
        margin: 12px auto 0 auto;
    }

    .disclaimer {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.08) 0%, rgba(251, 191, 36, 0.04) 100%);
        border-left: 3px solid #fbbf24;
        padding: 12px 18px;
        border-radius: 10px;
        font-size: 0.82rem;
        color: #d6d3d1;
        line-height: 1.6;
        margin-bottom: 28px;
        backdrop-filter: blur(4px);
        border: 1px solid rgba(251, 191, 36, 0.12);
    }
    .disclaimer strong { color: #fbbf24; font-weight: 600; }

    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .section-header .icon { font-size: 1.1rem; }
    .section-header .label { color: #e2e8f0; }

    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 20px 24px;
        backdrop-filter: blur(8px);
        transition: all 0.25s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    .glass-card:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,255,255,0.10);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transform: translateY(-1px);
    }

    .pred-card {
        border-radius: 12px;
        padding: 28px 20px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .pred-card::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 12px;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .pred-card:hover::before { opacity: 1; }
    .pred-card:hover { transform: translateY(-2px); }

    .pred-card-normal {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.10) 0%, rgba(52, 211, 153, 0.04) 100%);
        border: 1px solid rgba(52, 211, 153, 0.20);
    }
    .pred-card-normal::before {
        background: radial-gradient(ellipse at center, rgba(52, 211, 153, 0.06) 0%, transparent 70%);
    }
    .pred-card-pneumonia {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.10) 0%, rgba(248, 113, 113, 0.04) 100%);
        border: 1px solid rgba(248, 113, 113, 0.20);
    }
    .pred-card-pneumonia::before {
        background: radial-gradient(ellipse at center, rgba(248, 113, 113, 0.06) 0%, transparent 70%);
    }

    .pred-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
        position: relative;
        z-index: 1;
    }
    .pred-label-normal { color: #34d399; }
    .pred-label-pneumonia { color: #f87171; }
    .pred-sub {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }

    .conf-wrapper {
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        transition: all 0.25s ease;
    }
    .conf-wrapper:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,255,255,0.10);
        transform: translateY(-1px);
    }
    .conf-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .conf-value {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
        transition: color 0.3s ease;
    }

    .report-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 24px 28px;
        white-space: pre-wrap;
        line-height: 1.8;
        color: #cbd5e1;
        font-size: 0.9rem;
        transition: all 0.25s ease;
    }
    .report-box:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,255,255,0.10);
    }
    .report-box h1, .report-box h2, .report-box h3 {
        color: #f1f5f9 !important;
        margin-top: 16px;
        margin-bottom: 8px;
    }
    .report-box strong { color: #e2e8f0; }
    .report-box em { color: #94a3b8; }

    .img-container {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 8px;
        transition: all 0.25s ease;
        overflow: hidden;
    }
    .img-container:hover {
        border-color: rgba(255,255,255,0.12);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .img-container img {
        border-radius: 8px;
        transition: transform 0.3s ease;
    }
    .img-container:hover img { transform: scale(1.01); }

    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.08) 20%, rgba(255,255,255,0.08) 80%, transparent 100%);
        margin: 28px 0;
    }

    .pred-id-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.08);
        border: 1px solid rgba(56, 189, 248, 0.15);
        color: #7dd3fc;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
        letter-spacing: 0.3px;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 12px rgba(37, 99, 235, 0.25) !important;
        letter-spacing: 0.2px;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(37, 99, 235, 0.35) !important;
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
    }
    div.stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.02);
        border: 1px dashed rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 8px;
        transition: all 0.25s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(56, 189, 248, 0.30);
        background: rgba(56, 189, 248, 0.03);
    }
    div[data-testid="stFileUploader"] section {
        border: none !important;
        background: transparent !important;
    }
    div[data-testid="stFileUploader"] section button {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    div[data-testid="stFileUploader"] section button:hover {
        background: linear-gradient(135deg, #334155 0%, #475569 100%) !important;
        border-color: rgba(255,255,255,0.18) !important;
    }
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    div[data-testid="stFileUploader"] section span {
        color: #94a3b8 !important;
    }

    div[data-testid="stTabs"] { background: transparent; }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        background: transparent !important;
        color: #64748b !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 10px 20px !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.25s ease !important;
        border-radius: 0 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        color: #cbd5e1 !important;
        background: rgba(255,255,255,0.03) !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        background: transparent !important;
        font-weight: 600 !important;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-bar"] {
        background: transparent !important;
        border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    }
    div[data-testid="stTabs"] [role="tablist"] { gap: 4px; }

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(255,255,255,0.10) !important;
        background: rgba(255,255,255,0.03) !important;
    }
    div[data-testid="stExpander"] summary {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        padding: 12px 16px !important;
        border-radius: 12px !important;
    }
    div[data-testid="stExpander"] div[data-testid="stExpanderContent"] {
        padding: 8px 16px 16px 16px !important;
    }

    div[data-testid="stProgress"] > div {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        height: 8px !important;
    }
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
        border-radius: 10px !important;
        transition: width 0.5s ease !important;
    }

    .stSpinner { text-align: center; }
    .stSpinner > div {
        border-top-color: #38bdf8 !important;
        border-right-color: #38bdf8 !important;
        border-bottom-color: rgba(56, 189, 248, 0.2) !important;
        border-left-color: rgba(56, 189, 248, 0.2) !important;
        border-width: 3px !important;
        width: 36px !important;
        height: 36px !important;
    }
    .stSpinner p {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        margin-top: 8px !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px !important;
        border: none !important;
        padding: 12px 16px !important;
        backdrop-filter: blur(4px);
    }
    div[data-testid="stAlert"][role="status"]:has(svg[data-icon="info"]) {
        background: rgba(56, 189, 248, 0.08) !important;
        border: 1px solid rgba(56, 189, 248, 0.15) !important;
        color: #7dd3fc !important;
    }
    div[data-testid="stAlert"][role="status"]:has(svg[data-icon="check"]) {
        background: rgba(52, 211, 153, 0.08) !important;
        border: 1px solid rgba(52, 211, 153, 0.15) !important;
        color: #34d399 !important;
    }
    div[data-testid="stAlert"][role="status"]:has(svg[data-icon="alert"]) {
        background: rgba(251, 191, 36, 0.08) !important;
        border: 1px solid rgba(251, 191, 36, 0.15) !important;
        color: #fbbf24 !important;
    }
    div[data-testid="stAlert"][role="alert"] {
        background: rgba(248, 113, 113, 0.08) !important;
        border: 1px solid rgba(248, 113, 113, 0.15) !important;
        color: #f87171 !important;
    }

    .stCaption, p.caption { color: #475569 !important; font-size: 0.78rem !important; }
    a { color: #38bdf8; transition: color 0.2s ease; }
    a:hover { color: #7dd3fc; }

    .empty-state { text-align: center; padding: 48px 24px; color: #475569; }
    .empty-state .empty-icon { font-size: 3rem; margin-bottom: 12px; }
    .empty-state .empty-text { font-size: 0.95rem; color: #64748b; }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in-up { animation: fadeInUp 0.5s ease forwards; }

    @media (max-width: 768px) {
        .app-header h1 { font-size: 1.6rem !important; }
        .pred-label { font-size: 1.4rem; }
        .conf-value { font-size: 1.8rem; }
        .report-box { padding: 16px 18px; }
    }

    .block-container { padding-top: 1.5rem !important; }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 12px 16px;
        transition: all 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        background: rgba(255,255,255,0.04);
        border-color: rgba(255,255,255,0.10);
    }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Init DB ───────────────────────────────────────────────────────────
init_db()

# ── Helpers ───────────────────────────────────────────────────────────


def b64_to_image(b64: str):
    return base64.b64decode(b64)


def render_prediction_card(label: str, confidence: float):
    is_normal = label == "NORMAL"
    card_class = "pred-card-normal" if is_normal else "pred-card-pneumonia"
    label_class = "pred-label-normal" if is_normal else "pred-label-pneumonia"
    icon = "✅" if is_normal else "⚠️"
    return f"""\
<div class="pred-card {card_class} fade-in-up">
    <div class="pred-label {label_class}">{icon} {label}</div>
    <div class="pred-sub">Model Prediction</div>
</div>"""


def render_confidence(confidence: float):
    if confidence >= 0.8:
        color = "#34d399"
    elif confidence >= 0.5:
        color = "#fbbf24"
    else:
        color = "#f87171"
    return f"""\
<div class="conf-wrapper fade-in-up">
    <div class="conf-label">Confidence Score</div>
    <div class="conf-value" style="color:{color}">{confidence:.1%}</div>
</div>"""


# ── Header ────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="app-header">
    <h1>🩻 MediScan AI</h1>
    <p class="subtitle">Chest X-Ray Analysis &middot; Deep Learning &middot; Explainable AI</p>
    <div class="accent-line"></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="disclaimer">
    <strong>⚠️ Disclaimer:</strong> This tool is for <em>educational and demonstration
    purposes only</em>. It is <strong>not</strong> a substitute for professional medical
    diagnosis. Always consult a qualified healthcare provider for medical decisions.
</div>
""",
    unsafe_allow_html=True,
)

if not os.environ.get("GROQ_API_KEY"):
    st.warning("GROQ_API_KEY is not configured. LLM reports will be skipped.", icon="🔑")

# ── Tabs ──────────────────────────────────────────────────────────────
tab_predict, tab_history = st.tabs(["🔬 **Predict**", "📋 **History**"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: PREDICT
# ═══════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown(
        '<div class="section-header"><span class="icon">📤</span><span class="label">Upload Chest X-Ray</span></div>',
        unsafe_allow_html=True,
    )

    upload_col, hint_col = st.columns([2.5, 1.5])
    with upload_col:
        uploaded = st.file_uploader(
            "Upload a chest X-ray image (JPG, PNG)",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG",
            label_visibility="collapsed",
        )

    with hint_col:
        if not uploaded:
            st.markdown(
                """
<div class="glass-card" style="padding: 12px 16px; font-size: 0.8rem;">
    <strong style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Supported</strong><br>
    <span style="color: #64748b;">PNG, JPG, JPEG</span><br><br>
    <strong style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;">Analysis Includes</strong><br>
    <span style="color: #64748b;">Grad-CAM heatmap &amp; AI report</span>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

    analyze = st.button("🔬  Analyze Image", type="primary", use_container_width=True)

    if uploaded and analyze:
        image_bytes = uploaded.read()

        with st.spinner("⏳ Running Grad-CAM inference and generating report..."):
            try:
                tmp = Path(tempfile.mkdtemp()) / f"{uuid.uuid4().hex}{Path(uploaded.name).suffix}"
                tmp.write_bytes(image_bytes)
                pred_label, confidence, gradcam_pil = run_gradcam(str(tmp))
                tmp.unlink(missing_ok=True)
            except Exception as e:
                st.error(f"Grad-CAM inference failed: {e}")
                st.stop()

            try:
                llm_report = generate_report(pred_label, confidence)
            except Exception as e:
                llm_report = f"(Report generation failed: {e})"

            buf = io.BytesIO()
            gradcam_pil.save(buf, format="PNG")
            gradcam_b64 = base64.b64encode(buf.getvalue()).decode()

            pred_id = save_prediction(
                image_filename=uploaded.name,
                predicted_class=pred_label,
                confidence=confidence,
                llm_report=llm_report,
                gradcam_blob=gradcam_b64,
            )

        # Images side by side
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header"><span class="icon">📸</span><span class="label">Image Analysis</span></div>',
            unsafe_allow_html=True,
        )

        col_orig, col_heat = st.columns(2)

        with col_orig:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(image_bytes, caption="Original X-Ray", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_heat:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(gradcam_pil, caption="Grad-CAM Heatmap", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # Prediction + Confidence
        st.markdown(
            '<div class="section-header"><span class="icon">📊</span><span class="label">Diagnosis Results</span></div>',
            unsafe_allow_html=True,
        )
        col_pred, col_conf = st.columns([1, 1])

        with col_pred:
            st.markdown(
                render_prediction_card(pred_label, confidence),
                unsafe_allow_html=True,
            )

        with col_conf:
            st.markdown(
                render_confidence(confidence),
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # LLM Report
        st.markdown(
            '<div class="section-header"><span class="icon">📋</span><span class="label">AI-Generated Clinical Report</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="report-box fade-in-up">{llm_report}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="text-align: right; margin-top: 16px;">'
            f'<span class="pred-id-badge">Prediction #{pred_id}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════
# TAB 2: HISTORY
# ═══════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown(
        '<div class="section-header"><span class="icon">📁</span><span class="label">Past Predictions</span></div>',
        unsafe_allow_html=True,
    )

    try:
        preds = get_all_predictions()
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")
        st.stop()

    if not preds:
        st.markdown(
            """
<div class="empty-state">
    <div class="empty-icon">📭</div>
    <div class="empty-text">No predictions yet. Upload an image in the <strong>Predict</strong> tab to get started.</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        for p in preds:
            pred_class = p.predicted_class
            is_normal = pred_class == "NORMAL"
            icon = "✅" if is_normal else "⚠️"

            expander_label = (
                f"{icon} **#{p.id}**"
                f" · {pred_class}"
                f" · ({p.confidence:.1%})"
                f" · `{p.image_filename}`"
            )

            with st.expander(expander_label, expanded=False):
                c1, c2 = st.columns([1.2, 1])

                with c1:
                    st.markdown(
                        f'<div style="margin-bottom: 12px;">'
                        f'<span style="color:#64748b; font-size:0.75rem; text-transform:uppercase; '
                        f'letter-spacing:1px; font-weight:600;">Timestamp</span><br>'
                        f'<span style="color:#e2e8f0;">🕐 {p.timestamp}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div style="margin-bottom: 16px;">'
                        f'<span style="color:#64748b; font-size:0.75rem; text-transform:uppercase; '
                        f'letter-spacing:1px; font-weight:600;">Filename</span><br>'
                        f'<span style="color:#e2e8f0;">📄 <code style="color:#7dd3fc; background:rgba(56,189,248,0.06); '
                        f'padding:1px 6px; border-radius:4px;">{p.image_filename}</code></span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        f'<div style="margin-bottom: 4px;">'
                        f'<span style="color:#64748b; font-size:0.75rem; text-transform:uppercase; '
                        f'letter-spacing:1px; font-weight:600;">Confidence</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(p.confidence)
                    st.markdown(
                        f'<div style="text-align:right; margin-top: 2px;">'
                        f'<span style="color:#94a3b8; font-size:0.85rem; font-weight:600;">{p.confidence:.1%}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with c2:
                    if p.gradcam_blob:
                        st.markdown('<div class="img-container">', unsafe_allow_html=True)
                        st.image(
                            b64_to_image(p.gradcam_blob),
                            caption="Grad-CAM Heatmap",
                            use_container_width=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                if p.llm_report:
                    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="section-header" style="margin-bottom:12px;">'
                        '<span class="icon">📋</span><span class="label">Report</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="report-box">{p.llm_report}</div>',
                        unsafe_allow_html=True,
                    )
