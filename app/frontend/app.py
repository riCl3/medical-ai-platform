import base64

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Medical AI Diagnostic Assistant", layout="wide", page_icon="🩻")

st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────────────────── */
    .stApp { background: #f5f7fa; }
    h1, h2, h3, h4 { color: #1a1a2e !important; }

    /* ── Disclaimer banner ──────────────────────────────────────── */
    .disclaimer {
        background: linear-gradient(135deg, #fff8e1 0%, #fff3cd 100%);
        border-left: 5px solid #f59e0b;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #78350f;
        line-height: 1.5;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,.08);
    }
    .disclaimer strong { color: #92400e; }

    /* ── Section headers ────────────────────────────────────────── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 2px solid #e2e8f0;
    }

    /* ── Prediction card ────────────────────────────────────────── */
    .pred-card {
        border-radius: 10px;
        padding: 22px 26px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,.07);
    }
    .pred-card-normal {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #6ee7b7;
    }
    .pred-card-pneumonia {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fca5a5;
    }
    .pred-label {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 4px 0;
    }
    .pred-label-normal { color: #065f46; }
    .pred-label-pneumonia { color: #991b1b; }
    .pred-sub { font-size: 0.85rem; color: #64748b; margin-top: 4px; }

    /* ── Confidence bar wrapper ─────────────────────────────────── */
    .conf-wrapper {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,.05);
    }
    .conf-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
    }
    .conf-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Report box ─────────────────────────────────────────────── */
    .report-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 22px 26px;
        white-space: pre-wrap;
        line-height: 1.7;
        color: #334155;
        font-size: 0.92rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.05);
    }
    .report-box h1, .report-box h2, .report-box h3 {
        color: #1e293b !important;
        margin-top: 12px;
    }

    /* ── Image containers ───────────────────────────────────────── */
    .img-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,.05);
    }

    /* ── History expanders ──────────────────────────────────────── */
    .stExpander {
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        background: #ffffff !important;
    }

    /* ── Tabs styling ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* ── Divider ────────────────────────────────────────────────── */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 20px 0;
    }

    /* ── Info/success/error overrides for readability ────────────── */
    .stAlert > div { border-radius: 8px; }

    /* ── Caption overrides ──────────────────────────────────────── */
    .stCaption, p.caption { color: #64748b !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────
def api_available() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).ok
    except requests.ConnectionError:
        return False


def predict(image_bytes: bytes, filename: str) -> dict:
    resp = requests.post(
        f"{API_URL}/predict",
        files={"file": (filename, image_bytes)},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_history() -> list:
    resp = requests.get(f"{API_URL}/history", timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_detail(pred_id: int) -> dict:
    resp = requests.get(f"{API_URL}/history/{pred_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def b64_to_image(b64: str):
    return base64.b64decode(b64)


def render_prediction_card(label: str, confidence: float):
    is_normal = label == "NORMAL"
    card_class = "pred-card-normal" if is_normal else "pred-card-pneumonia"
    label_class = "pred-label-normal" if is_normal else "pred-label-pneumonia"
    icon = "✅" if is_normal else "⚠️"
    return f"""
    <div class="pred-card {card_class}">
        <div class="pred-label {label_class}">{icon} {label}</div>
        <div class="pred-sub">Model Prediction</div>
    </div>
    """


def render_confidence(confidence: float):
    if confidence >= 0.8:
        color = "#059669"
    elif confidence >= 0.5:
        color = "#d97706"
    else:
        color = "#dc2626"
    return f"""
    <div class="conf-wrapper">
        <div class="conf-label">Confidence Score</div>
        <div class="conf-value" style="color:{color}">{confidence:.1%}</div>
    </div>
    """


# ── Header ────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; margin-bottom:4px;'>🩻 Medical AI Diagnostic Assistant</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align:center; color:#64748b; margin-top:0; font-size:0.95rem;">'
    "Chest X-ray analysis powered by deep learning and explainable AI</p>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer">'
    "<strong>⚠️ Disclaimer:</strong> This tool is for <em>educational and demonstration "
    "purposes only</em>. It is <strong>not</strong> a substitute for professional medical "
    "diagnosis. Always consult a qualified healthcare provider for medical decisions."
    "</div>",
    unsafe_allow_html=True,
)

if not api_available():
    st.error(
        "🔌 Cannot reach the backend API at `localhost:8000`.\n\n"
        "Start it with: `uvicorn app.api.main:app --reload`"
    )
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────
tab_predict, tab_history = st.tabs(["🔍 **Predict**", "📋 **History**"])

# ── Tab 1: Predict ────────────────────────────────────────────────────
with tab_predict:
    upload_col, _ = st.columns([2, 1])
    with upload_col:
        uploaded = st.file_uploader(
            "Upload a chest X-ray image (JPG, PNG)",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG",
        )

    analyze = st.button("🔬  Analyze Image", type="primary", use_container_width=True)

    if uploaded and analyze:
        image_bytes = uploaded.read()

        with st.spinner("⏳ Running Grad-CAM inference and generating report..."):
            try:
                result = predict(image_bytes, uploaded.name)
            except requests.HTTPError as e:
                st.error(f"API error: {e.response.text}")
                st.stop()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        # ── Images side by side ───────────────────────────────────────
        st.markdown("<div class='section-header'>📸 Image Analysis</div>", unsafe_allow_html=True)
        col_orig, col_heat = st.columns(2)

        with col_orig:
            st.markdown("<div class='img-container'>", unsafe_allow_html=True)
            st.image(image_bytes, caption="Original X-Ray", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_heat:
            st.markdown("<div class='img-container'>", unsafe_allow_html=True)
            heatmap_bytes = b64_to_image(result["gradcam_image"])
            st.image(heatmap_bytes, caption="Grad-CAM Heatmap", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Prediction + Confidence ───────────────────────────────────
        st.markdown("<div class='section-header'>📊 Diagnosis</div>", unsafe_allow_html=True)
        col_pred, col_conf = st.columns([1, 1])

        with col_pred:
            st.markdown(
                render_prediction_card(result["predicted_class"], result["confidence"]),
                unsafe_allow_html=True,
            )

        with col_conf:
            st.markdown(
                render_confidence(result["confidence"]),
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── LLM Report ───────────────────────────────────────────────
        st.markdown("<div class='section-header'>📋 AI-Generated Report</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='report-box'>{result['llm_report']}</div>",
            unsafe_allow_html=True,
        )

        st.caption(f"Prediction ID: `{result['prediction_id']}`")

# ── Tab 2: History ────────────────────────────────────────────────────
with tab_history:
    st.markdown("<div class='section-header'>📁 Past Predictions</div>", unsafe_allow_html=True)

    try:
        history = fetch_history()
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")
        st.stop()

    if not history:
        st.info("📭 No predictions yet. Upload an image in the **Predict** tab to get started.")
    else:
        for item in history:
            pred_class = item["predicted_class"]
            icon = "✅" if pred_class == "NORMAL" else "⚠️"
            with st.expander(
                f"{icon}  **#{item['id']}**  —  {pred_class}  "
                f"({item['confidence']:.2%})  —  {item['image_filename']}"
            ):
                detail = fetch_detail(item["id"])

                c1, c2 = st.columns([1, 1])

                with c1:
                    st.markdown(f"**🕐 Timestamp:** {detail['timestamp']}")
                    st.markdown(f"**📄 File:** `{detail['image_filename']}`")
                    st.markdown("")
                    conf = detail["confidence"]
                    st.progress(conf)
                    st.caption(f"Confidence: {conf:.2%}")

                with c2:
                    if detail.get("gradcam_image"):
                        st.image(
                            b64_to_image(detail["gradcam_image"]),
                            caption="Grad-CAM Heatmap",
                            use_container_width=True,
                        )

                if detail.get("llm_report"):
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("<div class='section-header'>📋 Report</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='report-box'>{detail['llm_report']}</div>",
                        unsafe_allow_html=True,
                    )
