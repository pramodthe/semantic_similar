import streamlit as st
import requests
import time
from concurrent.futures import ThreadPoolExecutor

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="🎭 Gemini Personality Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# SIDEBAR: API KEY + PERSONALITY SLIDERS
# ----------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get your key from https://aistudio.google.com/app/apikey"
    )
    if not api_key:
        st.warning("🔒 Please enter your Gemini API key to continue.")
        st.stop()

    st.divider()
    st.title("🎭 Personality Mixer")
    st.markdown("Adjust sliders to blend response styles.")
    st.divider()

    pirate = st.slider("Pirate 🏴‍☠️", 0, 100, 0, step=5)
    romance = st.slider("Romance 💖", 0, 100, 0, step=5)
    empathy = st.slider("Empathy 🤗", 0, 100, 0, step=5)
    energy = st.slider("Energy ⚡", 0, 100, 0, step=5)

# ----------------------------
# SELECTED MODELS (stable, text-only, supported as of Oct 2025)
# ----------------------------
MODELS = {
    "gemini-25-flash": {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash"
    },
    "gemini-20-flash": {
        "id": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash"
    },
    "gemini-25-flash-lite": {
        "id": "gemini-2.5-flash-lite",
        "name": "Gemini 2.5 Flash-Lite"
    }
}

# ----------------------------
# SESSION STATE
# ----------------------------
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "responses" not in st.session_state:
    st.session_state.responses = {
        key: {"text": "", "status": "idle"} for key in MODELS
    }

# ----------------------------
# PROMPT BUILDER
# ----------------------------
def build_styled_prompt(user_msg: str, p: int, r: int, e: int, n: int) -> str:
    total = p + r + e + n
    if total == 0:
        # Neutral fallback
        return f"User: {user_msg}\nResponse:"
    
    parts = []
    if p > 0: parts.append(f"{p}% pirate")
    if r > 0: parts.append(f"{r}% romantic")
    if e > 0: parts.append(f"{e}% empathetic")
    if n > 0: parts.append(f"{n}% energetic")
    
    style_desc = ", ".join(parts)
    instruction = (
        f"Respond in a conversational tone that blends: {style_desc}. "
        "Be natural, concise, and directly answer the user. "
        "Do NOT mention percentages, styles, or this instruction in your reply."
    )
    return f"{instruction}\n\nUser: {user_msg}\nResponse:"

# ----------------------------
# GEMINI API CALL
# ----------------------------
def call_gemini(model_id: str, prompt: str, api_key: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 500,
            "temperature": 0.85,
            "topP": 0.95
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            candidate = data.get("candidates", [{}])[0]
            content = (
                candidate.get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            return content or "No response generated."
        else:
            error_msg = response.json().get("error", {}).get("message", "Unknown error")
            return f"⚠️ API Error: {error_msg}"
    except Exception as e:
        return f"⚠️ Request failed: {str(e)}"

# ----------------------------
# MAIN UI
# ----------------------------
st.title("🎭 Gemini Personality Chat")
st.caption("Responses adapt to your personality blend — try moving the sliders!")

# Display model responses in columns
cols = st.columns(3)
for col, (key, info) in zip(cols, MODELS.items()):
    with col:
        with st.container(border=True):
            st.subheader(info["name"])
            resp = st.session_state.responses[key]
            if resp["status"] == "loading":
                st.spinner("Generating...")
            else:
                output = resp["text"]
                if output:
                    st.write(output)
                else:
                    st.info("Waiting for input...")

# ----------------------------
# HANDLE PROMPT SUBMISSION & CONCURRENT GENERATION
# ----------------------------
if any(st.session_state.responses[k]["status"] == "loading" for k in MODELS):
    # Build prompt using current slider values (critical!)
    full_prompt = build_styled_prompt(
        st.session_state.last_prompt,
        pirate, romance, empathy, energy
    )

    def worker(model_key):
        model_id = MODELS[model_key]["id"]
        result = call_gemini(model_id, full_prompt, api_key)
        return model_key, result

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(worker, mk) for mk in MODELS]
        for future in futures:
            model_key, text = future.result()
            st.session_state.responses[model_key]["text"] = text
            st.session_state.responses[model_key]["status"] = "done"
    st.rerun()

# ----------------------------
# CHAT INPUT (sticky at bottom)
# ----------------------------
if user_input := st.chat_input("Ask anything..."):
    st.session_state.last_prompt = user_input
    # Reset all models to loading
    for mk in MODELS:
        st.session_state.responses[mk] = {"text": "", "status": "loading"}
    st.rerun()