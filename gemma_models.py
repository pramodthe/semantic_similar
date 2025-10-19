import streamlit as st
import requests
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Configure the page
st.set_page_config(
    page_title="Gemma Models Comparison",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Gemma Models Comparison")
st.markdown("Enter a prompt below to get responses from three different Gemma models simultaneously.")

# Initialize session state
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

if "responses" not in st.session_state:
    st.session_state.responses = {
        "gemma-27b": {"response": "", "status": "idle"},
        "gemma-12b": {"response": "", "status": "idle"},
        "gemma-4b": {"response": "", "status": "idle"}
    }

# Get API key from secrets
api_key = st.secrets.get("OPENROUTER_API_KEY")

if not api_key:
    st.error("OpenRouter API key not found in secrets. Please add your API key to `.streamlit/secrets.toml`")
    st.stop()

# Define the models
models = {
    "gemma-27b": {
        "id": "google/gemma-3-27b-it:free",
        "name": "Gemma 3 27B",
        "color": "#FF6B6B"
    },
    "gemma-12b": {
        "id": "google/gemma-3-12b-it:free", 
        "name": "Gemma 3 12B",
        "color": "#4ECDC4"
    },
    "gemma-4b": {
        "id": "google/gemma-3-4b-it:free",
        "name": "Gemma 3 4B", 
        "color": "#45B7D1"
    }
}

# Create the layout
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader(f"🧠 {models['gemma-27b']['name']}")
        if st.session_state.responses['gemma-27b']['status'] == 'loading':
            with st.spinner("Generating response..."):
                st.write(st.session_state.responses['gemma-27b']['response'])
        else:
            st.write(st.session_state.responses['gemma-27b']['response'])

with col2:
    with st.container(border=True):
        st.subheader(f"🧠 {models['gemma-12b']['name']}")
        if st.session_state.responses['gemma-12b']['status'] == 'loading':
            with st.spinner("Generating response..."):
                st.write(st.session_state.responses['gemma-12b']['response'])
        else:
            st.write(st.session_state.responses['gemma-12b']['response'])

with col3:
    with st.container(border=True):
        st.subheader(f"🧠 {models['gemma-4b']['name']}")
        if st.session_state.responses['gemma-4b']['status'] == 'loading':
            with st.spinner("Generating response..."):
                st.write(st.session_state.responses['gemma-4b']['response'])
        else:
            st.write(st.session_state.responses['gemma-4b']['response'])

# Function to call the API for a single model
def call_model(model_key, prompt):
    model_info = models[model_key]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_info["id"],
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                return f"Error: No choices in response for {model_info['name']}"
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to get responses from all models
def get_responses(prompt):
    if not prompt.strip():
        return
    
    # Set loading status
    for model_key in models:
        st.session_state.responses[model_key]["status"] = "loading"
        st.session_state.responses[model_key]["response"] = "Loading response..."
    
    # Update the UI to show loading state
    placeholder = st.empty()
    with placeholder.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            with st.container(border=True):
                st.subheader(f"🧠 {models['gemma-27b']['name']}")
                st.write(st.session_state.responses['gemma-27b']['response'])
        
        with col2:
            with st.container(border=True):
                st.subheader(f"🧠 {models['gemma-12b']['name']}")
                st.write(st.session_state.responses['gemma-12b']['response'])
        
        with col3:
            with st.container(border=True):
                st.subheader(f"🧠 {models['gemma-4b']['name']}")
                st.write(st.session_state.responses['gemma-4b']['response'])

    # Make API calls in parallel
    with ThreadPoolExecutor() as executor:
        futures = {
            model_key: executor.submit(call_model, model_key, prompt)
            for model_key in models
        }
        
        for model_key, future in futures.items():
            response = future.result()
            st.session_state.responses[model_key]["response"] = response
            st.session_state.responses[model_key]["status"] = "completed"

# Input section at the bottom
st.markdown("---")
st.subheader("Enter your prompt:")
prompt = st.text_area(
    "Type your message here:", 
    value=st.session_state.prompt,
    height=150,
    key="prompt_input"
)

if st.button("Generate Responses", type="primary"):
    if prompt.strip():
        st.session_state.prompt = prompt
        get_responses(prompt)
        st.rerun()
    else:
        st.warning("Please enter a prompt before submitting.")

# Add some instructions
with st.expander("ℹ️ How to use this app"):
    st.write("""
    1. Enter a prompt in the text box below
    2. Click "Generate Responses" to get results from all three Gemma models simultaneously
    3. View the responses in the three columns above
    4. Each column represents a different Gemma model (27B, 12B, and 4B parameters)
    """)