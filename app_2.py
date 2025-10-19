import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import openai
import torch 

# Setup OpenRouter client
try:
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
except Exception as e:
    st.error(f"Failed to initialize OpenAI client. Make sure your API key is set. Error: {e}")
    client = None

# Load a sentence-transformer model for embedding. This runs once and is cached.
@st.cache_resource
def load_embedding_model():
    """Loads the Sentence Transformer model."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Failed to load the embedding model. Please check your internet connection. Error: {e}")
        return None

embedding_model = load_embedding_model()

# --- Core Functions ---

def generate_persona_response(client, prompt, persona):
    """
    Calls the OpenRouter API to generate a response based on a specific persona.
    """
    if not client:
        return f"Error: API client is not configured. Cannot generate {persona} response."

    system_prompts = {
        "Pirate": "You are a swashbuckling pirate. Respond to all prompts with a pirate accent and vocabulary. Be adventurous and a little bit cheeky.",
        "Romance": "You are a hopeless romantic. Respond to all prompts with poetic, loving, and affectionate language. Speak from the heart.",
        "Empathy": "You are a deeply empathetic and caring friend. Respond with kindness, understanding, and support. Offer a listening ear.",
        "Energy": "You are an extremely energetic and enthusiastic hype-person. Respond with high energy, positivity, and lots of exclamation points! Motivate and excite!"
    }

    try:
        response = client.chat.completions.create(
            model="google/gemma-2-9b-it",  # You can choose any suitable model
            messages=[
                {"role": "system", "content": system_prompts.get(persona, "You are a helpful assistant.")},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred while generating the {persona} response: {e}")
        return f"Sorry, I couldn't generate a {persona} response due to an API error."


# --- REPLACED FUNCTION ---
def find_best_match(personas, persona_weights, generated_responses, model):
    """
    Finds the best response by blending persona embeddings and using cosine similarity.
    """
    if not model:
        st.warning("Embedding model not loaded. Cannot perform similarity search.")
        return "Error: Embedding model is unavailable."

    # 1. Embed the persona keywords and the generated responses
    # These tensors will be on the model's device (e.g., 'cpu' or 'mps' on a Mac)
    persona_keywords = [p.lower() for p in personas]
    persona_embeddings = model.encode(persona_keywords, convert_to_tensor=True)
    response_embeddings = model.encode(generated_responses, convert_to_tensor=True)

    # 2. Calculate the weighted average "target" vector *on the model's device*
    
    # Get the device (e.g., 'mps' or 'cpu') from the embeddings
    device = persona_embeddings.device

    # Create weights tensor directly on the correct device
    weights = torch.tensor(list(persona_weights.values()), dtype=torch.float32, device=device)
    
    # Normalize weights
    if weights.sum() == 0:
        # Avoid division by zero if all weights are 0
        weights = torch.ones_like(weights, device=device) / len(weights)
    else:
        weights = weights / weights.sum()

    # Calculate the weighted average "target" vector using torch operations
    # unsqueeze(1) makes weights [num_personas, 1] for broadcasting against [num_personas, embedding_dim]
    target_vector = torch.sum(weights.unsqueeze(1) * persona_embeddings, axis=0)
    
    # 3. Calculate cosine similarity. Both tensors are now on the same device.
    similarities = util.cos_sim(target_vector, response_embeddings)
    
    # 4. Find the index of the highest similarity score
    # Use torch.argmax and .item() to get the index as a Python integer
    best_match_index = torch.argmax(similarities).item()
    
    return generated_responses[best_match_index]
# --- END OF REPLACED FUNCTION ---


# --- Streamlit UI ---

st.set_page_config(page_title="Persona Chatbot Blender", layout="wide")

st.title("🤖 Persona Chatbot Blender")
st.markdown("Adjust the dials to blend different chatbot personalities and see the magic of semantic similarity at work!")

# Check for API key and model readiness
if not os.getenv("OPENROUTER_API_KEY"):
    st.warning("Your OpenRouter API key is not set. Please add it as an environment variable named `OPENROUTER_API_KEY`.", icon="🔑")
if not client or not embedding_model:
    st.stop()


# -- Sidebar for Persona Dials --
with st.sidebar:
    st.header("Persona Dials")
    st.markdown("Set the personality mix. The total will always be 100%.")

    personas = ["Pirate", "Romance", "Empathy", "Energy"]
    
    if 'sliders' not in st.session_state:
        st.session_state.sliders = {persona: 25 for persona in personas}

    # Use a dictionary to store slider values
    sliders = {}
    for persona in personas:
        sliders[persona] = st.slider(
            persona, 0, 100, st.session_state.sliders[persona], key=f"slider_{persona}"
        )

    # Normalization logic
    total = sum(sliders.values())
    if total > 0:
        factor = 100 / total
        normalized_sliders = {key: int(value * factor) for key, value in sliders.items()}
        
        # Adjust for rounding errors to ensure sum is exactly 100
        diff = 100 - sum(normalized_sliders.values())
        if diff != 0:
            # Add the difference to the largest value
            max_key = max(normalized_sliders, key=normalized_sliders.get)
            normalized_sliders[max_key] += diff

        st.session_state.sliders = normalized_sliders

    # Display current weights
    st.markdown("---")
    st.markdown("##### Current Blend:")
    for persona, weight in st.session_state.sliders.items():
        st.markdown(f"**{persona}:** `{weight}%`")
    st.markdown("---")


# -- Main Chat Interface --
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("What would you like to say?")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        final_response_placeholder = st.empty()
        
        with st.spinner("Generating persona responses..."):
            persona_responses = {
                persona: generate_persona_response(client, user_prompt, persona)
                for persona in personas
            }

        with st.spinner("Embedding and calculating the best blend..."):
            final_response = find_best_match(
                personas,
                st.session_state.sliders,
                list(persona_responses.values()),
                embedding_model
            )
            final_response_placeholder.markdown(final_response)

        # Optionally, show the individual persona responses for comparison
        with st.expander("See Individual Persona Responses"):
            for persona, response in persona_responses.items():
                st.markdown(f"**{persona} ({st.session_state.sliders[persona]}%):**")
                st.info(response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})