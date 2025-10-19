import os
import openai
import torch
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List

# Initialize FastAPI app
app = FastAPI(title="Persona Chatbot Blender API", version="1.0.0")

# Setup OpenRouter client
try:
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
except Exception as e:
    print(f"Failed to initialize OpenAI client. Make sure your API key is set. Error: {e}")
    client = None

# Load a sentence-transformer model for embedding. This runs once at startup.
def load_embedding_model():
    """Loads the Sentence Transformer model."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        print(f"Failed to load the embedding model. Please check your internet connection. Error: {e}")
        return None

embedding_model = load_embedding_model()

# Pydantic models for request/response
class PersonaWeights(BaseModel):
    Pirate: int = 0
    Romance: int = 0
    Empathy: int = 0
    Energy: int = 0

class GenerateRequest(BaseModel):
    prompt: str
    persona_weights: PersonaWeights

class GenerateResponse(BaseModel):
    final_response: str
    individual_responses: Dict[str, str]

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
        print(f"An error occurred while generating the {persona} response: {e}")
        return f"Sorry, I couldn't generate a {persona} response due to an API error."

def find_best_match(personas, persona_weights, generated_responses, model):
    """
    Finds the best response by blending persona embeddings and using cosine similarity.
    """
    if not model:
        print("Embedding model not loaded. Cannot perform similarity search.")
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

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not os.getenv("OPENROUTER_API_KEY"):
        return {"status": "error", "message": "OPENROUTER_API_KEY not set"}
    if not client:
        return {"status": "error", "message": "OpenAI client not initialized"}
    if not embedding_model:
        return {"status": "error", "message": "Embedding model not loaded"}
    return {"status": "ok", "message": "API is running"}

@app.post("/generate_response", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """Generate a blended response based on persona weights"""
    
    # Validate that persona weights sum to 100
    weights = request.persona_weights
    total_weight = weights.Pirate + weights.Romance + weights.Empathy + weights.Energy
    if total_weight != 100:
        # Normalize weights to sum to 100
        if total_weight > 0:
            factor = 100 / total_weight
            # Apply the factor to each weight, converting to int
            weights.Pirate = int(weights.Pirate * factor)
            weights.Romance = int(weights.Romance * factor)
            weights.Empathy = int(weights.Empathy * factor)
            weights.Energy = int(weights.Energy * factor)
            
            # Adjust for rounding errors to ensure sum is exactly 100
            diff = 100 - (weights.Pirate + weights.Romance + weights.Empathy + weights.Energy)
            if diff != 0:
                # Add the difference to the largest value
                max_weight = max(weights.Pirate, weights.Romance, weights.Empathy, weights.Energy)
                if max_weight == weights.Pirate:
                    weights.Pirate += diff
                elif max_weight == weights.Romance:
                    weights.Romance += diff
                elif max_weight == weights.Empathy:
                    weights.Empathy += diff
                else:
                    weights.Energy += diff

    # Validate API key and model readiness
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable not set")
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")

    try:
        # Define the personas
        personas = ["Pirate", "Romance", "Empathy", "Energy"]
        
        # Generate responses for each persona
        persona_responses = {}
        for persona in personas:
            weight = getattr(weights, persona)
            if weight > 0:  # Only generate if the persona has a positive weight
                persona_responses[persona] = generate_persona_response(client, request.prompt, persona)
            else:
                persona_responses[persona] = f"[Skipped - Weight: 0%]"

        # Find the best match using semantic similarity
        persona_weights_dict = {
            "Pirate": weights.Pirate / 100.0,
            "Romance": weights.Romance / 100.0,
            "Empathy": weights.Empathy / 100.0,
            "Energy": weights.Energy / 100.0
        }
        
        final_response = find_best_match(
            personas,
            persona_weights_dict,
            list(persona_responses.values()),
            embedding_model
        )

        return GenerateResponse(
            final_response=final_response,
            individual_responses=persona_responses
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")