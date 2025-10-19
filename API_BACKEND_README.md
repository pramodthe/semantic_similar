# Persona Chatbot Blender API

This is a FastAPI backend that provides API endpoints for the Persona Chatbot Blender application. It replaces the Streamlit UI with REST API endpoints for use with a React frontend.

## Features

- Generate AI responses with different personas (Pirate, Romance, Empathy, Energy)
- Blend responses based on weighted persona preferences
- Semantic similarity matching to find the best blended response
- Health check endpoint

## Requirements

- Python 3.7+
- OpenRouter API key (set as `OPENROUTER_API_KEY` environment variable)

## Installation

```bash
pip install "fastapi[all]"
pip install torch sentence-transformers openai
```

## Running the Server

```bash
uvicorn api_backend:app --host 0.0.0.0 --port 8000 --reload
```

Make sure to set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## API Endpoints

### GET /health
Health check endpoint to verify the service is running and properly configured.

Response:
```json
{
  "status": "ok|error",
  "message": "API is running|Error message"
}
```

### POST /generate_response
Generate a blended AI response based on persona weights.

Request body:
```json
{
  "prompt": "Your message here",
  "persona_weights": {
    "Pirate": 25,
    "Romance": 25,
    "Empathy": 25,
    "Energy": 25
  }
}
```

Response:
```json
{
  "final_response": "The blended AI response",
  "individual_responses": {
    "Pirate": "Pirate response",
    "Romance": "Romance response",
    "Empathy": "Empathy response",
    "Energy": "Energy response"
  }
}
```

## Example Usage

```python
import requests
import json

payload = {
    "prompt": "Tell me about artificial intelligence",
    "persona_weights": {
        "Pirate": 25,
        "Romance": 25,
        "Empathy": 25,
        "Energy": 25
    }
}

response = requests.post("http://localhost:8000/generate_response", 
                        data=json.dumps(payload), 
                        headers={"Content-Type": "application/json"})

result = response.json()
print(result["final_response"])
```

## Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)

## React Frontend Integration

The API is designed to be consumed by a React frontend. For example:

```javascript
const generateResponse = async (prompt, personaWeights) => {
  try {
    const response = await fetch('http://localhost:8000/generate_response', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt,
        persona_weights: personaWeights
      })
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error generating response:', error);
    throw error;
  }
};
```

## API Documentation

Interactive API documentation is available at:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)