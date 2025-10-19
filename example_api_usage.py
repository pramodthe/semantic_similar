"""
Example script to demonstrate how to call the FastAPI endpoints
This simulates what your React frontend would do using HTTP requests
"""

import requests
import json

# Base URL of the API (when running locally)
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health check response:", response.json())

def test_generate_response():
    """Test the generate_response endpoint"""
    # Example payload matching the Pydantic model
    payload = {
        "prompt": "Tell me about artificial intelligence",
        "persona_weights": {
            "Pirate": 25,
            "Romance": 25,
            "Empathy": 25,
            "Energy": 25
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/generate_response", 
                                data=json.dumps(payload), 
                                headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("Final response:", result["final_response"])
            print("\nIndividual responses:")
            for persona, response_text in result["individual_responses"].items():
                print(f"  {persona}: {response_text}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to the API. Make sure the FastAPI server is running on localhost:8000")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print("Testing the FastAPI backend...")
    test_health()
    print()
    test_generate_response()