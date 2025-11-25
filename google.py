import google.generativeai as genai
import os

# Replace with your actual API key
API_KEY = "AIzaSyBY_MCYBs0GDVxuZgaBcB-mCRQo8rqUKSY" 

genai.configure(api_key=API_KEY)

print("Available Generative Models for your API Key:")
print("-" * 40)

try:
    for m in genai.list_models():
        # We only care about models that generate text/content
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model Name: {m.name}")
            print(f"Display Name: {m.display_name}")
            print("-" * 20)
except Exception as e:
    print(f"Error fetching models: {e}")