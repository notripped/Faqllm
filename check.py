import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your GOOGLE_API_KEY from a .env file
load_dotenv()

# Configure the API key
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set your API key in a .env file or directly in your environment.")
    exit()

print("Available Gemini Models:")
print("-----------------------")

# List all available models
for m in genai.list_models():
    # Filter for models that support text generation if you're primarily interested in that
    # The 'generateContent' method is for text generation
    if "generateContent" in m.supported_generation_methods:
        print(f"Name: {m.name}")
        print(f"  Display Name: {m.display_name}")
        print(f"  Description: {m.description}")
        print(f"  Input Token Limit: {m.input_token_limit}")
        print(f"  Output Token Limit: {m.output_token_limit}")
        print(f"  Supported Methods: {m.supported_generation_methods}")
        print("-" * 30)

# Optional: You can also filter for models that support embeddings or other methods
print("\nAvailable Embedding Models:")
print("--------------------------")
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(f"Name: {m.name}")
        print(f"  Display Name: {m.display_name}")
        print(f"  Description: {m.description}")
        print(f"  Supported Methods: {m.supported_generation_methods}")
        print("-" * 30)