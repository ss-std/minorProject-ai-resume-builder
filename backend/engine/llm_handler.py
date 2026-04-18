import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

class AIHandler:
    def __init__(self, provider="gemini"):
        self.provider = provider
        # i remove the http_options to let the SDK auto-select the best endpoint
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY", "").strip()
        )

    def generate_content(self, prompt):
        # i am  using 'gemini-2.0-flash' as the primary because it is the 2026 stable standard
        # If that fails,  try the 1.5 version
        models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash"]
        
        for model_name in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model_name, 
                    contents=prompt
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                error_msg = str(e).lower()
                # If it's a 404,  try the next model in my list
                if "404" in error_msg:
                    continue
                # If it's a 429,  tell the user to wait
                if "429" in error_msg:
                    return "Error: Quota full. Please wait 1 minute."
                return f"AI Error: {str(e)}"
        
        return "Error: Could not connect to any AI models. Please check your API key."