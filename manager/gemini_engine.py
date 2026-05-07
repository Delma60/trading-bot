import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
class GeminiEngine:
    """Handles all interactions with the Google Gemini API for smart fallback and reasoning."""
    
    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.is_ready = False
        
        if not self.api_key:
            print("⚠️ [Gemini Engine]: GEMINI_API_KEY environment variable not set. Smart routing disabled.")
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel(model_name)
                self.is_ready = True
            except Exception as e:
                print(f"⚠️ [Gemini Engine]: Failed to initialize API: {e}")

    def route_intent(self, user_input: str, valid_intents: list, local_guess: str) -> str:
        """
        Acts as a smart fallback when the local Keras model is unsure.
        Returns the matching intent string, or "UNKNOWN".
        """
        if not self.is_ready:
            return "UNKNOWN"

        prompt = f"""
            You are an adaptive NLP layer for a trading bot. 
            User input: "{user_input}"
            Local model guess: "{local_guess}"
            Known categories: {valid_intents}

            Instructions:
            1. If it matches a 'Known category', respond with ONLY that tag.
            2. If it's a general question, respond with: "GENERAL_CHAT: [Your concise answer]"
            3. If it's a new type of command, respond with: "SUGGEST_NEW: [Short_Tag] | [A response for the bot to give]"

            Respond in one of those three formats ONLY.
            """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            gemini_intent = response.text.strip()
            print(f"[Gemini Engine]: Rerouted intent: '{local_guess}' => '{gemini_intent}'")
            
            return gemini_intent
                
        except Exception as e:
            print(f"[Gemini Engine]: API Error during routing - {e}")
            return "UNKNOWN"
            
    def ask_general_question(self, question: str) -> str:
        """Allows the bot to answer general trading questions."""
        if not self.is_ready:
            return "I am currently offline. Please set my API key to enable general chat."
            
        try:
            response = self.gemini_model.generate_content(f"You are a helpful trading assistant. Answer concisely: {question}")
            return response.text
        except Exception as e:
            return f"I encountered an error: {e}"