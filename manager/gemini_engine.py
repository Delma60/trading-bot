import os
import google.generativeai as genai

class GeminiEngine:
    """Handles all interactions with the Google Gemini API for smart fallback and reasoning."""
    
    def __init__(self, model_name: str = 'gemini-1.5-flash'):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.is_ready = False
        
        if not self.api_key:
            print("⚠️ [Gemini Engine]: GEMINI_API_KEY environment variable not set. Smart routing disabled.")
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(model_name)
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
        You are the NLP routing engine for an algorithmic trading bot.
        The user typed: "{user_input}"
        
        My local model guessed the intent is: "{local_guess}"
        
        Here is the master list of all valid intents you can choose from:
        {valid_intents}
        
        Which intent does the user's message best match? 
        Respond ONLY with the exact intent string. If it matches nothing, respond with "UNKNOWN".
        """
        
        try:
            response = self.model.generate_content(prompt)
            gemini_intent = response.text.strip()
            
            if gemini_intent in valid_intents:
                return gemini_intent
            else:
                return "UNKNOWN"
                
        except Exception as e:
            print(f"[Gemini Engine]: API Error during routing - {e}")
            return "UNKNOWN"
            
    def ask_general_question(self, question: str) -> str:
        """Allows the bot to answer general trading questions."""
        if not self.is_ready:
            return "I am currently offline. Please set my API key to enable general chat."
            
        try:
            response = self.model.generate_content(f"You are a helpful trading assistant. Answer concisely: {question}")
            return response.text
        except Exception as e:
            return f"I encountered an error: {e}"