import json
import pickle
import re
import threading
import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from pathlib import Path

class NLPEngine:
    """Handles intent prediction and text processing."""
    
    def __init__(self, intents_filepath: str, data_dir: str = "data"):
        self.stemmer = LancasterStemmer()
        self.intents_file = Path(intents_filepath)
        self.model_file = Path(data_dir) / "chatbot_model.keras"
        self.pickle_file = Path(data_dir) / "data.pickle"
        
        self.words, self.labels, self.training, self.output = [], [], [], []
        self.nlp_model = None
        self.intents_data = {}
        
        self._initialize()

    def _initialize(self):
        with self.intents_file.open("r") as f: 
            self.intents_data = json.load(f)
        self._process_data()
        self._build_model()

    def predict_intent(self, text: str) -> tuple[str, float]:
        """Returns the predicted intent tag and its confidence score."""
        bag = self._bag_of_words(text)
        results = self.nlp_model.predict(bag, verbose=0)[0]
        best_index = np.argmax(results)
        return self.labels[best_index], results[best_index]
    
    
    # manager/nlp_engine.py

    def learn_new_intent(self, tag: str, pattern: str, response: str, notify_callback=print):
        """Creates a new intent in the JSON file and triggers retraining."""
        # 1. Update the in-memory JSON data
        new_intent = {
            "tag": tag,
            "patterns": [pattern],
            "responses": [response]
        }
        
        if 'intents' not in self.intents_data:
            self.intents_data['intents'] = []
            
        self.intents_data['intents'].append(new_intent)
                
        # 2. Write back to intents.json
        try:
            with open(self.intents_file, "w") as f:
                json.dump(self.intents_data, f, indent=4)
        except Exception as e:
            notify_callback(f"⚠️ Failed to save new intent: {e}", priority="normal")
            return False
                
        # 3. Retrain immediately
        notify_callback(f"🧠 I've added '{tag}' to my knowledge base. Retraining...", priority="normal")
        self.background_training()
        notify_callback(f"✅ Retraining complete! Model updated.", priority="normal")
        return True
    
    def add_intent_pattern(self, tag: str, new_pattern: str, notify_callback=print):
        """Adds a new phrase to the intents JSON and retrains the model."""
        intent_found = False
        
        # 1. Update the in-memory JSON data
        for intent in self.intents_data.get('intents', []):
            if intent['tag'] == tag:
                if new_pattern not in intent['patterns']:
                    intent['patterns'].append(new_pattern)
                intent_found = True
                break
                
        if not intent_found:
            notify_callback(f"⚠️ Could not find intent '{tag}' in intents.json", priority="normal")
            return False
            
        # 2. Write the updated data back to intents.json
        try:
            with open(self.intents_file, "w") as f:
                json.dump(self.intents_data, f, indent=4)
        except Exception as e:
            notify_callback(f"⚠️ Failed to save to intents.json: {e}", priority="normal")
            return False
            
        # 3. Retrain the model immediately
        notify_callback(f"🧠 I am learning! Retraining my neural network...", priority="normal")
        self.background_training()
        notify_callback(f"✅ Retraining complete! Model updated.", priority="normal")
        
    def background_training(self):
        try:
            # 1. Force delete the old memory caches so it doesn't just reload them!
            if self.pickle_file.exists():
                self.pickle_file.unlink()
            if self.model_file.exists():
                self.model_file.unlink()
            
            # 2. Rebuild the brain from the freshly updated intents.json
            self._process_data()
            self._build_model()
            
            # notify_callback(f"✅ Learning complete! I am now fully updated.", priority="normal")
        except Exception as e:
            pass
            # notify_callback(f"⚠️ Error during retraining: {e}", priority="normal")
                
    def get_response_template(self, tag: str) -> str:
        """Fetches a random response string for a given intent tag."""
        import random
        intent_data = next((item for item in self.intents_data.get("intents", []) if item['tag'] == tag), None)
        if intent_data and intent_data.get('responses'):
            return random.choice(intent_data['responses'])
        return ""

    def extract_entities(self, text: str) -> dict:
        """Parses the text for trading-specific variables (entities)."""
        text_upper = text.upper()
        
        # Find 6-letter trading symbols (EURUSD, GBPUSD, BTCUSD, etc.)
        symbols = re.findall(r'\b[A-Z]{6}\b', text_upper)
        
        # Find timeframes (M1, M5, M15, M30, H1, H4, D1, W1, MN)
        timeframes = re.findall(r'\b(M1|M5|M15|M30|H1|H4|D1|W1|MN)\b', text_upper)
        
        # Find dollar amounts (e.g., "$50", "50 bucks", "100 dollars")
        money_matches = re.findall(r'\$?(\d+(?:\.\d+)?)\s*(?:DOLLARS|BUCKS)?', text_upper)
        
        # Find percentages (e.g., "1.5%", "5 percent")
        percentages = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|PERCENT)', text_upper)
        
        
        # Extract trade direction
        direction_match = re.search(r'\b(BUY|LONG|SELL|SHORT)\b', text_upper)
        direction = direction_match.group(1) if direction_match else None
        if direction == 'LONG': direction = 'BUY'
        if direction == 'SHORT': direction = 'SELL'
        
        return {
            "symbols": list(dict.fromkeys(symbols)) if symbols else [],  # Remove duplicates
            "timeframes": list(dict.fromkeys(timeframes)) if timeframes else [],
            "money": [float(m) for m in money_matches] if money_matches else [],
            "percentages": [float(p) for p in percentages] if percentages else [],
            "direction": direction
        }


    # ... Move _bag_of_words, _process_data, and _build_model from old chat.py into here ...
    # (Omitted for brevity, but you just copy-paste them from your original code)
    def _bag_of_words(self, s: str) -> np.ndarray:
        bag = [0 for _ in range(len(self.words))]
        s_words = [self.stemmer.stem(word.lower()) for word in nltk.word_tokenize(s)]
        
        for se in s_words:
            for i, w in enumerate(self.words):
                if w == se:
                    bag[i] = 1
                    
        return np.array([bag])
    
    def _build_model(self):
        if self.model_file.exists():
            self.nlp_model = load_model(str(self.model_file))
        else:
            self.nlp_model = Sequential([
                Dense(8, input_shape=(len(self.training[0]),), activation='relu'),
                Dense(8, activation='relu'),
                Dense(len(self.output[0]), activation='softmax')
            ])
            self.nlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.nlp_model.fit(self.training, self.output, epochs=200, batch_size=8, verbose=1)
            self.nlp_model.save(str(self.model_file))

    
    def _process_data(self):
        if self.pickle_file.exists():
            with self.pickle_file.open("rb") as f:
                self.words, self.labels, self.training, self.output = pickle.load(f)
        else:
            docs_x = []
            docs_y = []

            for intent in self.intents_data['intents']:
                for pattern in intent['patterns']:
                    wrds = nltk.word_tokenize(pattern)
                    self.words.extend(wrds)
                    docs_x.append(wrds) 
                    docs_y.append(intent['tag'])
                    
                if intent['tag'] not in self.labels:
                    self.labels.append(intent['tag'])

            self.words = sorted(list(set([self.stemmer.stem(w.lower()) for w in self.words if w not in ["?", "!", ".", ","]])))
            self.labels = sorted(self.labels)

            out_empty = [0 for _ in range(len(self.labels))]

            for x, doc in enumerate(docs_x):
                bag = []
                wrds = [self.stemmer.stem(w.lower()) for w in doc]
                for w in self.words:
                    bag.append(1 if w in wrds else 0)
                    
                output_row = out_empty[:]
                output_row[self.labels.index(docs_y[x])] = 1
                
                self.training.append(bag)
                self.output.append(output_row)

            self.training = np.array(self.training)
            self.output = np.array(self.output)

            with self.pickle_file.open("wb") as f:
                pickle.dump((self.words, self.labels, self.training, self.output), f)

    