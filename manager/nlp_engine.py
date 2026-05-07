import json
import pickle
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
        self.model = None
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
        results = self.model.predict(bag, verbose=0)[0]
        best_index = np.argmax(results)
        return self.labels[best_index], results[best_index]

    def get_response_template(self, tag: str) -> str:
        """Fetches a random response string for a given intent tag."""
        import random
        intent_data = next((item for item in self.intents_data.get("intents", []) if item['tag'] == tag), None)
        if intent_data and intent_data.get('responses'):
            return random.choice(intent_data['responses'])
        return ""

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
        if self.MODEL_FILE.exists():
            self.model = load_model(str(self.MODEL_FILE))
        else:
            self.model = Sequential([
                Dense(8, input_shape=(len(self.training[0]),), activation='relu'),
                Dense(8, activation='relu'),
                Dense(len(self.output[0]), activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.fit(self.training, self.output, epochs=200, batch_size=8, verbose=1)
            self.model.save(str(self.MODEL_FILE))

    
    def _process_data(self):
        if self.DATA_PICKLE.exists():
            with self.DATA_PICKLE.open("rb") as f:
                self.words, self.labels, self.training, self.output = pickle.load(f)
        else:
            docs_x = []
            docs_y = []

            for intent in self.data['intents']:
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

            with self.DATA_PICKLE.open("wb") as f:
                pickle.dump((self.words, self.labels, self.training, self.output), f)

    