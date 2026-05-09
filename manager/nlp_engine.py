"""
manager/nlp_engine.py

Upgraded NLP engine:
 - Symbol aliases: "gold" → XAUUSD, "bitcoin" → BTCUSD, etc.
 - Better entity extraction: lot sizes, pip values, 7-char symbols
 - Sentiment detection for user tone (positive / negative / neutral)
 - All existing functionality preserved
"""

import json
import pickle
import re
import threading
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import Dense                   # type: ignore
from pathlib import Path


# ── Alias table ──────────────────────────────────────────────────────────────
# Maps natural language terms → canonical broker symbols
SYMBOL_ALIASES: dict[str, str] = {
    # Metals
    "GOLD":       "XAUUSD",
    "SILVER":     "XAGUSD",
    "PLATINUM":   "XPTUSD",
    "PALLADIUM":  "XPDUSD",
    # Crypto
    "BITCOIN":    "BTCUSD",
    "BTC":        "BTCUSD",
    "ETHEREUM":   "ETHUSD",
    "ETH":        "ETHUSD",
    "LITECOIN":   "LTCUSD",
    "LTC":        "LTCUSD",
    # Commodities / indices
    "OIL":        "USOIL",
    "CRUDEOIL":   "USOIL",
    "NATGAS":     "NGAS",
    "DOW":        "US30",
    "SP500":      "US500",
    "NASDAQ":     "US100",
    "DAX":        "GER40",
    # Forex shorthands
    "EURO":       "EURUSD",
    "POUND":      "GBPUSD",
    "CABLE":      "GBPUSD",
    "SWISSY":     "USDCHF",
    "LOONIE":     "USDCAD",
    "AUSSIE":     "AUDUSD",
    "KIWI":       "NZDUSD",
    "YEN":        "USDJPY",
    # Major USD pairs
    "EURUSD":     "EURUSD",
    "GBPUSD":     "GBPUSD",
    "USDJPY":     "USDJPY",
    "USDCHF":     "USDCHF",
    "USDCAD":     "USDCAD",
    "AUDUSD":     "AUDUSD",
    "NZDUSD":     "NZDUSD",
    "USDCNH":     "USDCNH",
    "USDSEK":     "USDSEK",
    "USDNOK":     "USDNOK",
    "USDSGD":     "USDSGD",
    "USDHKD":     "USDHKD",
    "USDZAR":     "USDZAR",
    "USDMXN":     "USDMXN",
    "USDTRY":     "USDTRY",
    "USDRUB":     "USDRUB",
    "USDTHB":     "USDTHB",
    "USDPHP":     "USDPHP",
    "USDIDR":     "USDIDR",
    "USDKRW":     "USDKRW",
    "USDCLP":     "USDCLP",
    "USDPLN":     "USDPLN",
    "USDCZE":     "USDCZE",
    "USDHUF":     "USDHUF",
    "USDRON":     "USDRON",
    # Cross pairs (non-USD)
    "EUROGBP":    "EURGBP",
    "EURCHF":     "EURCHF",
    "EURJPY":     "EURJPY",
    "EURCAD":     "EURCAD",
    "EURAUD":     "EURAUD",
    "EURNZD":     "EURNZD",
    "GBPJPY":     "GBPJPY",
    "GBPCHF":     "GBPCHF",
    "GBPCAD":     "GBPCAD",
    "GBPAUD":     "GBPAUD",
    "GBPNZD":     "GBPNZD",
    "AUDJPY":     "AUDJPY",
    "AUDCAD":     "AUDCAD",
    "AUDNZD":     "AUDNZD",
    "NZDJPY":     "NZDJPY",
    "CADJPY":     "CADJPY",
    "CHFJPY":     "CHFJPY",
}

_POSITIVE_WORDS = {
    "good", "great", "excellent", "nice", "awesome", "profit", "win",
    "bull", "bullish", "long", "buy", "up", "rising", "breakout", "strong",
}
_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "loss", "crash", "drop", "bear", "bearish",
    "short", "sell", "down", "falling", "weak", "risk", "danger", "worried",
}


class NLPEngine:
    """Handles intent prediction, entity extraction, and text processing."""

    def __init__(self, intents_filepath: str, data_dir: str = "data"):
        self.stemmer      = PorterStemmer()
        self.intents_file = Path(intents_filepath)
        self.model_file   = Path(data_dir) / "chatbot_model.keras"
        self.pickle_file  = Path(data_dir) / "data.pickle"

        self.words: list    = []
        self.labels: list   = []
        self.training: list = []
        self.output: list   = []
        self.nlp_model      = None
        self.intents_data: dict = {}
        self.is_training = False
        self.model_ready = threading.Event()
        self._training_lock = threading.Lock()
        self._train_thread = None
        self._pending_retrain = False

        self._initialize()

    def _initialize(self):
        with self.intents_file.open("r") as f:
            self.intents_data = json.load(f)
        self._process_data()
        self._build_model()

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_intent(self, text: str) -> tuple[str, float]:
        """Returns (intent_tag, confidence) for the given text."""
        if self.nlp_model is None:
            self.model_ready.wait(timeout=5)
            if self.nlp_model is None:
                return "general", 0.0
        bag     = self._bag_of_words(text)
        results = self.nlp_model.predict(bag, verbose=0)[0]
        idx     = int(np.argmax(results))
        return self.labels[idx], float(results[idx])

    def process(self, text: str) -> dict:
        """Combined intent prediction and entity extraction."""
        intent, confidence = self.predict_intent(text)
        entities = self.extract_entities(text)
        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities.get('symbols', [])
        }

    # ── Entity extraction ─────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> dict:
        """
        Parses text for trading-specific variables.

        Extracted fields
        ----------------
        symbols    : list[str]  — broker symbols (aliases resolved, deduped)
        timeframes : list[str]  — M1, M5, M15, M30, H1, H4, D1, W1, MN
        money      : list[float]— dollar amounts
        lots       : float|None — explicit lot size (e.g. "0.5 lots")
        percentages: list[float]— percentage values
        direction  : str|None   — BUY or SELL (LONG/SHORT normalised)
        sentiment  : str        — positive | negative | neutral
        """
        text_upper = text.upper()
        words_lower = text.lower().split()

        # ── Symbols ───────────────────────────────────────────────────────────
        # 1. Resolve natural language aliases first
        alias_symbols = []
        for alias, canonical in SYMBOL_ALIASES.items():
            # Match whole word (e.g. "gold" not "golden")
            if re.search(rf'\b{re.escape(alias)}\b', text_upper):
                alias_symbols.append(canonical)

        # 2. Extract 6–7 letter uppercase ticker patterns
        ticker_symbols = [
            token for token in re.findall(r'\b[A-Z]{6,7}\b', text_upper)
            if token.endswith(("USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"))
        ]

        # 3. Merge and deduplicate (alias results take priority)
        all_symbols = alias_symbols + [s for s in ticker_symbols if s not in alias_symbols]
        symbols = list(dict.fromkeys(all_symbols))

        # ── Timeframes ────────────────────────────────────────────────────────
        timeframes = list(dict.fromkeys(
            re.findall(r'\b(M1|M5|M15|M30|H1|H4|D1|W1|MN)\b', text_upper)
        ))

        # ── Dollar amounts ────────────────────────────────────────────────────
        money_patterns = []
        money_patterns.extend(re.findall(r'(?:US\$|\$)\s*(\d+(?:\.\d+)?)\b', text_upper))
        money_patterns.extend(re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:USD|DOLLARS?|BUCKS?)\b', text_upper))
        money = [float(amount) for amount in money_patterns if amount and float(amount) > 0]

        # ── Explicit lot size ─────────────────────────────────────────────────
        lot_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:LOTS?|LOT)', text_upper)
        lots = float(lot_match.group(1)) if lot_match else None

        # ── Percentages ──────────────────────────────────────────────────────
        percentages = [
            float(p) for p in
            re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|PERCENT)', text_upper)
        ]

        # ── Direction ─────────────────────────────────────────────────────────
        dir_match = re.search(r'\b(BUY|LONG|SELL|SHORT)\b', text_upper)
        direction = dir_match.group(1) if dir_match else None
        if direction == "LONG":  direction = "BUY"
        if direction == "SHORT": direction = "SELL"

        # ── Sentiment ─────────────────────────────────────────────────────────
        words_set = set(words_lower)
        pos_hits  = len(words_set & _POSITIVE_WORDS)
        neg_hits  = len(words_set & _NEGATIVE_WORDS)
        if pos_hits > neg_hits:   sentiment = "positive"
        elif neg_hits > pos_hits: sentiment = "negative"
        else:                     sentiment = "neutral"

        return {
            "symbols":     symbols,
            "timeframes":  timeframes,
            "money":       money,
            "lots":        lots,
            "percentages": percentages,
            "direction":   direction,
            "sentiment":   sentiment,
        }

    # ── Sentiment shortcut ────────────────────────────────────────────────────

    def detect_sentiment(self, text: str) -> str:
        """Returns 'positive', 'negative', or 'neutral'."""
        return self.extract_entities(text)["sentiment"]

    # ── Learning / Retraining ─────────────────────────────────────────────────

    def learn_new_intent(self, tag: str, pattern: str, response: str,
                         notify_callback=print) -> bool:
        """Creates a brand-new intent entry and triggers retraining."""
        new_intent = {"tag": tag, "patterns": [pattern], "responses": [response]}
        self.intents_data.setdefault("intents", []).append(new_intent)

        if not self._save_intents(notify_callback):
            return False

        notify_callback(f"🧠 Added '{tag}' to knowledge base. Retraining...", priority="normal")
        self.background_training(wait=True, timeout=120)
        notify_callback("✅ Retraining complete!", priority="normal")
        return True

    def add_intent_pattern(self, tag: str, new_pattern: str,
                           notify_callback=print) -> bool:
        """Adds a new example phrase to an existing intent and retrains."""
        for intent in self.intents_data.get("intents", []):
            if intent["tag"] == tag:
                if new_pattern not in intent["patterns"]:
                    intent["patterns"].append(new_pattern)
                if not self._save_intents(notify_callback):
                    return False
                notify_callback("🧠 Learning new phrase. Retraining...", priority="normal")
                self.background_training(wait=True, timeout=120)
                notify_callback("✅ Retraining complete!", priority="normal")
                return True

        notify_callback(f"⚠️ Intent '{tag}' not found in intents.json", priority="normal")
        return False

    def background_training(self, wait: bool = False, timeout: float = 120.0):
        """Wipe cached model/pickle and rebuild from updated intents.json."""
        with self._training_lock:
            if self._train_thread and getattr(self._train_thread, 'is_alive', lambda: False)():
                self._pending_retrain = True
            else:
                for path in (self.pickle_file, self.model_file):
                    if path.exists():
                        path.unlink()
                self._process_data()
                self._build_model()

        if wait:
            self.model_ready.wait(timeout=timeout)

    def get_response_template(self, tag: str) -> str:
        import random
        intent = next(
            (i for i in self.intents_data.get("intents", []) if i["tag"] == tag), None
        )
        return random.choice(intent["responses"]) if intent and intent.get("responses") else ""

    # ── Internal NLP mechanics ────────────────────────────────────────────────

    def _bag_of_words(self, s: str) -> np.ndarray:
        bag    = [0] * len(self.words)
        tokens = [self.stemmer.stem(w.lower()) for w in nltk.word_tokenize(s)]
        for token in tokens:
            for i, w in enumerate(self.words):
                if w == token:
                    bag[i] = 1
        return np.array([bag])

    def _build_model(self):
        if self.model_file.exists():
            self.nlp_model = load_model(str(self.model_file))
            self.model_ready.set()
            return

        def train_model():
            self.is_training = True
            self.model_ready.clear()
            try:
                model = Sequential([
                    Dense(128, input_shape=(len(self.training[0]),), activation="relu"),
                    Dense(64, activation="relu"),
                    Dense(len(self.output[0]), activation="softmax"),
                ])
                model.compile(
                    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
                )
                model.fit(self.training, self.output, epochs=200, batch_size=8, verbose=0)
                model.save(str(self.model_file))
                self.nlp_model = model
            finally:
                self.is_training = False
                with self._training_lock:
                    if self._pending_retrain:
                        self._pending_retrain = False
                        for path in (self.pickle_file, self.model_file):
                            if path.exists():
                                path.unlink()
                        self._process_data()
                        self._train_thread = threading.Thread(target=train_model, daemon=True)
                        self._train_thread.start()
                        return
                    self.model_ready.set()

        with self._training_lock:
            if self._train_thread and getattr(self._train_thread, 'is_alive', lambda: False)():
                self._pending_retrain = True
                return
            self._train_thread = threading.Thread(target=train_model, daemon=True)
            self._train_thread.start()

    def _process_data(self):
        if self.pickle_file.exists():
            with self.pickle_file.open("rb") as f:
                self.words, self.labels, self.training, self.output = pickle.load(f)
            return

        docs_x, docs_y = [], []

        for intent in self.intents_data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                self.words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])
            if intent["tag"] not in self.labels:
                self.labels.append(intent["tag"])

        stop = {"?", "!", ".", ","}
        self.words = sorted({self.stemmer.stem(w.lower()) for w in self.words if w not in stop})
        self.labels = sorted(self.labels)

        blank = [0] * len(self.labels)
        for doc, tag in zip(docs_x, docs_y):
            stemmed = [self.stemmer.stem(w.lower()) for w in doc]
            bag     = [1 if w in stemmed else 0 for w in self.words]
            row     = blank[:]
            row[self.labels.index(tag)] = 1
            self.training.append(bag)
            self.output.append(row)

        self.training = np.array(self.training)
        self.output   = np.array(self.output)

        with self.pickle_file.open("wb") as f:
            pickle.dump((self.words, self.labels, self.training, self.output), f)

    def _save_intents(self, notify_callback) -> bool:
        try:
            with self.intents_file.open("w") as f:
                json.dump(self.intents_data, f, indent=4)
            return True
        except Exception as exc:
            notify_callback(f"⚠️ Failed to save intents.json: {exc}", priority="normal")
            return False