import pandas as pd
from strategies.models.news_classifier import NewsFetcher, NewsClassifier, CLUSTER_WEIGHTS

class NewsTradingStrategy:
    """
    Consumes the ML/DL News Classifier pipeline to generate trading signals.
    """
    def __init__(self):
        self.fetcher = NewsFetcher()
        self.classifier = NewsClassifier()
        # Track processed articles to avoid redundant signal generation
        self.processed_ids = set()
        self.last_article_id = None  # Track the article that led to the last signal

    def analyze(self, df: pd.DataFrame, symbol: str = "EURUSD") -> dict:
        """
        Fetches the latest news, runs it through the ML pipeline, 
        and generates a directional trading signal.
        """
        # 1. Pull recent news (cached internally to avoid spamming RSS feeds)
        articles = self.fetcher.fetch(max_age_hours=2)
        
        if not articles:
            return {"action": "WAIT", "confidence": 0.0, "reason": "No recent news."}

        bullish_score = 0.0
        bearish_score = 0.0
        top_reasons = []
        top_article_id = None
        max_weight = 0.0

        # 2. Process unseen articles
        for article in articles:
            if article.id in self.processed_ids:
                continue

            # Run through the deep learning & unsupervised ML pipeline
            classified = self.classifier.classify(article, symbol=symbol)
            
            # 3. Filter out Fake / Low Relevance News
            if classified.is_fake or classified.relevance < 0.20:
                self.processed_ids.add(article.id)
                continue

            # 4. Aggregate Impact & Sentiment
            impact_multiplier = CLUSTER_WEIGHTS.get(classified.cluster_label, 1.0)
            signal_weight = classified.confidence * impact_multiplier

            if classified.sentiment == "BULLISH":
                bullish_score += signal_weight
                top_reasons.append(f"Bullish {classified.cluster_label}: {classified.title}")
                if signal_weight > max_weight:
                    max_weight = signal_weight
                    top_article_id = article.id
            elif classified.sentiment == "BEARISH":
                bearish_score += signal_weight
                top_reasons.append(f"Bearish {classified.cluster_label}: {classified.title}")
                if signal_weight > max_weight:
                    max_weight = signal_weight
                    top_article_id = article.id
            
            self.processed_ids.add(article.id)

        # Set last article id for feedback
        self.last_article_id = top_article_id

        # 5. Determine Final Action
        total_score = bullish_score + bearish_score
        
        if total_score == 0:
            return {"action": "WAIT", "confidence": 0.0, "reason": "News is neutral or irrelevant."}

        # Calculate a normalized confidence score bounded between 0.5 and 0.95
        normalized_confidence = min(0.95, 0.5 + (max(bullish_score, bearish_score) / (total_score + 1e-5)) * 0.45)

        if bullish_score > bearish_score * 1.5:
            return {
                "action": "BUY",
                "confidence": round(normalized_confidence, 2),
                "reason": " | ".join(top_reasons[:2]),
                "article_id": self.last_article_id
            }
        elif bearish_score > bullish_score * 1.5:
            return {
                "action": "SELL",
                "confidence": round(normalized_confidence, 2),
                "reason": " | ".join(top_reasons[:2]),
                "article_id": self.last_article_id
            }

        return {"action": "WAIT", "confidence": 0.0, "reason": "Conflicting news sentiment."}