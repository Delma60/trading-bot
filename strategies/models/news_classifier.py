"""
strategies/models/news_classifier.py — Lightweight Heuristic News Classification

Provides sentiment and cluster labeling for financial news using keyword matching
and domain lexicon. No external API dependencies, works offline.
"""

from dataclasses import dataclass


@dataclass
class NewsClassification:
    """Output structure for classified news."""
    sentiment: str        # BULLISH, BEARISH, NEUTRAL
    confidence: float     # 0.0 to 1.0
    cluster_label: str    # "Rate Decision", "Inflation", etc.
    is_fake: bool         # False (heuristic classifier can't detect fakes)
    relevance: float      # 0.0 to 1.0
    title: str = ""


class NewsClassifier:
    """
    Heuristic-based NLP classifier for financial news. 
    Assigns sentiment and cluster labels based on hawkish/dovish market lexicon.
    
    No external dependencies, works entirely offline. Perfect for use in 
    NewsTradingStrategy without needing heavy ML libraries or API keys.
    """
    
    BULLISH_TERMS = [
        "hike", "growth", "beat", "surges", "hawkish", "expansion", "upbeat", 
        "record high", "strong", "surge", "jump", "bull", "rally", "bounce"
    ]
    BEARISH_TERMS = [
        "cut", "contraction", "miss", "plunges", "dovish", "recession", "downbeat", 
        "slump", "weak", "fall", "crash", "bear", "selloff", "decline"
    ]
    
    CLUSTERS = {
        "Rate Decision": ["rate", "fed", "ecb", "hike", "cut", "bps", "powell", "lagarde", "interest"],
        "Inflation Data": ["cpi", "pce", "inflation", "ppi", "prices"],
        "Labor Market": ["nfp", "payrolls", "unemployment", "jobless", "employment"],
        "GDP & Growth": ["gdp", "growth", "contraction", "recession", "expansion"],
        "Geopolitical": ["war", "tensions", "conflict", "sanctions", "tariffs"],
    }

    def classify(self, article: dict, symbol: str = None) -> NewsClassification:
        """
        Classify a news article as bullish/bearish and assign a cluster label.
        
        Parameters
        ----------
        article : dict
            Dictionary with 'title' and 'description' keys
        symbol : str
            Optional. If provided, boost relevance if symbol appears in text.
            
        Returns
        -------
        NewsClassification
            Sentiment, confidence, cluster label, and relevance score.
        """
        title = str(article.get("title", "")).lower()
        description = str(article.get("description", "")).lower()
        text = title + " " + description
        
        # Count sentiment indicators
        bull_score = sum(1 for term in self.BULLISH_TERMS if term in text)
        bear_score = sum(1 for term in self.BEARISH_TERMS if term in text)
        
        # Determine Sentiment
        if bull_score > bear_score:
            sentiment = "BULLISH"
            conf = min(0.4 + (bull_score * 0.15), 0.95)
        elif bear_score > bull_score:
            sentiment = "BEARISH"
            conf = min(0.4 + (bear_score * 0.15), 0.95)
        else:
            sentiment = "NEUTRAL"
            conf = 0.2

        # Determine Cluster
        assigned_cluster = "General Market"
        for cluster, keywords in self.CLUSTERS.items():
            if any(kw in text for kw in keywords):
                assigned_cluster = cluster
                break
                
        # Relevance based on symbol mention
        relevance = 0.5
        if symbol:
            base, quote = symbol[:3].lower(), symbol[3:6].lower()
            if base in text or quote in text:
                relevance = 0.9

        return NewsClassification(
            sentiment=sentiment,
            confidence=round(conf, 2),
            cluster_label=assigned_cluster,
            is_fake=False,
            relevance=round(relevance, 2),
            title=article.get("title", "Unknown Event")
        )
