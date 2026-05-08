"""
strategies/models/news_classifier.py — Lightweight Heuristic News Classification

Provides sentiment and cluster labeling for financial news using keyword matching
and domain lexicon. The NewsFetcher uses RSS/Atom feeds to collect recent articles.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import xml.etree.ElementTree as ET
from typing import List, Optional

import requests


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
        def get_field(field_name: str) -> str:
            if isinstance(article, dict):
                return str(article.get(field_name, "") or "")
            return str(getattr(article, field_name, "") or "")

        title = get_field("title").lower()
        description = get_field("description").lower()
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
            title=get_field("title") or "Unknown Event"
        )


import hashlib
from datetime import datetime, timedelta

# Cluster importance weights for signal scoring
CLUSTER_WEIGHTS = {
    "Rate Decision":   2.0,
    "Inflation Data":  1.8,
    "Labor Market":    1.5,
    "GDP & Growth":    1.3,
    "Geopolitical":    1.2,
    "General Market":  1.0,
}


class NewsArticle:
    """Simple article container."""
    def __init__(self, title: str, description: str = "", url: str = ""):
        self.title       = title
        self.description = description
        self.url         = url
        self.id          = hashlib.md5(title.encode()).hexdigest()[:12]
        self.published   = datetime.now()


class NewsFetcher:
    """
    Lightweight RSS fetcher. Pulls financial headlines from free feeds.
    Falls back gracefully if network is unavailable.
    """

    RSS_FEEDS = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://rss.cnbc.com/rss/cnbc_world.xml",
    ]

    def __init__(self):
        self._cache: list[NewsArticle] = []
        self._last_fetch: datetime = datetime.min

    def fetch(self, max_age_hours: int = 2) -> list[NewsArticle]:
        """
        Returns cached articles if fresh, otherwise attempts RSS fetch.
        Silently returns empty list if network is unavailable.
        """
        now = datetime.now()
        if self._cache and (now - self._last_fetch) < timedelta(hours=max_age_hours):
            return self._cache

        articles = []
        for feed_url in self.RSS_FEEDS:
            try:
                articles.extend(self._parse_feed(feed_url))
            except Exception:
                continue  # Network unavailable — skip silently

        if articles:
            self._cache = articles
            self._last_fetch = now

        return self._cache

    def _parse_feed(self, url: str) -> list[NewsArticle]:
        """Parse an RSS feed URL into NewsArticle objects."""
        try:
            import urllib.request
            import xml.etree.ElementTree as ET

            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read()

            root = ET.fromstring(content)
            articles = []
            for item in root.iter("item"):
                title = item.findtext("title", "").strip()
                desc  = item.findtext("description", "").strip()
                link  = item.findtext("link", "").strip()
                if title:
                    articles.append(NewsArticle(title, desc, link))

            return articles[:20]  # limit per feed
        except Exception:
            return []
