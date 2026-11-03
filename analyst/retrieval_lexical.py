"""Neutral lexical primitives; no routing, persistence, or model dependencies."""
from __future__ import annotations

from collections import Counter
import math
import re


def tokenize(text: str) -> list[str]:
    STOP_WORDS = {
        # Articles
        "a",
        "an",
        "the",
        # Pronouns
        "i",
        "me",
        "my",
        "myself",
        "you",
        "your",
        "yourself",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "we",
        "us",
        "our",
        "ours",
        "they",
        "them",
        "their",
        "theirs",
        "it",
        "its",
        # Auxiliary verbs and conversational filler
        "am",
        "are",
        "be",
        "been",
        "being",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "was",
        "were",
        "will",
        "would",
        "could",
        "should",
        "please",
        # Common question words
        "how",
        "what",
        "when",
        "where",
        "who",
        "why",
        # Common connectors
        "and",
        "but",
        "or",
        "if",
        "so",
        "the",
        "than",
        "then",
        # Common prepositions
        "at",
        "in",
        "of",
        "on",
        "to",
        # Generic request language
        "tell",
        "please",
        "really",
        "just",
    }
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOP_WORDS]



MEMORY_LEXICAL_CONTRACT = 'memory-bm25-ascii-identifiers-v1'


def memory_tokenize(text):
    """Keep existing terms/negation and also retain compound identifiers."""
    return tokenize(text) + re.findall(r'[a-z0-9]+(?:[-_.:/][a-z0-9]+)+', text.lower())


def bm25_scores(query_terms, documents, *, k1=1.5, b=0.75):
    """Score one already scoped corpus of term-frequency mappings."""
    n = len(documents)
    average = sum(sum(freq.values()) for freq in documents) / (n or 1)
    df = Counter(term for freq in documents for term in freq)
    result = []
    for freq in documents:
        length_factor = k1 * (1 - b + b * sum(freq.values()) / (average or 1))
        result.append(sum(math.log(1 + (n - df[t] + .5) / (df[t] + .5))
                          * freq[t] * (k1 + 1) / (freq[t] + length_factor)
                          for t in set(query_terms) if freq.get(t, 0)))
    return tuple(result)
