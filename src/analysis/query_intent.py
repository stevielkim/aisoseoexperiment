"""
Query intent classification and analysis.

This module provides comprehensive query intent detection including:
- Primary intent classification (informational, navigational, transactional, commercial)
- Question pattern detection (who, what, where, when, why, how)
- Query complexity scoring
- Local intent detection
- Multi-intent flagging

Based on industry-standard taxonomies from Google, Ahrefs, and SEMrush research.
"""

import re
from typing import Dict, List, Tuple


class QueryIntentClassifier:
    """Classify search query intent using pattern matching and heuristics."""

    def __init__(self):
        """Initialize classifier with intent patterns."""
        self.patterns = self._load_intent_patterns()

    def _load_intent_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        Load regex patterns for each intent type.

        Returns:
        --------
        dict : Intent type -> list of compiled regex patterns
        """
        return {
            'informational': [
                re.compile(r'^\s*(what|how|why|when|where|who)\s+(is|are|does|do|did|can|should|would)', re.I),
                re.compile(r'\b(guide|tutorial|learn|understand|explain|definition|meaning)\b', re.I),
                re.compile(r'^\s*(define|introduction to|overview of)\b', re.I),
            ],
            'navigational': [
                re.compile(r'\b(login|sign in|signin|log in|account)\b', re.I),
                re.compile(r'\b(official|website|homepage|site)\b', re.I),
                # Brand-specific patterns would go here (requires brand list)
            ],
            'transactional': [
                re.compile(r'\b(buy|purchase|order|shop|checkout)\b', re.I),
                re.compile(r'\b(price|cost|pricing|pay)\b', re.I),
                re.compile(r'\b(deal|discount|coupon|sale|promo)\b', re.I),
                re.compile(r'\b(download|get|install|subscribe|signup|register)\b', re.I),
            ],
            'commercial': [
                re.compile(r'^\s*(best|top)\b', re.I),
                re.compile(r'\b(review|reviews|rating|ratings)\b', re.I),
                re.compile(r'\b(compare|comparison|versus|vs)\b', re.I),
                re.compile(r'\b(affordable|cheap|budget|value)\b', re.I),
                re.compile(r'\b(pros and cons|worth it)\b', re.I),
            ],
            'local': [
                re.compile(r'\bnear me\b', re.I),
                re.compile(r'\bnearby\b', re.I),
                re.compile(r'\bopen now\b', re.I),
                re.compile(r'\bin [A-Z][a-z]+(?:,?\s+[A-Z]{2})?\b', re.I),  # in [City] or in [City, ST]
                re.compile(r'\b\d{5}\b'),  # ZIP codes
            ]
        }

    def classify(self, query: str) -> Dict:
        """
        Classify query intent comprehensively.

        Parameters:
        -----------
        query : str
            The search query to classify

        Returns:
        --------
        dict with keys:
            - primary_intent: str (informational, navigational, transactional, commercial, or other)
            - intent_scores: dict (scores for each intent type)
            - question_type: str (who, what, where, when, why, how, or none)
            - has_local_intent: bool
            - query_complexity: str (simple, moderate, complex)
            - query_word_count: int
            - starts_with_how: int (0/1)
            - starts_with_what: int (0/1)
            - starts_with_why: int (0/1)
            - starts_with_when: int (0/1)
            - starts_with_where: int (0/1)
            - starts_with_who: int (0/1)
            - contains_best: int (0/1)
            - contains_top: int (0/1)
            - contains_vs: int (0/1)
            - contains_review: int (0/1)
            - contains_how_to: int (0/1)
        """
        query_lower = query.lower().strip()

        # Calculate intent scores
        intent_scores = self._calculate_intent_scores(query_lower)

        # Determine primary intent
        primary_intent = max(intent_scores, key=intent_scores.get)
        if intent_scores[primary_intent] == 0:
            primary_intent = 'other'

        # Detect question pattern
        question_type = self.detect_question_pattern(query_lower)

        # Detect local intent
        has_local_intent = self.detect_local_intent(query_lower)

        # Calculate complexity
        complexity, word_count = self.calculate_complexity(query)

        # Binary feature flags
        starts_with_how = int(query_lower.startswith(('how to', 'how do', 'how does', 'how can', 'how should')))
        starts_with_what = int(query_lower.startswith(('what is', 'what are', 'what does', 'what do', 'what can')))
        starts_with_why = int(query_lower.startswith(('why is', 'why are', 'why does', 'why do', 'why did')))
        starts_with_when = int(query_lower.startswith(('when is', 'when are', 'when does', 'when do', 'when did')))
        starts_with_where = int(query_lower.startswith(('where is', 'where are', 'where does', 'where do', 'where can')))
        starts_with_who = int(query_lower.startswith(('who is', 'who are', 'who does', 'who did', 'who can')))

        contains_best = int('best ' in query_lower or query_lower.startswith('best'))
        contains_top = int('top ' in query_lower or query_lower.startswith('top'))
        contains_vs = int(' vs ' in query_lower or ' vs. ' in query_lower or ' versus ' in query_lower)
        contains_review = int('review' in query_lower)
        contains_how_to = int('how to' in query_lower or 'how-to' in query_lower)

        return {
            'primary_intent': primary_intent,
            'intent_confidence': intent_scores[primary_intent] if primary_intent != 'other' else 0.0,
            'question_type': question_type,
            'has_local_intent': int(has_local_intent),
            'query_complexity': complexity,
            'query_word_count': word_count,
            'starts_with_how': starts_with_how,
            'starts_with_what': starts_with_what,
            'starts_with_why': starts_with_why,
            'starts_with_when': starts_with_when,
            'starts_with_where': starts_with_where,
            'starts_with_who': starts_with_who,
            'contains_best': contains_best,
            'contains_top': contains_top,
            'contains_vs': contains_vs,
            'contains_review': contains_review,
            'contains_how_to': contains_how_to,
        }

    def _calculate_intent_scores(self, query: str) -> Dict[str, float]:
        """
        Calculate confidence scores for each intent type.

        Parameters:
        -----------
        query : str
            Lowercase query string

        Returns:
        --------
        dict : Intent type -> confidence score (0.0-1.0)
        """
        scores = {
            'informational': 0.0,
            'navigational': 0.0,
            'transactional': 0.0,
            'commercial': 0.0,
        }

        # Check each pattern and accumulate scores
        for intent_type, patterns in self.patterns.items():
            if intent_type == 'local':
                continue  # Handle local separately

            matches = sum(1 for pattern in patterns if pattern.search(query))
            # Normalize score: 1 match = 0.5, 2+ matches = 1.0
            scores[intent_type] = min(1.0, matches * 0.5)

        return scores

    def detect_question_pattern(self, query: str) -> str:
        """
        Detect question word at start of query.

        Parameters:
        -----------
        query : str
            Query string (preferably lowercase)

        Returns:
        --------
        str : Question type (who, what, where, when, why, how) or 'none'
        """
        question_patterns = {
            'who': r'^\s*who\b',
            'what': r'^\s*what\b',
            'where': r'^\s*where\b',
            'when': r'^\s*when\b',
            'why': r'^\s*why\b',
            'how': r'^\s*how\b',
        }

        for q_type, pattern in question_patterns.items():
            if re.match(pattern, query, re.I):
                return q_type

        return 'none'

    def calculate_complexity(self, query: str) -> Tuple[str, int]:
        """
        Score query complexity based on word count and structure.

        Parameters:
        -----------
        query : str
            Query string

        Returns:
        --------
        tuple : (complexity_label, word_count)
            complexity_label: 'simple', 'moderate', or 'complex'
            word_count: number of words in query
        """
        words = query.strip().split()
        word_count = len(words)

        if word_count <= 2:
            complexity = 'simple'
        elif word_count <= 5:
            complexity = 'moderate'
        else:
            complexity = 'complex'

        return complexity, word_count

    def detect_local_intent(self, query: str) -> bool:
        """
        Detect if query has local/geographic intent.

        Parameters:
        -----------
        query : str
            Query string (preferably lowercase)

        Returns:
        --------
        bool : True if query has local intent
        """
        local_patterns = self.patterns.get('local', [])
        return any(pattern.search(query) for pattern in local_patterns)

    def categorize_query(self, query: str) -> str:
        """
        Legacy categorization function for backward compatibility.
        Maps to the original 5-category system used in traditional_seo.py.

        Parameters:
        -----------
        query : str
            Query string

        Returns:
        --------
        str : 'How-to', 'Informational', 'Comparison', 'Best-of', or 'Other'
        """
        query_lower = query.lower().strip()

        # How-to queries
        if query_lower.startswith(('how to', 'how do', 'how does', 'how can')):
            return 'How-to'

        # Informational queries
        if query_lower.startswith(('what is', 'what are', 'what causes', 'what does', 'why is', 'why are')):
            return 'Informational'

        # Comparison queries
        if ' vs ' in query_lower or ' vs. ' in query_lower or ' versus ' in query_lower or 'comparison' in query_lower:
            return 'Comparison'

        # Best-of queries
        if query_lower.startswith(('best ', 'top ')) or re.search(r'^\d+\s+(best|top)', query_lower):
            return 'Best-of'

        return 'Other'


def extract_query_intent_features(query: str) -> Dict:
    """
    Extract all intent features from a query string.
    Convenience function for pipeline integration.

    Parameters:
    -----------
    query : str
        The search query

    Returns:
    --------
    dict : All intent features ready for CSV export
    """
    classifier = QueryIntentClassifier()
    features = classifier.classify(query)

    # Add legacy category for backward compatibility
    features['query_category'] = classifier.categorize_query(query)

    return features


# Example usage and validation
if __name__ == '__main__':
    classifier = QueryIntentClassifier()

    # Test queries
    test_queries = [
        "how to bake sourdough bread",
        "best laptops for students 2025",
        "what is seo",
        "google maps",
        "buy iphone 15 online",
        "pizza near me",
        "python vs java",
        "amazon prime login",
    ]

    print("Query Intent Classification Examples:\n")
    print("-" * 100)

    for query in test_queries:
        result = classifier.classify(query)
        print(f"\nQuery: '{query}'")
        print(f"  Primary Intent: {result['primary_intent']} (confidence: {result['intent_confidence']:.2f})")
        print(f"  Question Type: {result['question_type']}")
        print(f"  Complexity: {result['query_complexity']} ({result['query_word_count']} words)")
        print(f"  Local Intent: {bool(result['has_local_intent'])}")
        print(f"  Legacy Category: {classifier.categorize_query(query)}")
