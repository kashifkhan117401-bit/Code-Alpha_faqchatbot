import json
from dataclasses import dataclass
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_utils import clean_and_lemmatize


@dataclass
class MatchResult:
    question: str
    answer: str
    score: float


class FAQMatcher:

    def __init__(self, faqs_path: str = "faqs.json", confidence_threshold: float = 0.15):
        self.faqs_path = faqs_path
        self.confidence_threshold = confidence_threshold

        self.faqs: List[dict] = self._load_faqs(faqs_path)
        self.questions = [faq["question"] for faq in self.faqs]
        self.answers = [faq["answer"] for faq in self.faqs]

        
        self.cleaned_questions = [clean_and_lemmatize(q) for q in self.questions]


        self.cleaned_corpus = [
            f"{clean_and_lemmatize(q)} {clean_and_lemmatize(q)} {clean_and_lemmatize(a)}"
            for q, a in zip(self.questions, self.answers)
        ]


        self.vectorizer = TfidfVectorizer()
        self.faq_matrix = self.vectorizer.fit_transform(self.cleaned_corpus)

    @staticmethod
    def _load_faqs(path: str) -> List[dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise ValueError("faqs.json must contain a non-empty list of {question, answer} objects.")
        return data

    def best_match(self, user_query: str) -> Optional[MatchResult]:
      
        cleaned_query = clean_and_lemmatize(user_query)
        if not cleaned_query:
            return None

        query_vec = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vec, self.faq_matrix).flatten()

        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])

        if best_score < self.confidence_threshold:
            return None

        return MatchResult(
            question=self.questions[best_idx],
            answer=self.answers[best_idx],
            score=best_score,
        )

    def top_matches(self, user_query: str, k: int = 3) -> List[MatchResult]:
       
        cleaned_query = clean_and_lemmatize(user_query)
        if not cleaned_query:
            return []

        query_vec = self.vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vec, self.faq_matrix).flatten()

        ranked_idx = similarities.argsort()[::-1][:k]
        return [
            MatchResult(question=self.questions[i], answer=self.answers[i], score=float(similarities[i]))
            for i in ranked_idx
        ]

    def respond(self, user_query: str) -> str:
        """Convenience method: returns a ready-to-display chatbot reply string."""
        match = self.best_match(user_query)
        if match is None:
            return (
                "I'm sorry, I couldn't find a confident answer to that. "
                "Could you rephrase your question, or contact support@example.com for help?"
            )
        return match.answer


if __name__ == "__main__":
    matcher = FAQMatcher()
    test_queries = [
        "How do I return something I bought?",
        "when will my package arrive",
        "can I pay with paypal",
        "what's the meaning of life",
    ]
    for q in test_queries:
        result = matcher.best_match(q)
        print(f"Q: {q}")
        if result:
            print(f"  -> Matched FAQ: {result.question} (score={result.score:.3f})")
            print(f"  -> Answer: {result.answer}")
        else:
            print("  -> No confident match found.")
        print()
