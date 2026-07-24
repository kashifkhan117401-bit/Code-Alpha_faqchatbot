# FAQ Chatbot

A simple FAQ chatbot that matches a user's question to the closest FAQ using
NLP preprocessing (NLTK) and TF-IDF + cosine similarity (scikit-learn), with
a small Flask-based chat UI.

## How it works

1. **FAQ storage** — `faqs.json` holds a list of `{question, answer}` pairs.
   Replace this file with FAQs for your own product/topic; no code changes
   needed as long as the same structure is kept.

2. **Preprocessing** (`nlp_utils.py`) — for every piece of text (stored FAQ
   questions/answers and incoming user queries) we:
   - lowercase
   - strip punctuation and numbers
   - tokenize (NLTK `word_tokenize`)
   - remove English stopwords
   - lemmatize each token (NLTK `WordNetLemmatizer`)

3. **Matching** (`faq_matcher.py`) —
   - All FAQs are vectorized with a single `TfidfVectorizer` fitted over the
     cleaned corpus (question text is blended with its own answer so that
     keywords that only appear in an answer, like "PayPal", can still be
     matched to a related question).
   - An incoming query is cleaned the same way, vectorized, and compared to
     every FAQ vector with cosine similarity.
   - The highest-scoring FAQ is returned if its score clears a confidence
     threshold (default `0.15`); otherwise the bot returns a fallback
     "I don't know" message instead of guessing.

4. **UI** (`app.py` + `templates/index.html`) — a small Flask app serves a
   chat interface. Each bot reply shows a small confidence meter and which
   stored FAQ question it matched, so you can see the matching in action.

## Setup

```bash
pip install -r requirements.txt
python3 -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"
```

## Run the chat UI

```bash
python3 app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Run matching from the command line (no UI)

```bash
python3 faq_matcher.py
```

This runs a few example queries against the FAQ set and prints the matched
question, answer, and similarity score for each — useful for testing new
FAQ entries or tuning the confidence threshold.

## Using your own FAQs

Edit `faqs.json`:

```json
[
  { "question": "How do I reset my password?", "answer": "Click 'Forgot password' on the login screen..." },
  { "question": "What are your support hours?", "answer": "We're available 9am-6pm, Monday to Friday." }
]
```

No other changes are required — the vectorizer is refit automatically the
next time the app starts.

## Tuning match quality

- **Threshold**: lower `confidence_threshold` in `FAQMatcher(...)` (in
  `app.py` or `faq_matcher.py`) to make the bot answer more often (at the
  risk of wrong matches), or raise it to make the bot say "I don't know"
  more often (safer, but less helpful on paraphrased questions).
- **More FAQs, better matches**: TF-IDF matching improves as you add more
  FAQs with varied phrasing of the same topic, since it increases the
  chance a user's wording overlaps with something in the index.
- **Known limitation**: this is a lexical (word-overlap) method, so pure
  synonyms with no shared words (e.g. "get my money back" vs "return
  policy") may not match well. For stronger semantic matching, swap the
  TF-IDF vectorizer in `faq_matcher.py` for sentence embeddings (e.g.
  `sentence-transformers`) and use cosine similarity on those instead — the
  rest of the pipeline (Flask app, UI) stays the same.

## Files

| File | Purpose |
|---|---|
| `faqs.json` | FAQ data (question/answer pairs) |
| `nlp_utils.py` | Text cleaning: tokenize, remove stopwords, lemmatize |
| `faq_matcher.py` | TF-IDF vectorization + cosine similarity matching |
| `app.py` | Flask server (chat API + serves the UI) |
| `templates/index.html` | Chat UI |
| `requirements.txt` | Python dependencies |
