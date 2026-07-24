<div align="center">

# 💬 FAQ Chatbot

**A simple FAQ chatbot that matches a user's question to the closest FAQ**
using NLP preprocessing (NLTK) and TF‑IDF + cosine similarity (scikit‑learn),
with a small Flask‑based chat UI.

`Python` · `Flask` · `NLTK` · `scikit-learn`

</div>

---

## 📖 How It Works

<table>
<tr><td width="40" align="center"><b>1</b></td><td>

**FAQ storage**
`faqs.json` holds a list of `{question, answer}` pairs. Replace this file with FAQs for your own product/topic — no code changes needed as long as the same structure is kept.

</td></tr>
<tr><td align="center"><b>2</b></td><td>

**Preprocessing** — `nlp_utils.py`
For every piece of text (stored FAQ questions/answers *and* incoming user queries) we:

- lowercase
- strip punctuation and numbers
- tokenize (NLTK `word_tokenize`)
- remove English stopwords
- lemmatize each token (NLTK `WordNetLemmatizer`)

</td></tr>
<tr><td align="center"><b>3</b></td><td>

**Matching** — `faq_matcher.py`

- All FAQs are vectorized with a single `TfidfVectorizer` fitted over the cleaned corpus (question text is blended with its own answer so that keywords that only appear in an answer, like *"PayPal"*, can still be matched to a related question).
- An incoming query is cleaned the same way, vectorized, and compared to every FAQ vector with **cosine similarity**.
- The highest‑scoring FAQ is returned if its score clears a confidence threshold (default `0.15`); otherwise the bot returns a fallback *"I don't know"* message instead of guessing.

</td></tr>
<tr><td align="center"><b>4</b></td><td>

**UI** — `app.py` + `templates/index.html`
A small Flask app serves a chat interface. Each bot reply shows a small confidence meter and which stored FAQ question it matched, so you can see the matching in action.

</td></tr>
</table>

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
python3 -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"
```

## ▶️ Run the Chat UI

```bash
python3 app.py
```

> Open the printed local URL (e.g. `http://127.0.0.1:5000`) in your browser.

## 🖥️ Run Matching from the Command Line

*(no UI — useful for quickly testing new FAQ entries or tuning the threshold)*

```bash
python3 faq_matcher.py
```

This runs a few example queries against the FAQ set and prints the matched question, answer, and similarity score for each.

---

## 📝 Using Your Own FAQs

Edit `faqs.json`:

```json
[
  { "question": "How do I reset my password?", "answer": "Click 'Forgot password' on the login screen..." },
  { "question": "What are your support hours?", "answer": "We're available 9am-6pm, Monday to Friday." }
]
```

No other changes are required — the vectorizer is refit automatically the next time the app starts.

---

## 🎛️ Tuning Match Quality

| Lever | Effect |
|---|---|
| **Lower the threshold** | `confidence_threshold` in `FAQMatcher(...)` (in `app.py` or `faq_matcher.py`) — the bot answers more often, at the risk of wrong matches |
| **Raise the threshold** | The bot says *"I don't know"* more often — safer, but less helpful on paraphrased questions |
| **Add more FAQs** | TF‑IDF matching improves as you add more FAQs with varied phrasing of the same topic, since it increases the chance a user's wording overlaps with something in the index |

> **⚠️ Known limitation**
> This is a **lexical** (word‑overlap) method, so pure synonyms with no shared words (e.g. *"get my money back"* vs. *"return policy"*) may not match well. For stronger semantic matching, swap the TF‑IDF vectorizer in `faq_matcher.py` for sentence embeddings (e.g. `sentence-transformers`) and use cosine similarity on those instead — the rest of the pipeline (Flask app, UI) stays the same.

---

## 📂 Files

| File | Purpose |
|---|---|
| `faqs.json` | FAQ data (question/answer pairs) |
| `nlp_utils.py` | Text cleaning: tokenize, remove stopwords, lemmatize |
| `faq_matcher.py` | TF‑IDF vectorization + cosine similarity matching |
| `app.py` | Flask server (chat API + serves the UI) |
| `templates/index.html` | Chat UI |
| `requirements.txt` | Python dependencies |
