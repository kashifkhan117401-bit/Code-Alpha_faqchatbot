

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def ensure_nltk_data():
    """Download required NLTK resources if they are not already present."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, pkg in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


ensure_nltk_data()

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_and_lemmatize(text: str) -> str:
   
    if not text:
        return ""

    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\d+", " ", text)          
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOPWORDS and t.isalpha()]
    tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


if __name__ == "__main__":
    sample = "What is YOUR return policy for damaged items??"
    print(f"Original : {sample}")
    print(f"Cleaned  : {clean_and_lemmatize(sample)}")
