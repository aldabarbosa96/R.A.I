import os, pickle, ssl, nltk

ssl._create_default_https_context = ssl._create_unverified_context

from collections import Counter

OUT = "data/processed/unigrams.pkl"
words = [w.upper() for w in nltk.corpus.brown.words()]
unigrams = Counter(w for w in words if w.isalpha() and len(w) > 1)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "wb") as f:
    pickle.dump(dict(unigrams), f)

print(f"Saved {len(unigrams)} words  →  {OUT}")
