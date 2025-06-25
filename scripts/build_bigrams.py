import pickle, os
from collections import Counter
import nltk

OUT_PATH = os.path.join("src", "data", "bigrams.pkl")
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "


def norm(text: str) -> str:
    return "".join(c if c in ALPHABET else " " for c in text.upper())


def main():
    from nltk.corpus import brown
    bigrams = Counter()
    for sent in brown.sents():
        t = norm(" ".join(sent))
        for i in range(len(t) - 1):
            bg = t[i: i + 2]
            if " " in bg:  # ignoramos bigramas con espacios
                continue
            bigrams[bg] += 1
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(dict(bigrams.most_common(3000)), f)
    print(f"Saved {len(bigrams)} bigrams → {OUT_PATH}")


if __name__ == "__main__":
    main()
