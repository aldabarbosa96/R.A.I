#!/usr/bin/env python
"""
Genera unigramas y bigramas a nivel de palabra para español.

Uso:
    python -m scripts.build_es_word_ngrams  \
           --out-dir src/data               \
           data/corpus1.txt data/corpus2.txt ...

Si no pasa ficheros, lee STDIN.
"""
import argparse, pickle, re, sys
from collections import Counter
from pathlib import Path
from itertools import tee
from tqdm import tqdm   # pip install tqdm

TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3) ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def iter_sentence_tokens(path):
    with path.open(encoding="utf8") as fh:
        for line in fh:
            toks = TOKEN_RE.findall(line.lower())
            if toks:
                yield toks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path,
                    help="Archivos de texto plano en español")
    ap.add_argument("--min-count", type=int, default=1,
                    help="frecuencia mínima a guardar")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if not args.files:
        args.files = [Path("-")]      # leer STDIN

    unigrams, bigrams = Counter(), Counter()

    for fp in args.files:
        it = iter_sentence_tokens(fp) if fp != Path("-") \
            else (TOKEN_RE.findall(l.lower()) for l in sys.stdin)
        for sent in tqdm(it, desc=f"↻ {fp}"):
            unigrams.update(sent)
            bigrams.update(pairwise(sent))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "es_unigrams.pkl").open("wb") as f:
        pickle.dump({w:c for w,c in unigrams.items() if c>=args.min_count}, f)
    with (out_dir / "es_bigrams.pkl").open("wb") as f:
        pickle.dump({bg:c for bg,c in bigrams.items() if c>=args.min_count}, f)

    print(f"📦 Guardados  {len(unigrams):,} unigramas  "
          f"y  {len(bigrams):,} bigramas  →  {out_dir}")

if __name__ == "__main__":
    main()
