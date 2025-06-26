"""
vocab.py · Utilidades de vocabulario para la GA word-level
Autor: tú mismo :)
"""

from pathlib import Path
import random
import pickle
from typing import List, Dict

# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────
VOCAB_SIZE = 50_000  # cambia lo que necesites
_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_FILE = _CACHE_DIR / f"vocab_es_{VOCAB_SIZE}.pkl"

# Tokens especiales
PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<bos>", "<eos>"

# ─────────────────────────────────────────────────────────────
# Dependencia opcional: wordfreq
# ─────────────────────────────────────────────────────────────
try:
    from wordfreq import top_n_list  # noqa: F401
except ModuleNotFoundError:  # ⇠ IDE no mostrará “unresolved” después
    top_n_list = None  # sustituimos por marcador


# ─────────────────────────────────────────────────────────────
# Construcción / carga del vocabulario
# ─────────────────────────────────────────────────────────────
def _build_vocab() -> Dict[str, int]:
    """
    Descarga las VOCAB_SIZE palabras españolas más frecuentes
    (requiere `pip install wordfreq`). Si la librería no existe,
    lanza una excepción clara.
    """
    if top_n_list is None:
        raise RuntimeError(
            "El módulo 'wordfreq' no está instalado. "
            "Ejecuta `pip install wordfreq` e inténtalo de nuevo."
        )

    tokens = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN] + top_n_list("es", VOCAB_SIZE)
    return {tok: i for i, tok in enumerate(tokens)}


def _load_or_build_vocab() -> Dict[str, int]:
    if _CACHE_FILE.exists():
        with _CACHE_FILE.open("rb") as fh:
            vocab = pickle.load(fh)
    else:
        vocab = _build_vocab()
        with _CACHE_FILE.open("wb") as fh:
            pickle.dump(vocab, fh)
    return vocab


# Carga global (solo lectura tras esta línea)
TOKEN2IDX: Dict[str, int] = _load_or_build_vocab()
IDX2TOKEN: List[str] = [None] * len(TOKEN2IDX)
for tok, idx in TOKEN2IDX.items():
    IDX2TOKEN[idx] = tok

PAD_ID, UNK_ID, BOS_ID, EOS_ID = (
    TOKEN2IDX[PAD_TOKEN],
    TOKEN2IDX[UNK_TOKEN],
    TOKEN2IDX[BOS_TOKEN],
    TOKEN2IDX[EOS_TOKEN],
)


# ─────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────
def encode(text: str, add_bos_eos: bool = True) -> List[int]:
    tokens = text.lower().split()
    ids = [TOKEN2IDX.get(tok, UNK_ID) for tok in tokens]
    return ([BOS_ID] + ids + [EOS_ID]) if add_bos_eos else ids


def decode(ids: List[int], skip_special: bool = True) -> str:
    words = []
    for i in ids:
        tok = IDX2TOKEN[i] if i < len(IDX2TOKEN) else UNK_TOKEN
        if skip_special and tok in {PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN}:
            continue
        words.append(tok)
    return " ".join(words)


def random_sentence(max_len: int = 12) -> List[int]:
    length = random.randint(1, max_len)
    ids = [random.randrange(4, len(TOKEN2IDX)) for _ in range(length)]
    return [BOS_ID] + ids + [EOS_ID]


def vocab_size() -> int:
    return len(TOKEN2IDX)


# ─────────────────────────────────────────────────────────────
# Prueba rápida
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    s = "hola mundo cruel"
    ids = encode(s)
    print("ORIG   :", s)
    print("IDS    :", ids)
    print("DECODE :", decode(ids))
    print("RANDOM :", decode(random_sentence()))
