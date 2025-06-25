# ================================================================
# FILE: src/scenarios/language_fluency.py
# ================================================================
import string, random, pickle
from pathlib import Path          # <-- ¡importa Path aquí!
from scenarios.base_scenario import Scenario


class LanguageFluencyScenario(Scenario):
    """
    Fitness = combinación de:
      • Frecuencia de bigramas sin espacios
      • Frecuencia de palabras reales (aún vacío → 0)
      • Bonus de estructura básica (mayúscula inicial, espacios, punto final)
      • Penalización por repeticiones (“AAAA”, “LL” …)
    """

    def __init__(self, length: int = 40,
                 bigram_file: str = "data/bigrams.pkl"):
        super().__init__(gene_length=length)
        self.charset = string.ascii_uppercase + " "

        # --- carga Bigramas -------------------------------------------------
        base_dir = Path(__file__).resolve().parent.parent     # .../src
        ngram_path = base_dir / bigram_file                   # .../src/data/bigrams.pkl
        if not ngram_path.exists():
            raise FileNotFoundError(f"Bigram file not found → {ngram_path}")

        with ngram_path.open("rb") as f:
            self.bigram_freq: dict[str, int] = pickle.load(f)

        # (opcional) diccionario de palabras → por ahora vacío
        self.ug_freq: dict[str, int] = {}

        # pesos (ajusta a tu gusto)
        self.W_BG = 1.0   # bigramas
        self.W_UG = 2.0   # unigramas / palabras
        self.W_STR = 5.0  # estructura
        self.W_REP = 1.0  # penalización repes

    # ---------------- Genes aleatorios ------------------------------------------------
    def random_genes(self):
        return [random.choice(self.charset) for _ in range(self.gene_length)]

    # ---------------- Evaluación ------------------------------------------------------
    def evaluate(self, genes):
        s = "".join(genes)

        # 1️⃣  bigramas sin espacios
        bg_score = sum(self.bigram_freq.get(s[i:i+2], 0)
                       for i in range(len(s)-1)
                       if " " not in s[i:i+2])

        # 2️⃣  palabras reales (aún 0 salvo que cargues un diccionario)
        words = s.split()
        ug_score = sum(self.ug_freq.get(w, 0) for w in words)

        # 3️⃣  estructura sencilla
        struct = 0
        if s and s[0].isalpha():            struct += 1
        if " " in s:                        struct += 1
        if s.strip().endswith((".", " ")):  struct += 1

        # 4️⃣  repeticiones consecutivas (AAA o LL, etc.)
        rep_pen = sum(1 for i in range(len(s)-2)
                      if s[i:i+2] == s[i+1:i+3] and " " not in s[i:i+2])

        return (self.W_BG * bg_score
                + self.W_UG * ug_score
                + self.W_STR * struct
                - self.W_REP * rep_pen)
