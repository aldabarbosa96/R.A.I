import string
import random
import pickle
from pathlib import Path
from .base_scenario import Scenario


class LanguageFluencyScenario(Scenario):
    """
    Fitness = combinación de:
      • Frecuencia de bigramas sin espacios
      • Frecuencia de palabras reales (aún vacío → 0)
      • Bonus de estructura básica (mayúscula inicial, espacios, punto final)
      • Penalización por repeticiones (“AAAA”, “LL” …)
    """

    def __init__(
            self,
            length: int = 40,
            bigram_file: str = "data/processed/bigrams.pkl"
    ):
        super().__init__(gene_length=length)
        self.charset = string.ascii_uppercase + " "

        # --- carga Bigramas -------------------------------------------------
        # partir de la raíz del proyecto (dos niveles arriba de este archivo)
        project_root = Path(__file__).resolve().parents[2]
        ngram_path = project_root / bigram_file

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


    # ---------------- Genes aleatorios ----------------------------
    def random_genes(self):
        return [random.choice(self.charset) for _ in range(self.gene_length)]

    # ---------------- Evaluación ----------------------------------
    def evaluate(self, genes):
        s = "".join(genes)

        # 1️⃣ Frecuencia de bigramas sin espacios
        bg_score = sum(
            self.bigram_freq.get(s[i:i + 2], 0)
            for i in range(len(s) - 1)
            if " " not in s[i:i + 2]
        )

        # 2️⃣ Frecuencia de palabras reales
        words = s.split()
        ug_score = sum(self.ug_freq.get(w, 0) for w in words)

        # 3️⃣ Estructura sencilla
        struct = 0
        if s and s[0].isalpha():           struct += 1
        if " " in s:                       struct += 1
        if s.strip().endswith((".", " ")): struct += 1

        # 4️⃣ Penalización por repeticiones
        rep_pen = sum(
            1 for i in range(len(s) - 2)
            if s[i:i + 2] == s[i + 1:i + 3] and " " not in s[i:i + 2]
        )

        return (
                self.W_BG * bg_score +
                self.W_UG * ug_score +
                self.W_STR * struct -
                self.W_REP * rep_pen
        )
