"""
ga_word.py · Algoritmo Genético “word-level”
Autor: tú mismo :)
Requiere:  pip install wordfreq
"""

from __future__ import annotations
from pathlib import Path
from typing  import List, Tuple, Iterable
import random, time, csv

import vocab                               # tu módulo recién creado
from wordfreq import zipf_frequency        # frecuencia Zipf para el fitness


# ───────────────────────────────────────────
# Parámetros por defecto (pueden sobre-escribirse vía __init__)
# ───────────────────────────────────────────
DEF_POP_SIZE        = 300
DEF_MAX_GENS        = 10_000
NO_IMPROVE_LIMIT    = 400
MUT_RATE            = 0.025
ELITISM             = 2                   # nº individuos que pasan tal cual


# ───────────────────────────────────────────
# Tipos y utilidades de genoma
# ───────────────────────────────────────────
Genome = List[int]                         # secuencia de IDs (incl. BOS/EOS)


def _random_genome(max_len: int = 12) -> Genome:
    return vocab.random_sentence(max_len=max_len)


def _mutate(g: Genome) -> Genome:
    """Sustitución, inserción y borrado simples."""
    out = g[:]

    # sustitución
    for i in range(1, len(out) - 1):                     # saltamos BOS/EOS
        if random.random() < MUT_RATE:
            out[i] = random.randrange(4, vocab.vocab_size())

    # inserción
    if random.random() < MUT_RATE:
        idx = random.randrange(1, len(out) - 1)
        out.insert(idx, random.randrange(4, vocab.vocab_size()))

    # borrado
    if len(out) > 5 and random.random() < MUT_RATE:
        idx = random.randrange(1, len(out) - 1)
        out.pop(idx)

    return out


def _crossover(a: Genome, b: Genome) -> Genome:
    """1-punto (en palabras)."""
    cut_a = random.randrange(1, len(a) - 1)
    cut_b = random.randrange(1, len(b) - 1)
    return a[:cut_a] + b[cut_b:]


def _fitness(g: Genome) -> float:
    """Suma de frecuencias Zipf ⇒ recompensa vocabulario + común."""
    words = vocab.decode(g, skip_special=True).split()
    if not words:
        return 0.0
    return sum(zipf_frequency(w, "es") for w in words)


# ───────────────────────────────────────────
# Algoritmo principal
# ───────────────────────────────────────────
class GAWord:
    """
    GAWord(pop_size=300, max_gens=10_000)

    * No necesita “target” (optimiza naturalidad via Zipf).
    * run() es un *generador* que va emitiendo (gen, best_fit, best_genome).
    """

    # ── construcción ──────────────────────────────────────────────────────
    def __init__(self,
                 pop_size:      int = DEF_POP_SIZE,
                 max_gens:      int = DEF_MAX_GENS,
                 runs_dir:      Path | str | None = None):

        # parámetros
        self.pop_size     = pop_size
        self.max_gens     = max_gens

        # población inicial
        self.pop  : List[Genome] = [_random_genome() for _ in range(pop_size)]
        self.fits : List[float]  = [_fitness(g) for g in self.pop]

        # carpeta y CSV de métricas
        runs_root = Path(runs_dir or Path(__file__).resolve().parent.parent / "runs")
        runs_root.mkdir(exist_ok=True)
        ts        = time.strftime("%Y-%m-%dT%H-%M-%S")
        self.csv_path = runs_root / f"{ts}_run_wordlevel.csv"

        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["gen", "best_fit", "avg_fit", "diversity"])

    # ── logging interno ───────────────────────────────────────────────────
    def _log(self, gen: int, best_g: Genome, best_f: float) -> None:
        avg_f      = sum(self.fits) / len(self.fits)
        diversity  = len({tuple(g) for g in self.pop})
        sent       = vocab.decode(best_g)

        print(f"Gen {gen:<5d} • Mejor: {sent} (fit={best_f:,.3f}) "
              f"- Avg {avg_f:,.2f} - Div {diversity} - μ {MUT_RATE}")

        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([gen, best_f, avg_f, diversity])

    # ── bucle GA (generador) ──────────────────────────────────────────────
    def run(self) -> Iterable[Tuple[int, float, Genome]]:
        best_fit, best_gen = -1.0, 0

        for gen in range(self.max_gens + 1):
            # — registro & yield —
            cur_best_idx  = max(range(self.pop_size), key=self.fits.__getitem__)
            cur_best_fit  = self.fits[cur_best_idx]
            cur_best_gen  = self.pop[cur_best_idx]

            if cur_best_fit > best_fit:
                best_fit, best_gen = cur_best_fit, gen

            self._log(gen, cur_best_gen, cur_best_fit)
            yield gen, cur_best_fit, cur_best_gen

            # — parada si no mejora —
            if gen - best_gen >= NO_IMPROVE_LIMIT:
                print(f"🛑 Sin mejora en {NO_IMPROVE_LIMIT} generaciones. "
                      f"Paro en la {gen}.")
                break

            # — selección por ruleta —
            total_fit = sum(self.fits)
            probs     = [f / total_fit for f in self.fits]
            new_pop   : List[Genome] = []

            # elitismo
            elites = sorted(zip(self.pop, self.fits),
                            key=lambda x: x[1],
                            reverse=True)[:ELITISM]
            new_pop.extend([e[0] for e in elites])

            # reproducción hasta llenar población
            while len(new_pop) < self.pop_size:
                p1, p2 = random.choices(self.pop, weights=probs, k=2)
                child  = _crossover(p1, p2)
                child  = _mutate(child)
                new_pop.append(child)

            # nueva generación
            self.pop  = new_pop
            self.fits = [_fitness(g) for g in self.pop]
