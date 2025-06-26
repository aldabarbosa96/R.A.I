"""
ga_word.py · Algoritmo Genético “word-level”
Autor:   <tu nombre>
Licencia: MIT

Requiere:
    pip install wordfreq colorama (opcional para color en Windows)

Descripción rápida
──────────────────
• Optimiza la “naturalidad” de frases en castellano sin objetivo explícito.
• Fitness = frecuencia Zipf (wordfreq) + heurísticos anti-repetición,
  bonus por longitud razonable, etc.
• Se apoya en el módulo local `vocab.py`, que mapea palabras ↔ ID.
"""

from __future__ import annotations

# ───── imports estándar ──────────────────────────────────────────────
from pathlib import Path
from typing import List, Tuple, Iterable
import random, time, csv, argparse, pickle, sys
from collections import Counter

# ───── terceros ──────────────────────────────────────────────────────
from wordfreq import zipf_frequency
try:
    from colorama import Fore, Style;  Fore.OK = Fore.GREEN
except Exception:                       # colorama no instalado
    class _Dummy:
        RESET = OK = RED = CYAN = ""
    Fore = Style = _Dummy()             # type: ignore

# ───── internos ──────────────────────────────────────────────────────
import vocab  # tu módulo recién creado

# ─────────────────────────────────────────────────────────────────────
# Parámetros por defecto (pueden sobre-escribirse al instanciar GAWord)
# ─────────────────────────────────────────────────────────────────────
DEF_POP_SIZE   = 300
DEF_MAX_GENS   = 10_000
NO_IMPROVE_LIMIT = 400

MUT_RATE = 0.025          # prob. de mutación por gen (≈ palabra)
ELITISM  = 2              # nº individuos que pasan sin cambios

# ─ Fitness ───────────────────────────────────────────────────────────
LEN_PEN_BASE   = 5        # Nº palabras a partir del cual empieza a penalizar
LEN_PEN_ALPHA  = 1.10     # Factor exponencial de castigo
WORD_BONUS     = 0.30     # Premio por palabra
WORD_BONUS_CAP = 6
DUP_PENALTY    = 1.00     # Castigo lineal por duplicados

# ─ Tipado ────────────────────────────────────────────────────────────
Genome = List[int]        # secuencia de IDs (incl. BOS/EOS)

# ═════════════════════════════════════════════════════════════════════
#   UTILIDADES DE GENOMA
# ═════════════════════════════════════════════════════════════════════
def _random_genome(max_len: int = 12) -> Genome:
    """Frase aleatoria ≈≤ max_len palabras (IDs)."""
    return vocab.random_sentence(max_len=max_len)


def _mutate(g: Genome) -> Genome:
    """Sustitución, inserción y borrado simples."""
    out = g[:]

    # sustitución
    for i in range(1, len(out) - 1):           # saltamos BOS/EOS
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
    """Cruzamiento 1-punto (en palabras)."""
    cut_a = random.randrange(1, len(a) - 1)
    cut_b = random.randrange(1, len(b) - 1)
    return a[:cut_a] + b[cut_b:]


# ═════════════════════════════════════════════════════════════════════
#   FUNCIÓN DE FITNESS
# ═════════════════════════════════════════════════════════════════════
def _fitness(g: Genome) -> float:
    """
    • Zipf de palabras *únicas*             (mayor ⇒ mejor)
    • Penalización suave de longitud
    • Bonus moderado por nº de palabras
    • Penalización lineal por repetición
    """
    words = vocab.decode(g, skip_special=True).split()
    if not words:
        return 0.0

    cnt = Counter(words)

    # 1) Frecuencia Zipf (sólo una vez por token distinto)
    fit = sum(zipf_frequency(w, "es") for w in cnt)

    # 2) Penalización por longitud excesiva
    n = len(words)
    if n > LEN_PEN_BASE:
        fit /= LEN_PEN_ALPHA ** (n - LEN_PEN_BASE)

    # 3) Bonus por nº de palabras
    fit += WORD_BONUS * min(n, WORD_BONUS_CAP)

    # 4) Penalización por duplicados
    dup = n - len(cnt)          # apariciones extra
    fit -= DUP_PENALTY * dup

    return fit


# ═════════════════════════════════════════════════════════════════════
#   CLASE PRINCIPAL GAWord
# ═════════════════════════════════════════════════════════════════════
class GAWord:
    """
    GAWord(pop_size=300, max_gens=10_000, ...)

    run() es un *generador*:
        for gen, fit, genome in GAWord(...).run():
            ...

    Cada iteración devuelve:
        • Nº de generación
        • Fitness del mejor individuo actual
        • El genoma (lista de IDs)
    """

    # ─── construcción ────────────────────────────────────────────────
    def __init__(self,
                 pop_size: int = DEF_POP_SIZE,
                 max_gens: int = DEF_MAX_GENS,
                 runs_dir: Path | str | None = None):

        # parámetros
        self.pop_size  = pop_size
        self.max_gens  = max_gens

        # población inicial
        self.pop  : List[Genome] = [_random_genome() for _ in range(pop_size)]
        self.fits : List[float]  = [_fitness(g)    for g in self.pop]

        # carpeta y CSV de métricas
        runs_root = Path(runs_dir or Path(__file__).resolve().parent.parent / "runs")
        runs_root.mkdir(exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        self.csv_path = runs_root / f"{ts}_run_word.csv"

        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["gen", "best_fit", "avg_fit", "diversity"])

        # para dump del mejor
        self.best_pickle = runs_root / f"{ts}_best_sentence.pkl"

    # ─── logging interno ─────────────────────────────────────────────
    def _log(self, gen: int, best_g: Genome, best_f: float) -> None:
        avg_f = sum(self.fits) / len(self.fits)
        diversity = len({tuple(g) for g in self.pop})
        sent = vocab.decode(best_g)

        col = Fore.OK if gen == 0 or best_f == max(self.fits) else ""
        print(f"{col}Gen {gen:<4d} • Mejor: {sent} (fit={best_f:,.3f}) "
              f"- Avg {avg_f:,.2f} - Div {diversity} - μ {MUT_RATE}{Style.RESET_ALL}")

        with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([gen, best_f, avg_f, diversity])

    # ─── bucle GA (generador) ────────────────────────────────────────
    def run(self) -> Iterable[Tuple[int, float, Genome]]:
        best_fit, best_genome, best_gen = -1.0, None, 0

        for gen in range(self.max_gens + 1):
            # registro & yield
            idx_best = max(range(self.pop_size), key=self.fits.__getitem__)
            cur_best_fit = self.fits[idx_best]
            cur_best_gen = self.pop[idx_best]

            if cur_best_fit > best_fit:
                best_fit, best_genome, best_gen = cur_best_fit, cur_best_gen, gen
                # guardamos dump cada vez que hay nuevo récord
                with self.best_pickle.open("wb") as fh:
                    pickle.dump(best_genome, fh)

            self._log(gen, cur_best_gen, cur_best_fit)
            yield gen, cur_best_fit, cur_best_gen

            # parada si no mejora
            if gen - best_gen >= NO_IMPROVE_LIMIT:
                print(f"{Fore.RED}🛑 Sin mejora en {NO_IMPROVE_LIMIT} generaciones. "
                      f"Paro en la {gen}.{Style.RESET_ALL}")
                break

            # selección (ruleta)
            total_fit = sum(self.fits)
            probs = [f / total_fit for f in self.fits]
            new_pop: List[Genome] = []

            # elitismo
            elites = sorted(zip(self.pop, self.fits),
                            key=lambda x: x[1],
                            reverse=True)[:ELITISM]
            new_pop.extend([e[0] for e in elites])

            # reproducción
            while len(new_pop) < self.pop_size:
                p1, p2 = random.choices(self.pop, weights=probs, k=2)
                child = _crossover(p1, p2)
                child = _mutate(child)
                new_pop.append(child)

            # nueva generación
            self.pop  = new_pop
            self.fits = [_fitness(g) for g in self.pop]


# ═════════════════════════════════════════════════════════════════════
#   CLI SENCILLA
# ═════════════════════════════════════════════════════════════════════
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m ga_word",
        description="Algoritmo Genético para generar frases en castellano.")

    p.add_argument("--pop-size",  type=int, default=DEF_POP_SIZE,
                   help=f"Población (def. {DEF_POP_SIZE})")
    p.add_argument("--max-gens",  type=int, default=DEF_MAX_GENS,
                   help=f"Nº máximo de generaciones (def. {DEF_MAX_GENS})")
    p.add_argument("--seed",      type=int, default=None,
                   help="Semilla RNG (para reproducibilidad)")

    return p.parse_args(argv)


def _main():
    args = _parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    ga = GAWord(pop_size=args.pop_size,
                max_gens=args.max_gens)

    # iteramos sin hacer nada extra; run() ya loguea en consola/CSV
    for _ in ga.run():
        pass

    # muestra el campeón final
    with ga.best_pickle.open("rb") as fh:
        best = pickle.load(fh)
    print(f"\n🏆  Mejor global →  «{vocab.decode(best)}»  "
          f"(fit={_fitness(best):.3f})")
    print(f"📦  Dump guardado en: {ga.best_pickle}")


if __name__ == "__main__":
    _main()
