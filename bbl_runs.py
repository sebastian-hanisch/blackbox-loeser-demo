"""Läufe für die App und die Bänder (keine Streamlit-Abhängigkeit, nur numpy): eine Welt, Kleinste Quadrate (KQ), SPO+ und Black-Box-Interpolation mit verschiedenen lam und Verlusten.

Herkunft: Aufbau nach `spo-kuerzester-weg-demo/skw_runs.py` (Stück 3), für dieses Stück neu geschrieben (Verfahrensschlüssel statt fester Spalten). Zufall nur über SplitMix64 (`bbl_rng`).

Verfahrensschlüssel (`method_key`):
    "spo"                    SPO+ (Adam, Start bei KQ, Parameter mit kleinstem Trainingsverlust wie in Stück 3)
    "bb_regret_<lam>"        Interpolation mit dem Regret-Verlust (dL/dy = beobachtete Kosten), z. B. "bb_regret_10"
    "bb_regret_clean_<lam>"  dasselbe mit den mittleren Kosten m(x) statt der beobachteten (nur in der Simulation bekannt, Zusatzmessung)
    "bb_hamming_<lam>"       Interpolation mit dem Hamming-Abstand zum verrauschten Zielweg (bester Weg zu den beobachteten Kosten)
    "bb_hamming_clean_<lam>" dasselbe mit dem sauberen Zielweg (bester Weg zu den mittleren Kosten; nur in der Simulation bekannt, Zusatzmessung)
Alle Interpolations-Läufe nehmen den Parameter nach dem letzten Schritt (`select="last"`).
"""

from dataclasses import dataclass, replace

import numpy as np

from bbl_blackbox import train_bb
from bbl_grid import Grid
from bbl_losses import mse_per_arc, normalized_regret
from bbl_training import fit_least_squares, predict, train_spo_plus
from bbl_world import Dataset, WorldSpec, make_dataset

SNAPSHOT_STEPS = (0, 5, 10, 25, 50, 100, 200, 300)


def method_key(kind: str, lam: float | None = None, loss: str = "regret", target: str = "noisy") -> str:
    if kind == "spo":
        return "spo"
    return f"bb_{loss}_{'clean_' if target == 'clean' else ''}{lam:g}"


def parse_key(key: str) -> tuple[str, float | None, str, str]:
    """Umkehrung von `method_key`: (Art, lam, Verlust, Zielweg)."""
    if key == "spo":
        return "spo", None, "regret", "noisy"
    parts = key.split("_")
    if parts[0] != "bb" or parts[1] not in ("regret", "hamming"):
        raise ValueError(key)
    return "bb", float(parts[-1]), parts[1], "clean" if parts[2] == "clean" else "noisy"


@dataclass(frozen=True)
class RunSpec:
    """Alles, was einen Lauf festlegt."""

    deg: int = 6
    noise: float = 0.0
    n: int = 1000
    seed: int = 0
    p: int = 5
    h: int = 5
    steps: int = 300
    lr: float = 0.05
    n_test: int = 1000

    @property
    def world(self) -> WorldSpec:
        return WorldSpec(p=self.p, deg=self.deg, noise=self.noise)


class CountingGrid(Grid):
    """Wie `Grid`, zählt aber die gelösten Instanzen (Zeilen): für die Aufwandsangabe (zwei Löseraufrufe je Schritt und Fall bei der Interpolation)."""

    def __init__(self, h: int = 5):
        super().__init__(h)
        self.rows = 0

    def solve(self, C):
        self.rows += np.atleast_2d(C).shape[0]
        return super().solve(C)


@dataclass
class WorldRun:
    spec: RunSpec
    grid: Grid
    data: Dataset
    coef: dict           # Schlüssel -> Parameter ("kq" und die Verfahren)
    regret: dict         # Schlüssel -> normierter Regret auf den Testfällen ("orakel", "kq" und die Verfahren), Bruchteil
    mse: dict            # Schlüssel -> mittlerer quadratischer Fehler je Bogen auf den Testkosten ("kq" und die Verfahren)
    curve: dict          # Schlüssel -> {Schritt: Testregret nach so vielen Schritten}
    solved_rows: dict    # Schlüssel -> gelöste Instanzen im Training, einschließlich der einmaligen Lösung auf den beobachteten Kosten (n Fälle)


def train_method(grid: Grid, data: Dataset, key: str, steps: int, lr: float, coef_kq: np.ndarray, snapshot_steps=()):
    """Trainiert ein Verfahren und gibt das `TrainResult` zurück."""
    kind, lam, loss, target = parse_key(key)
    if kind == "spo":
        return train_spo_plus(grid, data.X_train, data.C_train, steps, lr, init=coef_kq, snapshot_steps=snapshot_steps)
    Ystar = dldy_costs = None
    if target == "clean":
        if loss == "hamming":
            Ystar = grid.solve(data.M_train)[0]
        else:
            dldy_costs = data.M_train
    return train_bb(grid, data.X_train, data.C_train, lam, loss, Ystar, steps, lr, init=coef_kq, snapshot_steps=snapshot_steps, dldy_costs=dldy_costs)


def run_world(spec: RunSpec, keys=("spo", "bb_regret_1"), snapshot_steps=SNAPSHOT_STEPS) -> WorldRun:
    grid = Grid(spec.h)
    data = make_dataset(spec.world, grid.d, spec.n, spec.n_test, spec.seed)
    coef_kq = fit_least_squares(data.X_train, data.C_train)
    Ct = data.C_test
    coef, curve, solved = {"kq": coef_kq}, {}, {}
    for key in keys:
        counter = CountingGrid(spec.h)
        res = train_method(counter, data, key, spec.steps, spec.lr, coef_kq, snapshot_steps)
        coef[key] = res.coef
        solved[key] = counter.rows
        curve[key] = {s: normalized_regret(grid, Ct, predict(c, data.X_test)) for s, c in res.snapshots.items()}
    chat = {k: predict(c, data.X_test) for k, c in coef.items()}
    chat["orakel"] = data.M_test
    regret = {k: normalized_regret(grid, Ct, v) for k, v in chat.items()}
    mse = {k: mse_per_arc(Ct, v) for k, v in chat.items() if k != "orakel"}
    return WorldRun(spec, grid, data, coef, regret, mse, curve, solved)


def cell_result(run: WorldRun) -> dict:
    """Ergebnisse eines Laufs: Regret in Prozent (Schlüssel wie in `regret`), Verlaufskurven in Prozent."""
    return {
        "seed": run.spec.seed,
        "regret": {k: 100.0 * v for k, v in run.regret.items()},
        "mse": dict(run.mse),
        "curve": {k: {str(s): 100.0 * v for s, v in c.items()} for k, c in run.curve.items()},
    }


def run_seeds(spec: RunSpec, keys, seeds, snapshot_steps=SNAPSHOT_STEPS) -> list[dict]:
    return [cell_result(run_world(replace(spec, seed=int(s)), keys, snapshot_steps)) for s in seeds]


def summarize(values) -> dict:
    """Median, Minimum, Maximum einer Liste."""
    a = np.asarray(list(values), dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}
