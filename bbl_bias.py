"""Der Teilgradient an der Wahrheit (keine Streamlit-Abhängigkeit, nur numpy): ist er im Mittel null, wenn die Vorhersage genau die wahren bedingten Mittelkosten trifft?

Ein Verlustgradient, dessen Mittel an der Wahrheit nicht null ist, macht die Wahrheit zu keinem Ruhepunkt des Trainings, auch wenn das Modell sie schon hat. Je Merkmalsvektor x gibt es die wahren
Mittelkosten m(x) und viele Rauschziehungen c = m(x) * eps; die Vorhersage ist ch = m(x). Gemessen wird

    Verhältnis(x) = |Mittel über die Ziehungen des Gradienten| / Mittel über die Ziehungen von |Gradient|      (Normen über die Bögen)

0 heißt: im Mittel null (stationär); 1 heißt: alle Ziehungen zeigen in dieselbe Richtung. Die Zusammenfassung ist der Median des Verhältnisses über die Merkmalsvektoren.

Verglichen werden drei Gradienten (Minimierung, nach ch):

    SPO+           2 (w*(c) - w(2 ch - c))
    Fenchel-Young  w*(c) - w(ch + sigma z)  mit einer Gauß-Störung z je Fall (M = 1 Störung); aus dem Gedächtnis nach Berthet u. a. (Abstract-Ebene) nachgebaut, Vorzeichen am
                   rauschfreien Fall geprüft; kein Gegenstand dieser Demo, nur der Vergleich
    Interpolation  (w(ch + lam c) - w(ch)) / lam                       (Regret-Verlust)

Neu geschrieben für diese Demo nach der Messlogik der Vormessung `eo_v8_bias.py` (dort numpy-Zufall, hier SplitMix64 und eine Welt je Zufalls-Seed).
"""

from dataclasses import dataclass

import numpy as np

from bbl_blackbox import bb_gradient
from bbl_rng import SplitMix64, derive_seed
from bbl_world import TAG_TRUTH, WorldSpec, draw_weights, mean_costs

N_X, N_DRAW = 60, 400   # Merkmalsvektoren und Rauschziehungen je Vektor

# (Name, Art, Parameter) in der Reihenfolge der Anzeige
KINDS = (
    ("SPO+", "spo", {}),
    ("Fenchel-Young (σ = 0,3)", "fy", {"sigma": 0.3}),
    ("Fenchel-Young (σ = 1)", "fy", {"sigma": 1.0}),
    ("Interpolation (λ = 1)", "bb", {"lam": 1.0}),
    ("Interpolation (λ = 10)", "bb", {"lam": 10.0}),
)


@dataclass(frozen=True)
class TruthDraws:
    M: np.ndarray        # (nx, d) wahre Mittelkosten
    C: np.ndarray        # (nx, ndraw, d) beobachtete Kosten
    Z: np.ndarray        # (nx, ndraw, d) Standardnormalstörungen für Fenchel-Young


def draw_truth(spec: WorldSpec, d: int, seed: int, nx: int = N_X, ndraw: int = N_DRAW) -> TruthDraws:
    """Merkmalsvektoren, Rauschziehungen und Störungen der Welt zu einem Zufalls-Seed (getrennter Teilstrom TAG_TRUTH)."""
    B = draw_weights(d, spec, seed)
    rng = SplitMix64(derive_seed(seed, TAG_TRUTH))
    X = rng.normal(nx * spec.p).reshape(nx, spec.p)
    M = mean_costs(B, X, spec)
    if spec.noise > 0:
        eps = 1.0 - spec.noise + 2.0 * spec.noise * rng.uniform(nx * ndraw * d).reshape(nx, ndraw, d)
    else:
        eps = np.ones((nx, ndraw, d))
    Z = rng.normal(nx * ndraw * d).reshape(nx, ndraw, d)
    return TruthDraws(M, M[:, None, :] * eps, Z)


def gradients_at_truth(solver, draws: TruthDraws, art: str, **kw) -> np.ndarray:
    """Teilgradient der Art `art` ("spo", "fy", "bb") an der Vorhersage ch = m(x), je Rauschziehung: Form (nx, ndraw, d)."""
    nx, nd, d = draws.C.shape
    Chat = np.repeat(draws.M[:, None, :], nd, axis=1).reshape(-1, d)
    C = draws.C.reshape(-1, d)
    Wstar, _ = solver.solve(C)
    if art == "spo":
        W2, _ = solver.solve(2.0 * Chat - C)
        G = 2.0 * (Wstar - W2)
    elif art == "fy":
        W, _ = solver.solve(Chat + kw["sigma"] * draws.Z.reshape(-1, d))
        G = Wstar - W
    elif art == "bb":
        G = bb_gradient(solver, Chat, C, kw["lam"])
    else:
        raise ValueError(art)
    return G.reshape(nx, nd, d)


def relative_mean_gradient(G: np.ndarray) -> np.ndarray:
    """Verhältnis je Merkmalsvektor: |Mittel des Gradienten über die Ziehungen| / Mittel von |Gradient| (Form (nx,)); 0, wenn alle Gradienten null sind."""
    num = np.linalg.norm(G.mean(axis=1), axis=1)
    den = np.linalg.norm(G, axis=2).mean(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def bias_table(solver, spec: WorldSpec, seed: int, nx: int = N_X, ndraw: int = N_DRAW) -> dict:
    """Median des Verhältnisses über die Merkmalsvektoren für alle Gradienten in KINDS: {Name: Wert}."""
    draws = draw_truth(spec, solver.d, seed, nx, ndraw)
    return {name: float(np.median(relative_mean_gradient(gradients_at_truth(solver, draws, art, **kw)))) for name, art, kw in KINDS}
