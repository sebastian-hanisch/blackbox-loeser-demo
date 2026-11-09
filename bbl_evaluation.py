"""Kennzahlen, Verdikt-Logik und Tabellen für die App (keine Streamlit-Abhängigkeit): Handbeispiel, Lauf einer Welt, vorgerechnete Bänder.

Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`, `tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Werten.
Zahlen erscheinen mit Dezimalkomma (`de`)."""

from dataclasses import dataclass

import numpy as np

import bbl_bands as B
import bbl_constants as C
from bbl_blackbox import bb_parts, dldy_regret
from bbl_grid import two_paths
from bbl_runs import RunSpec


def _shown(value: float, digits: int = 1) -> float:
    """Der Wert so, wie er mit `digits` Stellen angezeigt wird (Vergleiche stimmen mit dem überein, was die Leser sehen)."""
    return float(f"{float(value):.{digits}f}")


def de(value: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma; negative Nullen (-0,0) erscheinen als 0,0."""
    text = f"{float(value):.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace(".", ",")


def de_g(value: float) -> str:
    """Kurze Zahl mit Dezimalkomma ohne überflüssige Nullen (für λ: 0,25 / 0,5 / 1 / 10)."""
    return f"{float(value):g}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    noise: float = C.DEFAULT_NOISE
    seed: int = C.DEFAULT_SEED
    lam: float = C.DEFAULT_LAMBDA_HAND
    c1: float = C.DEFAULT_C1HAT
    c2: float = C.DEFAULT_C2HAT

    def run_spec(self) -> RunSpec:
        return RunSpec(deg=C.DEG, noise=self.noise, n=C.N_TRAIN, seed=self.seed, p=C.N_FEATURES, h=C.GRID_H, steps=C.STEPS, lr=C.LR, n_test=C.N_TEST)


# --- Experiment 1: das Handbeispiel ------------------------------------------------------------------------------------------------------

def crossing_lambda(c1hat: float, c2hat: float) -> float | None:
    """Das λ, ab dem die Wahl mit ĉ_λ = ĉ + λ c vom Weg unter ĉ abweicht (Handbeispiel, c = (3, 5)).

    Wählt ĉ Weg 2 (ĉ₁ > ĉ₂), ist die Differenz der Kosten ĉ_λ,₁ − ĉ_λ,₂ = (ĉ₁ − ĉ₂) + λ (c₁ − c₂) = (ĉ₁ − ĉ₂) − 2 λ und wird bei λ* = (ĉ₁ − ĉ₂) / (c₂ − c₁) = (ĉ₁ − ĉ₂) / 2 null (Gleichstand, dort gilt die
    feste Regel: Weg 1). Wählt ĉ schon Weg 1 (ĉ₁ ≤ ĉ₂), bleibt die Differenz für jedes λ > 0 negativ: es gibt keine Kreuzung (None)."""
    if c1hat > c2hat:
        return (c1hat - c2hat) / (C.HAND_C[1] - C.HAND_C[0])
    return None


def hand_numbers(c1hat: float, c2hat: float, lam: float) -> dict:
    """Handbeispiel: zwei Wege mit je einem Bogen, wahre Kosten c = (3, 5), Vorhersage ĉ = (c1hat, c2hat), Regret-Verlust (dL/dy = c)."""
    solver = two_paths()
    c = np.array([C.HAND_C])
    chat = np.array([[c1hat, c2hat]])
    g, W1, W2 = bb_parts(solver, chat, dldy_regret(c), lam)
    chat_lam = chat + lam * c
    return {
        "lam": float(lam),
        "chosen": int(np.argmax(W1[0])) + 1,
        "chosen_lam": int(np.argmax(W2[0])) + 1,
        "best": int(np.argmax(solver.solve(c)[0][0])) + 1,
        "chat_lam": (float(chat_lam[0, 0]), float(chat_lam[0, 1])),
        "tie": bool(chat_lam[0, 0] == chat_lam[0, 1]),
        "changed": bool(np.any(W1 != W2)),
        "gradient": (float(g[0, 0]), float(g[0, 1])),
        "crossing": crossing_lambda(c1hat, c2hat),
        "regret": float(c[0] @ W1[0] - solver.solve(c)[1][0]),
    }


def hand_curve(c1hat: float, c2hat: float, points: int = C.LAMBDA_CURVE[2]) -> dict:
    """Erste Komponente des Gradienten gegen λ (logarithmisch gerastert, dazu die Werte aus `LAMBDA_HAND`)."""
    solver = two_paths()
    lo, hi, _ = C.LAMBDA_CURVE
    lams = np.unique(np.concatenate([np.geomspace(lo, hi, points), np.array(C.LAMBDA_HAND)]))
    c = np.array(C.HAND_C)
    chat = np.array([c1hat, c2hat])
    g1 = np.array([bb_parts(solver, chat[None, :], c[None, :], float(lam))[0][0, 0] for lam in lams])
    return {"lam": lams, "g1": g1}


def verdict_hand(h: dict) -> tuple[str, str]:
    """Verdikt des Handbeispiels aus den Zahlen von `hand_numbers`."""
    lam = de_g(h["lam"])
    g = f"({de(h['gradient'][0], 2)}; {de(h['gradient'][1], 2)})"
    if h["chosen"] == h["best"]:
        return "info", (
            f"Die Vorhersage wählt schon den richtigen Weg (Weg {h['best']}). Mit λ = {lam} ändert sich die Wahl auf ĉ_λ nicht, der Gradient ist {g}: nichts zu verbessern. "
            "Das gilt hier für jedes λ größer als 0 (ĉ_λ liegt noch weiter auf der Seite von Weg 1)."
        )
    if not h["changed"]:
        return "warning", (
            f"Mit λ = {lam} wählt auch ĉ_λ = ({de(h['chat_lam'][0], 2)}; {de(h['chat_lam'][1], 2)}) Weg {h['chosen_lam']}, die Wahl ändert sich nicht: der Gradient ist {g}, "
            f"obwohl Weg {h['chosen']} falsch ist. Der Gradient wird erst ab λ = {de_g(h['crossing'])} von null verschieden."
        )
    if h["tie"]:
        return "info", (
            f"Gleichstand: ĉ_λ = ({de(h['chat_lam'][0], 2)}; {de(h['chat_lam'][1], 2)}). Die feste Regel (kleinster Weg) wählt Weg {h['chosen_lam']}; der Gradient ist {g}. "
            "Bei kleinerem λ wäre der Gradient (0,00; 0,00), bei größerem 1/λ mal (1; −1): genau bei diesem λ springt er."
        )
    return "success", (
        f"Die Wahl kippt von Weg {h['chosen']} auf Weg {h['chosen_lam']}: Gradient {g} = (Lösung auf ĉ_λ − Lösung auf ĉ) / λ. "
        f"Je größer λ, desto kleiner der Gradient (hier 1/λ mal (1; −1)), und desto gröber die Interpolation."
    )


# --- Experiment 2: ohne und mit Rauschen --------------------------------------------------------------------------------------------------

LAM_KEYS = {lam: f"bb_regret_{lam:g}" for lam in C.LAMBDA_LIVE}


def verdict_noise(noise: float, kq: float, spo: float, bb: dict, worse: dict, seeds: int) -> tuple[str, str]:
    """Verdikt der Rauschfrage: Mediane des Regrets in Prozent (`bb`: λ -> Median) und Zahl der Seeds, in denen die Interpolation schlechter als KQ ist (`worse`: λ -> Zahl)."""
    lams = sorted(bb)
    bb_text = " / ".join(de(bb[lam]) for lam in lams)
    lam_text = " / ".join(de_g(lam) for lam in lams)
    scope = f"Median über {seeds} Zufalls-Seeds, Rauschstufe {de_g(noise)}"
    worse_text = " / ".join(str(worse[lam]) for lam in lams)
    if all(_shown(bb[lam]) > _shown(kq) for lam in lams):
        return "warning", (
            f"Mit Rauschen liegt die Interpolation bei λ = {lam_text} bei {bb_text} % gegen {de(kq)} % für Kleinste Quadrate (SPO+: {de(spo)} %; {scope}). "
            f"Sie ist in {worse_text} von {seeds} Seeds schlechter als KQ."
        )
    if all(_shown(bb[lam]) < _shown(kq) for lam in lams):
        return "success", (
            f"Ohne Rauschen liegt die Interpolation bei λ = {lam_text} bei {bb_text} % gegen {de(kq)} % für Kleinste Quadrate (SPO+: {de(spo)} %; {scope}); "
            f"in {worse_text} von {seeds} Seeds ist sie schlechter als KQ."
        )
    return "info", f"Gemischtes Bild: Interpolation bei λ = {lam_text}: {bb_text} % gegen {de(kq)} % für KQ und {de(spo)} % für SPO+ ({scope})."


def noise_rows(noise: float) -> list[dict]:
    """Zeilen der Tabelle von Experiment 2 aus den Bändern (eine Rauschstufe): Verfahren, Median, Spanne, Zahl der Seeds schlechter als KQ."""
    cell = B.cell(noise)["summary"]
    rows = [("Orakel (kennt die mittleren Kosten)", "orakel"), ("Kleinste Quadrate (KQ)", "kq"), ("SPO+", "spo")]
    rows += [(f"Interpolation, λ = {de_g(lam)}", key) for lam, key in LAM_KEYS.items()]
    out = []
    for label, key in rows:
        s = cell[key]
        out.append({"Verfahren": label, "key": key, "median": s["median"], "min": s["min"], "max": s["max"],
                    "schlechter": None if key in ("orakel", "kq") else B.count_worse(noise, key)})
    return out


# --- Experiment 3: der Teilgradient an der Wahrheit -----------------------------------------------------------------------------------------

def verdict_bias(noise: float, values: dict, seeds: int) -> tuple[str, str]:
    """Verdikt der Messung an der Wahrheit: `values` Name -> Median des Verhältnisses |Mittel| / Mittel von |·| (Median über `seeds` Welten)."""
    spo, bb1, bb10 = values["SPO+"], values["Interpolation (λ = 1)"], values["Interpolation (λ = 10)"]
    scope = f"Median über {seeds} Zufalls-Seeds, Rauschstufe {de_g(noise)}"
    if _shown(max(spo, bb1, bb10), 3) == 0:
        return "info", f"Ohne Rauschen sind an der Wahrheit SPO+ und die Interpolation beide genau null (Verhältnis {de(spo, 3)}; {scope}): die Wahrheit ist stationär."
    if _shown(min(bb1, bb10), 3) > _shown(spo, 3):
        return "warning", (
            f"An der Wahrheit bleibt der Mittelwert des Gradienten bei der Interpolation bei {de(bb1, 3)} (λ = 1) und {de(bb10, 3)} (λ = 10) seiner mittleren Größe, bei SPO+ bei {de(spo, 3)} "
            f"({scope}): die Interpolation ist hier nicht stationär."
        )
    return "info", f"An der Wahrheit: SPO+ {de(spo, 3)}, Interpolation {de(bb1, 3)} (λ = 1) und {de(bb10, 3)} (λ = 10) ({scope})."


# --- Experiment 4: Verlust und Zielweg ---------------------------------------------------------------------------------------------------------

NOISY_KEYS = ("bb_regret_0.3", "bb_regret_1", "bb_regret_10", "bb_hamming_0.3", "bb_hamming_1", "bb_hamming_10")
CLEAN_KEYS = ("bb_regret_clean_1", "bb_regret_clean_10", "bb_hamming_clean_0.3", "bb_hamming_clean_1", "bb_hamming_clean_10")

KEY_LABELS = {
    "bb_regret_0.3": "Regret-Verlust, λ = 0,3", "bb_regret_1": "Regret-Verlust, λ = 1", "bb_regret_10": "Regret-Verlust, λ = 10", "bb_regret_100": "Regret-Verlust, λ = 100",
    "bb_hamming_0.3": "Hamming-Abstand, λ = 0,3", "bb_hamming_1": "Hamming-Abstand, λ = 1", "bb_hamming_10": "Hamming-Abstand, λ = 10",
    "bb_regret_clean_1": "Regret-Verlust mit mittleren Kosten, λ = 1", "bb_regret_clean_10": "Regret-Verlust mit mittleren Kosten, λ = 10",
    "bb_hamming_clean_0.3": "Hamming-Abstand, sauberer Zielweg, λ = 0,3", "bb_hamming_clean_1": "Hamming-Abstand, sauberer Zielweg, λ = 1",
    "bb_hamming_clean_10": "Hamming-Abstand, sauberer Zielweg, λ = 10",
}


def loss_rows(noise: float) -> list[dict]:
    """Zeilen der Tabelle von Experiment 4 (eine Rauschstufe): KQ, die sechs Varianten mit verrauschtem Ziel, die fünf mit sauberem Ziel (Zusatzmessung)."""
    cell = B.cell(noise)["summary"]
    rows = [{"Verfahren": "Kleinste Quadrate (KQ)", "key": "kq", "gruppe": "KQ"}]
    rows += [{"Verfahren": KEY_LABELS[k], "key": k, "gruppe": "verrauscht"} for k in NOISY_KEYS]
    rows += [{"Verfahren": KEY_LABELS[k], "key": k, "gruppe": "sauber"} for k in CLEAN_KEYS]
    for r in rows:
        s = cell[r["key"]]
        r.update({"median": s["median"], "min": s["min"], "max": s["max"],
                  "schlechter": None if r["key"] == "kq" else B.count_worse(noise, r["key"])})
    return rows


def verdict_loss(noise: float, kq: float, noisy: list[float], clean: list[float], seeds: int) -> tuple[str, str]:
    """Verdikt der Verlustfrage: Mediane des Regrets in Prozent der sechs Varianten mit verrauschtem Ziel (`noisy`) und der fünf mit sauberem Ziel (`clean`), KQ als Bezug."""
    scope = f"Median über {seeds} Zufalls-Seeds, Rauschstufe {de_g(noise)}"
    a, b = de(min(noisy)), de(max(noisy))
    c, d = de(min(clean)), de(max(clean))
    if _shown(min(noisy)) > _shown(kq):
        return "warning", (
            f"Die Wahl des Verlusts ändert nichts: alle sechs Varianten mit verrauschtem Ziel liegen bei {a} bis {b} % gegen {de(kq)} % für KQ ({scope}). "
            f"Mit sauberem Ziel (nur in der Simulation bekannt) liegen die fünf Varianten bei {c} bis {d} %."
        )
    if _shown(max(noisy)) < _shown(kq):
        return "success", (
            f"Ohne Rauschen liegen alle sechs Varianten bei {a} bis {b} % gegen {de(kq)} % für KQ ({scope}); die Wahl des Verlusts ändert am Vorzeichen des Befunds nichts. "
            f"Mit sauberem Ziel sind es {c} bis {d} % (ohne Rauschen ist das Ziel von selbst sauber)."
        )
    return "info", f"Gemischtes Bild: verrauschtes Ziel {a} bis {b} %, sauberes Ziel {c} bis {d} %, KQ {de(kq)} % ({scope})."
