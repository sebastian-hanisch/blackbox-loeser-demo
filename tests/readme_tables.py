"""Erzeugt die Zahlentabellen der README aus den vorgerechneten Bändern (`bbl_bands.json`) und den Konstanten der Vormessung.

`tests/test_claims.py` prüft, dass jede dieser Zeilen wörtlich in der README steht: die Zahlen der README sind damit nie von Hand abgeschrieben, und ändert sich eine Messung,
schlägt der Test an. Ausgabe zum Einfügen: `python tests/readme_tables.py`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bbl_bands as B  # noqa: E402
from bbl_evaluation import KEY_LABELS, de, de_g  # noqa: E402

# Messung der Vormessung (Wegwerf-Skripte, nicht in dieser Demo gerechnet): `entscheidungsorientiert-planung/vormessung/out_eo_v8c_bb_hamming.txt` (20 Seeds, Median)
# und `out_eo_v8_bias.txt`; Rauschstufe 0 / 0,5, normierter Regret in Prozent. Plattform: Windows, numpy-Zufall (nicht SplitMix64), 300 Adam-Schritte.
VORMESSUNG = {
    "kq": (1.87, 12.47),
    "bb_regret_0.3": (0.60, 18.06), "bb_regret_1": (0.54, 21.48), "bb_regret_10": (1.06, 21.03),
    "bb_hamming_0.3": (1.01, 20.93), "bb_hamming_1": (1.06, 21.15), "bb_hamming_10": (0.94, 21.26),
}
VORMESSUNG_BIAS = {  # (Rauschstufe 0, Rauschstufe 0,5), Median über 60 Merkmalsvektoren einer Welt
    "SPO+": (0.000, 0.049),
    "Fenchel-Young (σ = 0,3)": (0.710, 0.100),
    "Fenchel-Young (σ = 1)": (0.706, 0.114),
    "Interpolation (λ = 1)": (0.000, 0.715),
    "Interpolation (λ = 10)": (0.000, 0.700),
}
# Gegenprobe mit mehr Rauschziehungen (Messung dieser README, Welt Zufalls-Seed 0, Rauschstufe 0,5, 60 Merkmalsvektoren; Skript im Bau-Protokoll, der Test prüft die Zeilen 100 und 1600 als Bänder)
DRAWS = {100: (0.087, 0.140, 0.709), 400: (0.048, 0.122, 0.709), 1600: (0.023, 0.110, 0.708), 6400: (0.011, 0.111, 0.709)}


def _band(s: dict) -> str:
    return f"{de(s['median'], 2)} [{de(s['min'], 2)}; {de(s['max'], 2)}]"


def result_table(noise: float) -> list[str]:
    """Regret je Verfahren: Median [Minimum; Maximum] in Prozent, Zahl der Seeds, in denen das Verfahren streng besser bzw. streng schlechter als KQ ist."""
    cell = B.cell(noise)
    lines = ["| Verfahren | Regret (Median [Minimum; Maximum]) | besser als KQ | schlechter als KQ |", "|---|---|---|---|"]
    names = {"orakel": "Orakel (kennt die mittleren Kosten)", "kq": "Kleinste Quadrate (KQ)", "spo": "SPO+", **KEY_LABELS}
    for key in cell["summary"]:
        s = cell["summary"][key]
        if key in ("orakel", "kq"):
            lines.append(f"| {names[key]} | {_band(s)} % | – | – |")
        else:
            lines.append(f"| {names[key]} | {_band(s)} % | {B.count_better(noise, key)} von {B.n_seeds()} | {B.count_worse(noise, key)} von {B.n_seeds()} |")
    return lines


def curve_table(noise: float) -> list[str]:
    """Testregret nach t Adam-Schritten: Median über die Seeds in Prozent."""
    keys = [("spo", "SPO+"), ("bb_regret_1", "Interpolation λ = 1"), ("bb_regret_10", "Interpolation λ = 10"), ("bb_regret_100", "Interpolation λ = 100")]
    lines = ["| Schritte | " + " | ".join(name for _, name in keys) + " |", "|---|" + "---|" * len(keys)]
    curves = {key: B.curve_median(noise, key) for key, _ in keys}
    for step in B.own()["meta"]["snapshot_steps"]:
        lines.append(f"| {step} | " + " | ".join(f"{de(curves[key][step], 2)} %" for key, _ in keys) + " |")
    return lines


def bias_table() -> list[str]:
    lines = ["| Verfahren | Rauschstufe 0 (Median [Minimum; Maximum] über 20 Welten) | Rauschstufe 0,5 (Median [Minimum; Maximum] über 20 Welten) |", "|---|---|---|"]
    b0, b5 = B.bias(0.0)["summary"], B.bias(0.5)["summary"]
    for name in b0:
        f = lambda s: f"{de(s['median'], 3)} [{de(s['min'], 3)}; {de(s['max'], 3)}]"  # noqa: E731
        lines.append(f"| {name} | {f(b0[name])} | {f(b5[name])} |")
    return lines


def draws_table() -> list[str]:
    lines = ["| Rauschziehungen je Merkmalsvektor | SPO+ | Fenchel-Young (σ = 1) | Interpolation (λ = 1) |", "|---|---|---|---|"]
    for nd, (a, b, c) in DRAWS.items():
        lines.append(f"| {nd} | {de(a, 3)} | {de(b, 3)} | {de(c, 3)} |")
    return lines


def comparison_table() -> list[str]:
    """Diese Messung gegen die Vormessung (Median des Regrets in Prozent, Rauschstufe 0 / 0,5)."""
    lines = ["| Verfahren | Vormessung (Rauschstufe 0 / 0,5) | diese Demo (Rauschstufe 0 / 0,5) |", "|---|---|---|"]
    names = {"kq": "Kleinste Quadrate (KQ)", **KEY_LABELS}
    for key, (v0, v5) in VORMESSUNG.items():
        lines.append(f"| {names[key]} | {de(v0, 2)} / {de(v5, 2)} | {de(B.median(0.0, key), 2)} / {de(B.median(0.5, key), 2)} |")
    return lines


def effort_table() -> list[str]:
    s = B.own()["meta"]["gelöste_instanzen_300_schritte"]
    th = lambda v: f"{v:,}".replace(",", ".")  # noqa: E731
    return ["| Verfahren | gelöste Instanzen (n = 1000, 300 Schritte) |", "|---|---|", f"| SPO+ | {th(s['spo'])} |", f"| Interpolation | {th(s['bb_regret_1'])} |"]


def main():
    for title, lines in (
        ("Rauschstufe 0", result_table(0.0)), ("Rauschstufe 0,5", result_table(0.5)), ("Kurve 0", curve_table(0.0)), ("Kurve 0,5", curve_table(0.5)),
        ("Wahrheit", bias_table()), ("Ziehungen", draws_table()), ("Vergleich", comparison_table()), ("Aufwand", effort_table()),
    ):
        print(f"### {title}")
        print("\n".join(lines))
        print()


if __name__ == "__main__":
    main()
