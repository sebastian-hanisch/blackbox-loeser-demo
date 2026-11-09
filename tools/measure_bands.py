"""Misst die Bänder, die die App als "vorgerechnet" zeigt, und schreibt `bbl_bands.json` (feste Zufalls-Seeds 0 bis 19, 300 Adam-Schritte, Welt Grad 6, n = 1000).

    python tools/measure_bands.py [--jobs 2] [--out bbl_bands.json]

Inhalt: (1) je Rauschstufe (0 und 0,5) und Zufalls-Seed ein Lauf mit KQ, SPO+ und der Interpolation: Regret-Verlust mit lam = 0,3 / 1 / 10 / 100, Hamming-Verlust mit lam = 0,3 / 1 / 10
(verrauschter Zielweg) und, als Zusatzmessung dieser Demo, mit sauberem Zielweg bzw. mit den mittleren Kosten als dL/dy (Regret-Verlust, lam = 1 / 10; Hamming, lam = 0,3 / 1 / 10);
Testregret nach 0, 5, 10, 25, 50, 100, 200, 300 Schritten und am Ende; (2) der Teilgradient an der Wahrheit (bbl_bias.py) je Zufalls-Seed und Rauschstufe.
Die Zahlen sind trainiert und plattformabhängig; Tests prüfen sie nur als Bänder und Ungleichungen. Nie mehr als zwei Prozesse (`--jobs`)."""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from bbl_bias import KINDS, N_DRAW, N_X, bias_table  # noqa: E402
from bbl_grid import Grid  # noqa: E402
from bbl_runs import SNAPSHOT_STEPS, RunSpec, cell_result, run_world  # noqa: E402
from bbl_world import WorldSpec  # noqa: E402

SEEDS = tuple(range(20))
NOISES = (0.0, 0.5)
KEYS = (
    "spo",
    "bb_regret_0.3", "bb_regret_1", "bb_regret_10", "bb_regret_100",
    "bb_hamming_0.3", "bb_hamming_1", "bb_hamming_10",
    "bb_regret_clean_1", "bb_regret_clean_10",
    "bb_hamming_clean_0.3", "bb_hamming_clean_1", "bb_hamming_clean_10",
)


def _run_cell(args):
    noise, seed = args
    return (noise, seed, cell_result(run_world(RunSpec(noise=noise, seed=seed), KEYS, SNAPSHOT_STEPS)))


def _bias_cell(args):
    noise, seed = args
    return (noise, seed, bias_table(Grid(5), WorldSpec(p=5, deg=6, noise=noise), seed))


def _summary(values):
    a = np.asarray(values, dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "bbl_bands.json"))
    args = ap.parse_args()
    t0 = time.time()
    cells = [(e, s) for e in NOISES for s in SEEDS]
    with Pool(args.jobs) as pool:
        runs = pool.map(_run_cell, cells, chunksize=2)
        bias = pool.map(_bias_cell, cells, chunksize=4)
    solved = run_world(RunSpec(noise=0.0, seed=0, n=1000), ("spo", "bb_regret_1"), snapshot_steps=()).solved_rows

    out = {
        "meta": {
            "beschreibung": "Eigene Messung dieser Demo (tools/measure_bands.py): lineares Modell, KQ, SPO+ und Black-Box-Interpolation (Adam, Lernrate 0,05, Start bei KQ, Interpolation: Parameter nach dem letzten Schritt), 5 x 5 Gitter, 5 Merkmale, Grad 6, 1000 Trainingsfälle, 1000 Testfälle, Zufalls-Seeds 0 bis 19.",
            "seeds": list(SEEDS), "steps": 300, "lr": 0.05, "n": 1000, "n_test": 1000, "snapshot_steps": list(SNAPSHOT_STEPS),
            "bias": {"n_x": N_X, "n_draw": N_DRAW, "arten": [name for name, _, _ in KINDS]},
            "einheiten": "Regret in Prozent der besten Wegkosten (Summe über die Testfälle); mse = mittlerer quadratischer Fehler je Bogen; Verlaufswerte: Testregret nach so vielen Schritten (Prozent)",
            "gelöste_instanzen_300_schritte": {k: int(v) for k, v in solved.items()},
            "laufzeit_s": round(time.time() - t0),
        },
        "cells": [],
        "bias": [],
    }
    for e in NOISES:
        rows = sorted([r for (ee, s, r) in runs if ee == e], key=lambda r: r["seed"])
        assert len(rows) == len(SEEDS)
        keys = ["orakel", "kq", *KEYS]
        regret = {k: [r["regret"][k] for r in rows] for k in keys}
        out["cells"].append({
            "noise": e,
            "regret": regret,
            "summary": {k: _summary(v) for k, v in regret.items()},
            "mse": {k: [r["mse"][k] for r in rows] for k in ("kq", *KEYS)},
            "curve": {k: {str(s): [r["curve"][k][str(s)] for r in rows] for s in SNAPSHOT_STEPS} for k in KEYS},
        })
        brows = sorted([(s, b) for (ee, s, b) in bias if ee == e])
        vals = {name: [b[name] for _, b in brows] for name, _, _ in KINDS}
        out["bias"].append({"noise": e, "values": vals, "summary": {k: _summary(v) for k, v in vals.items()}})
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {args.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
