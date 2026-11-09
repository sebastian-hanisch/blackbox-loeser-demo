"""Vorgerechnete Bänder (keine Streamlit-Abhängigkeit): `bbl_bands.json`, eigene Messung dieser Demo (`tools/measure_bands.py`, Zufalls-Seeds 0 bis 19)."""

import json
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def own() -> dict:
    return json.loads((HERE / "bbl_bands.json").read_text(encoding="utf-8"))


def cell(noise: float) -> dict:
    """Die Zelle einer Rauschstufe: Regret je Seed und Verfahren (Prozent), Zusammenfassung, Verlaufskurven."""
    for c in own()["cells"]:
        if c["noise"] == noise:
            return c
    raise KeyError(noise)


def median(noise: float, key: str) -> float:
    return cell(noise)["summary"][key]["median"]


def regrets(noise: float, key: str) -> list[float]:
    return cell(noise)["regret"][key]


def n_seeds() -> int:
    return len(own()["meta"]["seeds"])


def count_worse(noise: float, key: str, than: str = "kq") -> int:
    """In wie vielen Seeds ist `key` im Regret schlechter (streng größer) als `than`?"""
    a, b = regrets(noise, key), regrets(noise, than)
    return int(sum(x > y for x, y in zip(a, b)))


def count_better(noise: float, key: str, than: str = "kq") -> int:
    a, b = regrets(noise, key), regrets(noise, than)
    return int(sum(x < y for x, y in zip(a, b)))


def curve_median(noise: float, key: str) -> dict:
    """Median des Testregrets (Prozent) je Schrittzahl über die Seeds: {Schritt: Wert}."""
    import numpy as np

    return {int(s): float(np.median(v)) for s, v in cell(noise)["curve"][key].items()}


def bias(noise: float) -> dict:
    for b in own()["bias"]:
        if b["noise"] == noise:
            return b
    raise KeyError(noise)
