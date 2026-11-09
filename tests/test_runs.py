"""Läufe (bbl_runs): Verfahrensschlüssel, Zählung der gelösten Instanzen von Hand, Einheiten, Reproduzierbarkeit."""

import numpy as np
import pytest

import bbl_runs as R


@pytest.mark.parametrize("key,parts", [
    ("spo", ("spo", None, "regret", "noisy")),
    ("bb_regret_1", ("bb", 1.0, "regret", "noisy")),
    ("bb_regret_0.3", ("bb", 0.3, "regret", "noisy")),
    ("bb_regret_clean_10", ("bb", 10.0, "regret", "clean")),
    ("bb_hamming_100", ("bb", 100.0, "hamming", "noisy")),
    ("bb_hamming_clean_0.3", ("bb", 0.3, "hamming", "clean")),
])
def test_method_key_roundtrip(key, parts):
    assert R.parse_key(key) == parts
    assert R.method_key(*parts) == key


def test_unknown_keys_are_rejected():
    for bad in ("kq", "bb_quadrat_1", "xx_regret_1"):
        with pytest.raises(ValueError):
            R.parse_key(bad)


def test_counting_grid_counts_rows_and_solves_like_grid():
    g = R.CountingGrid(5)
    C = np.random.default_rng(0).uniform(1, 3, size=(7, 40))
    W, z = g.solve(C)
    assert g.rows == 7 and R.Grid(5).solve(C)[1].tolist() == z.tolist()
    g.solve(C[:2])
    assert g.rows == 9


def test_solved_rows_by_hand():
    """Zählung von Hand (n = 50, 3 Schritte). SPO+: einmal w*(c) auf n Fällen, dann je Schritt ein Löseraufruf (w(2ĉ - c)): n + 3 n = 200.
    Interpolation: je Schritt zwei Aufrufe (w(ĉ), w(ĉ_λ)), dazu einmal n Fälle für die Beobachtungsgröße (Regret: w*(c); Hamming: der Zielweg): n + 6 n = 350, auch bei sauberem Ziel."""
    spec = R.RunSpec(noise=0.5, n=50, steps=3, n_test=20)
    keys = ("spo", "bb_regret_1", "bb_regret_clean_1", "bb_hamming_1", "bb_hamming_clean_1")
    run = R.run_world(spec, keys, snapshot_steps=())
    assert run.solved_rows == {"spo": 200, "bb_regret_1": 350, "bb_regret_clean_1": 350, "bb_hamming_1": 350, "bb_hamming_clean_1": 350}


def test_full_size_count_is_the_documented_one():
    """Für n = 1000 und 300 Schritte: SPO+ 301 000, Interpolation 601 000 (je einschließlich der einmaligen Lösung auf den 1000 Trainingskosten); nur gerechnet, nicht trainiert."""
    n, steps = 1000, 300
    assert n + steps * n == 301_000 and n + 2 * steps * n == 601_000


def test_without_noise_clean_and_noisy_variants_are_identical():
    """Ohne Rauschen ist M = C: der saubere Zielweg ist der verrauschte, die mittleren Kosten in dL/dy sind die beobachteten; dieselben Parameter."""
    spec = R.RunSpec(noise=0.0, n=60, steps=4, n_test=30)
    run = R.run_world(spec, ("bb_hamming_1", "bb_hamming_clean_1", "bb_regret_1", "bb_regret_clean_1"), snapshot_steps=())
    assert np.array_equal(run.coef["bb_hamming_1"], run.coef["bb_hamming_clean_1"])
    assert np.array_equal(run.coef["bb_regret_1"], run.coef["bb_regret_clean_1"])


def test_with_noise_clean_and_noisy_variants_differ():
    spec = R.RunSpec(noise=0.5, n=60, steps=4, n_test=30)
    run = R.run_world(spec, ("bb_hamming_1", "bb_hamming_clean_1", "bb_regret_1", "bb_regret_clean_1"), snapshot_steps=())
    assert not np.array_equal(run.coef["bb_hamming_1"], run.coef["bb_hamming_clean_1"])
    assert not np.array_equal(run.coef["bb_regret_1"], run.coef["bb_regret_clean_1"])


def test_run_has_the_documented_fields_and_units():
    spec = R.RunSpec(noise=0.0, n=60, steps=6, n_test=40)
    run = R.run_world(spec, ("spo", "bb_regret_1"), snapshot_steps=(0, 6))
    assert set(run.regret) == {"kq", "spo", "bb_regret_1", "orakel"} and set(run.mse) == {"kq", "spo", "bb_regret_1"}
    assert abs(run.regret["orakel"]) < 1e-12
    assert set(run.curve["spo"]) == {0, 6} and run.curve["spo"][0] == pytest.approx(run.regret["kq"])
    row = R.cell_result(run)
    assert row["regret"]["kq"] == pytest.approx(100 * run.regret["kq"]) and row["curve"]["bb_regret_1"]["6"] == pytest.approx(100 * run.curve["bb_regret_1"][6])


def test_with_noise_the_oracle_has_positive_regret():
    run = R.run_world(R.RunSpec(noise=0.5, n=40, steps=2, n_test=300), ("spo",), snapshot_steps=())
    assert run.regret["orakel"] > 0.02


def test_run_seeds_is_reproducible_and_summarize_works():
    spec = R.RunSpec(noise=0.0, n=40, steps=3, n_test=30)
    rows = R.run_seeds(spec, ("bb_regret_1",), [0, 1], snapshot_steps=())
    assert [r["seed"] for r in rows] == [0, 1] and rows == R.run_seeds(spec, ("bb_regret_1",), [0, 1], snapshot_steps=())
    assert R.summarize([3.0, 1.0, 2.0, 10.0]) == {"median": 2.5, "min": 1.0, "max": 10.0}
