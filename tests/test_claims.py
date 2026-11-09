"""Claims: jede Zahl der README steht hier als wörtliche Tabellenzeile, Band oder Ungleichung (DEMO-PLAYBOOK §11a).

Zahlen-Tabelle der README -> Test: siehe die Tabelle „README-Aussage | Test“ im Abschnitt „Tests“ der README (jede Zeile nennt ihren Test, dieser steht hier oder in test_bias.py,
test_runs.py, test_blackbox.py, test_oracle_blackbox_loeser.py).

Trainierte Zahlen hängen von der Plattform ab: gegen die Bänder wird in Ungleichungen und mit Toleranz geprüft, exakt nur das Handbeispiel (`test_blackbox.py`), die Zählungen und die Tabellen
gegen die JSON-Datei (sie ist eine Messung dieser Demo, keine Rechnung der Tests)."""

from pathlib import Path

import numpy as np
import pytest

import bbl_bands as B
import bbl_constants as C
import readme_tables as T
from bbl_bias import bias_table
from bbl_evaluation import de
from bbl_grid import Grid
from bbl_runs import RunSpec, run_world
from bbl_world import WorldSpec

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
REGRET_LAMS = ("bb_regret_0.3", "bb_regret_1", "bb_regret_10", "bb_regret_100")
HAMMING = ("bb_hamming_0.3", "bb_hamming_1", "bb_hamming_10")
NOISY = REGRET_LAMS + HAMMING
CLEAN = ("bb_regret_clean_1", "bb_regret_clean_10", "bb_hamming_clean_0.3", "bb_hamming_clean_1", "bb_hamming_clean_10")


def _pairs(noise, a, b):
    return np.array(B.regrets(noise, a)) - np.array(B.regrets(noise, b))


def test_readme_contains_every_generated_table_line():
    blocks = [T.result_table(0.0), T.result_table(0.5), T.curve_table(0.0), T.curve_table(0.5), T.bias_table(), T.draws_table(), T.comparison_table(), T.effort_table()]
    missing = [line for lines in blocks for line in lines if line not in README]
    assert not missing, missing[:3]


def test_readme_has_no_leftover_placeholders_and_uses_the_agreed_words():
    assert "{{" not in README
    assert "[→ Demo live ausprobieren](https://sebastianhanisch-blackbox-loeser-demo.streamlit.app/)" in README   # Live-Link seit dem Deploy
    assert "TODO" not in README
    assert "Zufalls-Seed" in README and "Saatwert" not in README and "läuft lokal" not in README


def test_band_file_describes_the_documented_setup():
    meta = B.own()["meta"]
    assert meta["seeds"] == list(C.BAND_SEEDS) and meta["steps"] == C.STEPS and meta["lr"] == C.LR and meta["n"] == C.N_TRAIN and meta["n_test"] == C.N_TEST
    assert [c["noise"] for c in B.own()["cells"]] == [0.0, 0.5] and all(len(v) == 20 for c in B.own()["cells"] for v in c["regret"].values())
    assert set(B.cell(0.5)["regret"]) == {"orakel", "kq", "spo", *NOISY, *CLEAN}
    assert "Zufalls-Seeds 0 bis 19" in README and "300 Adam-Schritte" in README and "Lernrate 0,05" in README


def test_without_noise_the_interpolation_beats_kq_in_every_seed():
    """Rauschstufe 0: jede Interpolation mit Regret-Verlust ist in 20 von 20 Seeds besser als KQ; SPO+ ebenso; Medianbänder (Plattform: Toleranz)."""
    for key in ("spo",) + REGRET_LAMS:
        assert B.count_better(0.0, key) == 20 and B.count_worse(0.0, key) == 0, key
    kq = B.median(0.0, "kq")
    assert 1.6 < kq < 2.4                                              # KQ im Band um 2 % (Vormessung: 1,87)
    assert B.median(0.0, "spo") < 0.35 * kq                            # SPO+ senkt den Regret auf unter ein Drittel
    assert B.median(0.0, "bb_regret_1") < 0.4 * kq and B.median(0.0, "bb_regret_0.3") < 0.4 * kq
    assert abs(B.median(0.0, "bb_regret_1") - B.median(0.0, "spo")) < 0.15      # bei λ = 1 ungefähr gleich SPO+ (Abstand 0,06 Punkte in dieser Messung)
    assert abs(B.median(0.0, "orakel")) < 1e-9


def test_larger_lambda_is_coarser_without_noise():
    """λ = 10 und 100 liegen fast doppelt so hoch wie λ = 1 (Verhältnis 1,6 bis 2,2), aber unter KQ."""
    one = B.median(0.0, "bb_regret_1")
    for key in ("bb_regret_10", "bb_regret_100"):
        assert 1.6 < B.median(0.0, key) / one < 2.2, key
        assert B.median(0.0, key) < B.median(0.0, "kq")


def test_learning_curves():
    """Ohne Rauschen: Tiefpunkt von λ = 10 bei 25, von λ = 100 bei 50 Schritten, danach höher (Band 1,3 bis 2,0 des Tiefpunkts); mit Rauschen: ab 5 Schritten über dem KQ-Start."""
    c10, c100 = B.curve_median(0.0, "bb_regret_10"), B.curve_median(0.0, "bb_regret_100")
    assert min(c10, key=c10.get) == 25 and min(c100, key=c100.get) == 50
    assert 1.3 < c10[300] / c10[25] < 2.0 and 1.3 < c100[300] / c100[50] < 2.0
    steps = B.own()["meta"]["snapshot_steps"]
    kq5 = B.median(0.5, "kq")
    for key in ("spo",) + REGRET_LAMS[1:]:
        cv = B.curve_median(0.5, key)
        assert cv[0] == pytest.approx(kq5)                           # alle Kurven beginnen bei KQ
    for key in REGRET_LAMS[1:]:
        cv = B.curve_median(0.5, key)
        assert all(cv[s] > kq5 for s in steps if s >= 5), key        # schon nach 5 Schritten und danach immer über dem KQ-Start
    assert all(B.curve_median(0.5, "spo")[s] < kq5 for s in steps if s >= 5)  # SPO+ liegt mit Rauschen darunter


def test_with_noise_every_regret_interpolation_is_worse_than_kq_in_every_seed():
    """Rauschstufe 0,5: alle sieben Varianten mit verrauschtem Ziel in 20 von 20 Seeds schlechter als KQ; der kleinste Regret irgendeiner Variante in irgendeinem Seed liegt über dem größten von KQ."""
    for key in NOISY:
        assert B.count_worse(0.5, key) == 20 and B.count_better(0.5, key) == 0, key
    assert min(min(B.regrets(0.5, k)) for k in NOISY) > max(B.regrets(0.5, "kq"))
    kq = B.median(0.5, "kq")
    for key in REGRET_LAMS[1:]:
        assert 1.4 * kq < B.median(0.5, key) < 1.8 * kq, key         # "20 bis 22 %" gegen 13 %
    assert B.median(0.5, "bb_regret_0.3") < min(B.median(0.5, k) for k in REGRET_LAMS[1:])       # das kleinste λ ist besser als die übrigen drei
    assert float(np.median(_pairs(0.5, "bb_regret_1", "kq"))) == pytest.approx(8.0, abs=1.0)
    assert B.count_better(0.5, "spo") == 20 and B.median(0.5, "spo") < kq


def test_the_hamming_loss_does_not_change_the_verdict():
    """Hamming-Abstand zum verrauschten Zielweg: wie der Regret-Verlust. Ohne Rauschen λ = 1 und 10 in 20, λ = 0,3 in 18 Seeds besser als KQ; mit Rauschen 20 von 20 schlechter, Median 20 bis 23 %."""
    assert [B.count_better(0.0, k) for k in HAMMING] == [18, 20, 20] and [B.count_worse(0.0, k) for k in HAMMING] == [2, 0, 0]
    for key in HAMMING:
        assert 0.35 * B.median(0.0, "kq") < B.median(0.0, key) < 0.7 * B.median(0.0, "kq"), key
        assert 1.5 * B.median(0.5, "kq") < B.median(0.5, key) < 1.85 * B.median(0.5, "kq"), key


def test_clean_targets_beat_kq_and_close_most_of_the_gap():
    """Zusatzmessung: mit sauberem Ziel (mittlere Kosten in dL/dy bzw. Zielweg zu den mittleren Kosten) ist die Interpolation bei Rauschstufe 0,5 in 19 bis 20 von 20 Seeds besser als KQ,
    der Median liegt zwischen 11 und 12 %; Anteil der Lücke zwischen KQ und Orakel: SPO+ etwa ein Drittel, die sauberen Varianten 58 bis 77 %. Ohne Rauschen sind sauber und verrauscht gleich."""
    kq, orc = B.median(0.5, "kq"), B.median(0.5, "orakel")
    share = lambda k: (kq - B.median(0.5, k)) / (kq - orc)  # noqa: E731
    assert [B.count_better(0.5, k) for k in CLEAN] == [20, 20, 19, 20, 20] and [B.count_worse(0.5, k) for k in CLEAN] == [0, 0, 1, 0, 0]
    assert all(11.0 < B.median(0.5, k) < 12.0 for k in CLEAN)
    assert 0.25 < share("spo") < 0.37 and all(0.55 < share(k) < 0.8 for k in CLEAN)
    assert 0.74 < share("bb_regret_clean_1") < 0.80 and 0.56 < share("bb_hamming_clean_1") < 0.62
    for noisy, clean in (("bb_regret_1", "bb_regret_clean_1"), ("bb_regret_10", "bb_regret_clean_10"), ("bb_hamming_0.3", "bb_hamming_clean_0.3"), ("bb_hamming_1", "bb_hamming_clean_1"), ("bb_hamming_10", "bb_hamming_clean_10")):
        assert B.regrets(0.0, noisy) == B.regrets(0.0, clean), noisy                     # ohne Rauschen identisch (M = C)
        assert B.median(0.5, clean) < B.median(0.5, noisy) - 5.0, noisy                  # mit Rauschen um mehr als 5 Punkte besser


def test_gradient_at_the_truth_bands():
    """Rauschstufe 0: SPO+ und Interpolation in allen 20 Welten genau 0, Fenchel-Young 0,65 bis 0,75. Rauschstufe 0,5: SPO+ 0,03 bis 0,07, Interpolation 0,6 bis 0,8, Fenchel-Young 0,07 bis 0,2, in jeder Welt (Interpolation größer als das Fünffache von SPO+)."""
    b0, b5 = B.bias(0.0)["values"], B.bias(0.5)["values"]
    for name in ("SPO+", "Interpolation (λ = 1)", "Interpolation (λ = 10)"):
        assert max(b0[name]) == 0.0, name
    for name in ("Fenchel-Young (σ = 0,3)", "Fenchel-Young (σ = 1)"):
        assert all(0.65 < v < 0.75 for v in b0[name]), name
        assert all(0.07 < v < 0.2 for v in b5[name]), name
    assert all(0.03 < v < 0.07 for v in b5["SPO+"])
    for name in ("Interpolation (λ = 1)", "Interpolation (λ = 10)"):
        assert all(0.6 < v < 0.8 for v in b5[name]), name
        assert all(a > 5 * s for a, s in zip(b5[name], b5["SPO+"])), name


def test_more_draws_shrink_spo_plus_but_not_the_interpolation():
    """Welt Zufalls-Seed 0, Rauschstufe 0,5, 60 Merkmalsvektoren: bei 100 und 1600 Ziehungen die Werte der README (Toleranz 0,01); SPO+ fällt von 100 auf 1600 Ziehungen um mehr als die Hälfte (Wurzelgesetz: Faktor 4)."""
    grid = Grid(5)
    from bbl_bias import draw_truth, gradients_at_truth, relative_mean_gradient

    got = {}
    for nd in (100, 1600):
        d = draw_truth(WorldSpec(noise=0.5), grid.d, 0, nx=60, ndraw=nd)
        got[nd] = tuple(float(np.median(relative_mean_gradient(gradients_at_truth(grid, d, art, **kw))))
                        for art, kw in (("spo", {}), ("fy", {"sigma": 1.0}), ("bb", {"lam": 1.0})))
        assert got[nd] == pytest.approx(T.DRAWS[nd], abs=0.01), nd
    assert got[1600][0] < 0.5 * got[100][0] and abs(got[1600][2] - got[100][2]) < 0.02


def test_effort_counts():
    s = B.own()["meta"]["gelöste_instanzen_300_schritte"]
    assert s == {"spo": 301_000, "bb_regret_1": 601_000}
    assert s["spo"] == C.N_TRAIN + C.STEPS * C.N_TRAIN and s["bb_regret_1"] == C.N_TRAIN + 2 * C.STEPS * C.N_TRAIN
    assert "301.000" in README and "601.000" in README


def test_comparison_with_the_pre_measurement():
    """Gleiches Vorzeichen und Größenordnung wie die Vormessung: ohne Rauschen weichen die Werte um höchstens 0,2 Punkte ab, mit Rauschen um höchstens 1,5 Punkte."""
    for key, (v0, v5) in T.VORMESSUNG.items():
        assert abs(B.median(0.0, key) - v0) <= 0.2, key
        assert abs(B.median(0.5, key) - v5) <= 1.5, key
        if key != "kq":
            assert B.median(0.0, key) < B.median(0.0, "kq") and B.median(0.5, key) > B.median(0.5, "kq"), key
    for name, (v0, v5) in T.VORMESSUNG_BIAS.items():
        assert abs(B.bias(0.0)["summary"][name]["median"] - v0) <= 0.02, name
        assert abs(B.bias(0.5)["summary"][name]["median"] - v5) <= 0.03, name


def test_live_run_reproduces_the_bands():
    """Ein Lauf dieser Demo (Seed 0) reproduziert den Seed-0-Wert der Bänder: KQ und Orakel auf 0,05 Punkte, die trainierten Zahlen im Band (Plattform)."""
    for noise, tol in ((0.0, 0.3), (0.5, 2.5)):
        run = run_world(RunSpec(noise=noise, seed=0), ("spo", "bb_regret_1"), snapshot_steps=())
        for key in ("kq", "orakel"):
            assert 100 * run.regret[key] == pytest.approx(B.regrets(noise, key)[0], abs=0.05), (noise, key)
        for key in ("spo", "bb_regret_1"):
            assert 100 * run.regret[key] == pytest.approx(B.regrets(noise, key)[0], abs=tol), (noise, key)
    assert 100 * run.regret["bb_regret_1"] > 100 * run.regret["kq"]            # mit Rauschen (letzter Durchlauf) schlechter als KQ


def test_live_bias_table_matches_the_band_for_seed_zero():
    live = bias_table(Grid(5), WorldSpec(noise=0.5), 0)
    for name, v in live.items():
        assert v == pytest.approx(B.bias(0.5)["values"][name][0], abs=0.02), name


def test_app_text_numbers_come_from_the_bands():
    """Die Zahlen im Verdikt der Kernfrage stammen aus denselben Bändern wie die README (keine zweite Quelle)."""
    from bbl_evaluation import LAM_KEYS, verdict_noise

    bb = {lam: B.median(0.5, key) for lam, key in LAM_KEYS.items()}
    worse = {lam: B.count_worse(0.5, key) for lam, key in LAM_KEYS.items()}
    kind, text = verdict_noise(0.5, B.median(0.5, "kq"), B.median(0.5, "spo"), bb, worse, B.n_seeds())
    assert kind == "warning" and "20 / 20 / 20 von 20 Seeds" in text
    assert de(B.median(0.5, "kq")) in text and de(B.median(0.5, "bb_regret_1")) in text
    kind0, text0 = verdict_noise(0.0, B.median(0.0, "kq"), B.median(0.0, "spo"), {lam: B.median(0.0, key) for lam, key in LAM_KEYS.items()},
                                 {lam: B.count_worse(0.0, key) for lam, key in LAM_KEYS.items()}, B.n_seeds())
    assert kind0 == "success" and "0 / 0 / 0 von 20 Seeds" in text0
