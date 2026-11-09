"""Randfälle, an denen die Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import bbl_blackbox as BB
import bbl_evaluation as ev
import bbl_runs as R
from bbl_grid import Grid, PathSet


def test_exactly_two_nodes_per_side_gives_two_paths():
    """h = 2: der kleinste Fall mit etwas zu wählen (zwei Wege mit je 2 Bögen); die Interpolation läuft auch hier."""
    g = Grid(2)
    C = np.array([[1.0, 5.0, 1.0, 1.0]])
    assert g.solve(C)[0].sum() == 2.0
    grad = BB.bb_gradient(g, np.array([[1.0, 5.0, 1.0, 1.0]]), np.array([[5.0, 1.0, 1.0, 1.0]]), 1.0)
    assert grad.shape == (1, 4) and abs(grad.sum()) < 1e-12


def test_single_case_and_single_path():
    one = PathSet(np.array([[1.0, 1.0, 0.0]]))
    assert BB.bb_gradient(one, np.zeros((1, 3)), np.ones((1, 3)), 1.0).tolist() == [[0.0, 0.0, 0.0]]     # nur ein Weg: nichts kann sich ändern
    g = Grid(5)
    assert BB.bb_gradient(g, np.ones((1, 40)), np.ones((1, 40)), 2.0).shape == (1, 40)


def test_zero_forecast_and_zero_loss_derivative():
    """ĉ = 0: Gleichstand aller Wege (rechts vor unten); dL/dy = 0: ĉ_λ = ĉ, derselbe Weg, Gradient null."""
    g = Grid(5)
    assert np.all(BB.bb_gradient(g, np.zeros((2, 40)), np.zeros((2, 40)), 1.0) == 0)


def test_lambda_inclusive_boundaries_of_the_crossing():
    """λ genau gleich der Kreuzung λ* = 0,5: der Gleichstand gehört zum Wechsel (Gradient nicht null); knapp darunter null."""
    assert ev.hand_numbers(4.0, 3.0, 0.5)["gradient"] == (2.0, -2.0)
    assert ev.hand_numbers(4.0, 3.0, 0.25)["gradient"] == (0.0, 0.0)


def test_slider_extremes_of_the_hand_example_run():
    for c1, c2 in ((0.0, 0.0), (0.0, 8.0), (8.0, 0.0), (8.0, 8.0)):
        for lam in (0.1, 100.0):
            h = ev.hand_numbers(c1, c2, lam)
            kind, text = ev.verdict_hand(h)
            assert kind in ("info", "warning", "success") and text
            assert sum(h["gradient"]) == 0.0


def test_a_run_with_one_training_case_and_zero_steps_does_not_fail():
    run = R.run_world(R.RunSpec(noise=0.5, n=1, steps=0, n_test=5), ("spo", "bb_regret_1"), snapshot_steps=(0,))
    assert run.solved_rows["bb_regret_1"] == 1 and np.isfinite(run.regret["bb_regret_1"])


def test_nonpositive_lambda_is_an_error_not_a_silent_division():
    with pytest.raises(ValueError):
        BB.bb_gradient(Grid(5), np.ones((1, 40)), np.ones((1, 40)), 0.0)
