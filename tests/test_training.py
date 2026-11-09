"""Training (bbl_training): Adam von Hand, KQ auf linearer Wahrheit, SPO+-Trainingsverlust gegen KQ. Trainierte Zahlen nur als Ungleichungen und Bänder."""

import numpy as np
import pytest

import bbl_training as T
from bbl_runs import RunSpec, run_world
from bbl_world import WorldSpec, make_dataset


def test_adam_first_step_has_length_lr_in_every_coordinate():
    """Erster Adam-Schritt von Hand: m = 0,1 g, v = 0,001 g^2, nach der Verzerrungskorrektur m^ = g, v^ = g^2: Schritt = lr g / (|g| + 1e-8), also fast lr * Vorzeichen(g)."""
    g = np.array([2.0, -0.5, 100.0])
    res = T.adam_minimize(lambda c: (0.0, g), np.zeros(3), steps=1, lr=0.05, snapshot_steps=(1,))
    expected = -0.05 * g / (np.abs(g) + 1e-8)
    assert np.allclose(res.snapshots[1], expected, rtol=0, atol=1e-15)
    assert np.abs(res.snapshots[1] + 0.05 * np.sign(g)).max() < 1e-7


def test_adam_second_step_by_hand():
    """Zweiter Schritt mit konstantem Gradienten g = 1: m2 = 0,9 * 0,1 + 0,1 = 0,19, v2 = 0,999 * 0,001 + 0,001 = 0,001999; m^ = 0,19 / (1 - 0,81) = 1, v^ = 0,001999 / (1 - 0,998001) = 1: wieder Schritt lr."""
    res = T.adam_minimize(lambda c: (0.0, np.ones(1)), np.zeros(1), steps=2, lr=0.1, snapshot_steps=(2,))
    assert res.snapshots[2][0] == pytest.approx(-0.2, abs=1e-6)


def test_adam_minimizes_a_quadratic_and_returns_the_best_visited_point():
    target = np.array([3.0, -1.0])
    res = T.adam_minimize(lambda c: (float(np.sum((c - target) ** 2)), 2 * (c - target)), np.zeros(2), steps=400, lr=0.1)
    assert np.abs(res.coef - target).max() < 0.1
    assert res.losses.shape == (400,) and res.losses[0] == pytest.approx(10.0)
    assert res.losses[res.best_step] == pytest.approx(res.losses.min())      # Parameter und Verlust gehören zusammen
    assert res.best_step > 0


def test_adam_snapshot_zero_is_the_start_and_snapshots_are_copies():
    start = np.array([1.0, 2.0])
    res = T.adam_minimize(lambda c: (0.0, np.ones(2)), start, steps=3, lr=0.1, snapshot_steps=(0, 3))
    assert np.array_equal(res.snapshots[0], start)
    start[0] = 99.0
    assert res.snapshots[0][0] == 1.0


def test_least_squares_recovers_a_linear_truth_exactly(grid5):
    """Grad 1 ohne Rauschen: m(x) = (B x / sqrt(p) + 3) / 3,5 + 1 ist linear in x. KQ findet Achsenabschnitt 1 + 3/3,5 und Steigungen B / (3,5 sqrt(p)) bis auf Rundung."""
    spec = WorldSpec(p=5, deg=1, noise=0.0)
    data = make_dataset(spec, grid5.d, 200, 50, seed=3)
    coef = T.fit_least_squares(data.X_train, data.C_train)
    assert np.abs(coef[0] - (1 + 3 / 3.5)).max() < 1e-9
    assert np.abs(coef[1:] - data.B.T / (3.5 * np.sqrt(5))).max() < 1e-9
    assert np.abs(T.predict(coef, data.X_test) - data.C_test).max() < 1e-9


def test_predict_applies_intercept_and_slope():
    coef = np.array([[1.0, 2.0], [10.0, 0.0], [0.0, 100.0]])        # Achsenabschnitt (1, 2), Steigungen
    assert T.predict(coef, np.array([[1.0, 2.0]])).tolist() == [[11.0, 202.0]]


def test_spo_plus_training_loss_is_never_above_the_least_squares_start(grid5):
    """Der Start (KQ) gehört zu den besuchten Parametern: der Trainingsverlust des Ergebnisses ist höchstens der von KQ (jede Welt)."""
    for deg, noise, n, seed in [(6, 0.0, 100, 1), (6, 0.5, 100, 2), (4, 0.0, 300, 3)]:
        data = make_dataset(WorldSpec(deg=deg, noise=noise), grid5.d, n, 10, seed)
        kq = T.fit_least_squares(data.X_train, data.C_train)
        res = T.train_spo_plus(grid5, data.X_train, data.C_train, steps=40, init=kq)
        assert T.spo_plus_loss_mean(grid5, data.X_train, data.C_train, res.coef) <= T.spo_plus_loss_mean(grid5, data.X_train, data.C_train, kq) + 1e-12, (deg, noise, n)


def test_zero_steps_returns_the_start(grid5):
    data = make_dataset(WorldSpec(deg=4), grid5.d, 100, 10, seed=1)
    kq = T.fit_least_squares(data.X_train, data.C_train)
    res = T.train_spo_plus(grid5, data.X_train, data.C_train, steps=0, init=kq)
    assert np.array_equal(res.coef, kq) and res.losses.size == 0 and res.best_step == 0


def test_select_last_returns_the_parameter_after_the_final_step_and_best_the_one_with_the_smallest_loss():
    """Verlust (c - 1)^2 mit zu großer Lernrate: Adam springt über das Minimum hinaus; "best" hält den besten besuchten Punkt, "last" den Endpunkt nach `steps` Schritten."""
    target = np.array([1.0])
    fn = lambda c: (float(np.sum((c - target) ** 2)), 2 * (c - target))      # noqa: E731
    last = T.adam_minimize(fn, np.zeros(1), steps=7, lr=0.9, select="last")
    best = T.adam_minimize(fn, np.zeros(1), steps=7, lr=0.9, select="best")
    assert last.best_step == 7
    assert best.losses[best.best_step] == best.losses.min()
    assert best.best_step <= 6                           # der Verlust wird nur VOR jedem der 7 Schritte bewertet (Schritt 0 bis 6)
    with pytest.raises(ValueError):
        T.adam_minimize(fn, np.zeros(1), steps=2, select="irgendwas")


def test_select_last_equals_best_when_the_loss_falls_every_step():
    """Konstanter Gradient 1 (der Verlust fällt in jeder Runde): der Endpunkt ist der beste besuchte Punkt, beide Arten geben denselben Parameter zurück."""
    fn = lambda c: (float(c[0]), np.ones(1))      # noqa: E731
    res_last = T.adam_minimize(fn, np.zeros(1), steps=5, lr=0.1, select="last")
    res_best = T.adam_minimize(fn, np.zeros(1), steps=5, lr=0.1, select="best")
    assert res_last.coef[0] == pytest.approx(-0.5, abs=1e-6)
    assert res_best.coef[0] == pytest.approx(-0.4, abs=1e-6)          # "best" kennt den Verlust nur VOR jedem Schritt: der Punkt nach dem fünften Schritt wird nie bewertet
    assert res_best.best_step == 4


def test_l2_penalty_changes_the_gradient_not_the_reported_loss(grid5):
    rng = np.random.default_rng(0)
    X, C = rng.normal(size=(40, 5)), rng.uniform(1, 4, size=(40, 40))
    coef = rng.normal(size=(6, 40))
    l0, g0 = T.spo_plus_objective(grid5, X, C, 0.0)(coef)
    l1, g1 = T.spo_plus_objective(grid5, X, C, 0.5)(coef)
    assert l0 == l1
    assert np.allclose(g1 - g0, 0.5 * coef)
