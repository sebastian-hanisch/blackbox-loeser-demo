"""Orakel-Tests (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): Löser, Interpolationsgradient, Hamming-Identität und Handformel gegen unabhängige Rechenwege.

1. Der Löser (Rückwärtsrechnung) gegen die Aufzählung der 70 Wege. Die Wege werden in `conftest.independent_paths` rekursiv erzeugt (andere Konstruktion als `Grid.paths`),
   die Kosten je Weg als Summe der Bogenkosten gerechnet; Gleichstände (ganzzahlige Kosten) folgen derselben Regel (erster kleinster in der Reihenfolge R vor D).
2. Der Gradient der Interpolation gegen eine Rechnung, die nur die Aufzählung benutzt: (Aufzählung auf ĉ + λ dL/dy minus Aufzählung auf ĉ) / λ, für beide Verluste.
3. Der Hamming-Abstand gegen die Aufzählung aller Wege: sum |y - y*| = y . (1 - 2 y*) + sum y* für jeden der 70 Wege.
4. Das Handbeispiel gegen die geschlossene Formel (zwei Wege, c = (3, 5)): Gradient 0 für λ < λ*, danach (1, -1) / λ mit λ* = (ĉ₁ - ĉ₂) / 2, wenn ĉ₁ > ĉ₂, sonst für jedes λ null.
5. Die Zielfunktion des Trainings gegen die Rechnung mit der Aufzählung (Kettenregel, Beobachtungsgröße).

Im Scratch-Lauf (nicht im Repo) mit 300 Instanzen für den Löser und 500 für den Gradienten wiederholt: Löser stetig 7e-15, ganzzahlig 0, Gradient 0 Abweichung."""

import numpy as np
import pytest

import bbl_blackbox as BB
import bbl_evaluation as ev
from bbl_training import add_intercept
from conftest import independent_paths, independent_solve


def test_solver_equals_enumeration_on_random_instances(grid5):
    rng = np.random.default_rng(0)
    C = rng.uniform(0.5, 5.0, size=(60, 40))
    W, z = grid5.solve(C)
    Wo, zo = independent_solve(grid5, C)
    assert np.abs(z - zo).max() < 1e-12
    assert np.array_equal(W, Wo)                      # stetige Kosten: der beste Weg ist eindeutig


def test_solver_equals_enumeration_with_ties_on_integer_costs(grid5):
    rng = np.random.default_rng(1)
    C = rng.integers(0, 3, size=(60, 40)).astype(float)
    W, z = grid5.solve(C)
    Wo, zo = independent_solve(grid5, C)
    assert np.array_equal(z, zo) and np.array_equal(W, Wo)   # auch die Wahl bei Gleichstand (R vor D) stimmt überein


def test_independent_enumeration_has_70_paths(grid5):
    paths = independent_paths(grid5)
    assert len(paths) == 70 and all(len(arcs) == 8 for _, arcs in paths)


@pytest.mark.parametrize("lam", [0.3, 1.0, 10.0, 100.0])
def test_regret_interpolation_gradient_equals_the_enumeration_formula(grid5, lam):
    rng = np.random.default_rng(int(lam * 10))
    C = rng.uniform(0.5, 4.0, size=(40, 40))
    Chat = C + rng.normal(0, 1.0, size=C.shape)
    W1, _ = independent_solve(grid5, Chat)
    W2, _ = independent_solve(grid5, Chat + lam * C)
    assert np.allclose(BB.bb_gradient(grid5, Chat, BB.dldy_regret(C), lam), (W2 - W1) / lam, rtol=0, atol=1e-15)


@pytest.mark.parametrize("lam", [0.3, 1.0, 10.0])
def test_hamming_interpolation_gradient_equals_the_enumeration_formula(grid5, lam):
    rng = np.random.default_rng(7 + int(lam * 10))
    C = rng.uniform(0.5, 4.0, size=(40, 40))
    Chat = C + rng.normal(0, 1.0, size=C.shape)
    Ystar, _ = independent_solve(grid5, C)
    W1, _ = independent_solve(grid5, Chat)
    W2, _ = independent_solve(grid5, Chat + lam * (1.0 - 2.0 * Ystar))
    assert np.allclose(BB.bb_gradient(grid5, Chat, BB.dldy_hamming(Ystar), lam), (W2 - W1) / lam, rtol=0, atol=1e-15)


def test_hamming_identity_over_all_70_paths(grid5):
    """Für 0/1-Vektoren ist sum |y - y*| = y . (1 - 2 y*) + sum y*; der Hamming-Abstand der Demo gegen die Zählung der verschiedenen Bögen je Wegepaar (70 x 70 Paare)."""
    P = grid5.paths()
    for k in range(0, 70, 7):
        ystar = P[k]
        by_identity = P @ BB.dldy_hamming(ystar[None, :])[0] + ystar.sum()
        by_count = np.array([sum(1 for a, b in zip(w, ystar) if a != b) for w in P], dtype=float)
        assert np.array_equal(by_identity, by_count)
        assert np.array_equal(BB.hamming_distance(P, np.repeat(ystar[None, :], 70, axis=0)), by_count)


def _closed_form(c1hat: float, c2hat: float, lam: float):
    """Geschlossene Formel des Handbeispiels (c = (3, 5), Weg 1 = (1, 0), Weg 2 = (0, 1), Gleichstand: Weg 1)."""
    if c1hat <= c2hat:                       # unter ĉ gewinnt Weg 1; ĉ_λ,1 - ĉ_λ,2 = (ĉ₁ - ĉ₂) - 2 λ < 0: bleibt Weg 1
        return (0.0, 0.0)
    lam_star = (c1hat - c2hat) / 2.0         # ab hier (inklusive Gleichstand) gewinnt Weg 1
    return (1.0 / lam, -1.0 / lam) if lam >= lam_star else (0.0, 0.0)


def test_hand_example_equals_the_closed_form_on_a_grid_of_forecasts_and_lambdas():
    """9 x 9 Vorhersagen (Schritt 1) und die acht λ der App: die geschlossene Formel gegen `hand_numbers` (Löser mit Wegliste)."""
    import bbl_constants as C

    checked = 0
    for c1 in np.arange(0.0, 8.01, 1.0):
        for c2 in np.arange(0.0, 8.01, 1.0):
            for lam in C.LAMBDA_HAND:
                h = ev.hand_numbers(float(c1), float(c2), float(lam))
                assert h["gradient"] == pytest.approx(_closed_form(c1, c2, lam), abs=1e-12), (c1, c2, lam)
                checked += 1
    assert checked == 9 * 9 * 8


def test_crossing_lambda_matches_the_closed_form():
    assert ev.crossing_lambda(4.0, 3.0) == 0.5 and ev.crossing_lambda(8.0, 0.0) == 4.0
    for c1, c2 in ((3.0, 3.0), (2.0, 5.0), (0.0, 8.0)):
        assert ev.crossing_lambda(c1, c2) is None


def test_training_objective_equals_the_enumeration_calculation(grid5):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 5))
    C = rng.uniform(1, 4, size=(30, 40))
    coef = rng.normal(0, 0.3, size=(6, 40)) + 2.0
    loss, grad = BB.bb_objective(grid5, X, C, 2.0)(coef)
    Chat = add_intercept(X) @ coef
    W1, _ = independent_solve(grid5, Chat)
    W2, _ = independent_solve(grid5, Chat + 2.0 * C)
    Wstar, zstar = independent_solve(grid5, C)
    assert np.allclose(grad, add_intercept(X).T @ ((W2 - W1) / 2.0) / 30, rtol=0, atol=1e-14)
    assert loss == pytest.approx(float(np.mean(np.sum(C * W1, axis=1) - zstar)), abs=1e-12)
