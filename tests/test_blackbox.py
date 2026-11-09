"""Die Black-Box-Interpolation (bbl_blackbox): Handbeispiel, Eigenschaften, Randfälle, Zielfunktion des Trainings. Trainierte Zahlen nur als Ungleichungen und Bänder."""

import numpy as np
import pytest

import bbl_blackbox as BB
import bbl_training as T
from bbl_grid import PathSet
from bbl_world import WorldSpec, make_dataset


def test_hand_example_all_four_lambdas(two, hand_c, hand_chat):
    """Handrechnung, c = (3, 5), ĉ = (4, 3), w(ĉ) = Weg 2 = (0, 1). ĉ_λ = ĉ + λ c.
    λ = 0,25: ĉ_λ = (4,75; 4,25), Weg 2 wie zuvor, Gradient (0, 0). λ = 0,5: ĉ_λ = (5,5; 5,5), Gleichstand, feste Regel gibt Weg 1 = (1, 0): Gradient ((1,0) - (0,1)) / 0,5 = (2, -2).
    λ = 1: ĉ_λ = (7, 8), Weg 1: (1, -1). λ = 10: ĉ_λ = (34, 53), Weg 1: (1, -1) / 10 = (0,1; -0,1)."""
    expected = {0.25: [0.0, 0.0], 0.5: [2.0, -2.0], 1.0: [1.0, -1.0], 10.0: [0.1, -0.1]}
    for lam, g in expected.items():
        got = BB.bb_gradient(two, hand_chat, BB.dldy_regret(hand_c), lam)
        assert got[0].tolist() == pytest.approx(g), lam


def test_hand_example_solutions_and_shifted_costs(two, hand_c, hand_chat):
    g, W1, W2 = BB.bb_parts(two, hand_chat, hand_c, 0.5)
    assert W1.tolist() == [[0.0, 1.0]] and W2.tolist() == [[1.0, 0.0]]
    assert (hand_chat + 0.5 * hand_c).tolist() == [[5.5, 5.5]]            # exakter Gleichstand in Gleitkomma
    _, W1b, W2b = BB.bb_parts(two, hand_chat, hand_c, 0.25)
    assert W1b.tolist() == W2b.tolist() == [[0.0, 1.0]]


def test_lambda_towards_zero_gives_zero_while_the_path_does_not_change(two, hand_c, hand_chat):
    """Die Kreuzung liegt bei λ* = 0,5: für jedes kleinere λ ist die Wahl auf ĉ_λ dieselbe, der Gradient null."""
    for lam in (1e-9, 1e-3, 0.1, 0.25, 0.4999):
        assert BB.bb_gradient(two, hand_chat, hand_c, lam).tolist() == [[0.0, 0.0]], lam


def test_very_large_lambda_follows_the_formula_and_shrinks_like_one_over_lambda(two, hand_c, hand_chat):
    for lam in (1e3, 1e6):
        g = BB.bb_gradient(two, hand_chat, hand_c, lam)[0]
        assert g == pytest.approx(np.array([1.0, -1.0]) / lam, rel=1e-12)


def test_nonpositive_lambda_is_rejected(two, hand_c, hand_chat):
    for lam in (0.0, -1.0):
        with pytest.raises(ValueError):
            BB.bb_gradient(two, hand_chat, hand_c, lam)


def test_gradient_is_zero_exactly_when_the_two_solutions_agree_and_components_sum_to_zero(grid5):
    """Beide Lösungen sind Wege mit je 8 Bögen: die Komponenten von w2 - w1 summieren sich zu 0; g = 0 genau dann, wenn w2 = w1."""
    rng = np.random.default_rng(0)
    C = rng.uniform(0.5, 4, size=(80, 40))
    Chat = C + rng.normal(0, 1.0, size=C.shape)
    zero_rows = 0
    for lam in (0.1, 1.0, 10.0):
        g, W1, W2 = BB.bb_parts(grid5, Chat, C, lam)
        assert np.abs(g.sum(axis=1)).max() < 1e-12
        same = np.all(W1 == W2, axis=1)
        assert np.array_equal(same, np.all(g == 0, axis=1))
        zero_rows += int(same.sum())
    assert 0 < zero_rows < 240                      # beide Zweige kommen vor (Gradient null und nicht null)


def test_scaling_the_forecast_and_the_costs_together_does_not_change_the_gradient(grid5):
    """ĉ -> μ ĉ, c -> μ c bei gleichem λ: ĉ_λ -> μ ĉ_λ, jede Wahl bleibt (μ = 2 und 0,5 sind exakt in Gleitkomma): derselbe Gradient."""
    rng = np.random.default_rng(1)
    C = rng.uniform(0.5, 4, size=(40, 40))
    Chat = C + rng.normal(0, 1.0, size=C.shape)
    for mu in (2.0, 0.5):
        assert np.array_equal(BB.bb_gradient(grid5, Chat, C, 1.5), BB.bb_gradient(grid5, mu * Chat, mu * C, 1.5))


def test_trading_lambda_against_the_size_of_dldy_scales_the_gradient(grid5):
    """(λ, c) -> (λ / μ, μ c) lässt ĉ_λ gleich: die Wahlen bleiben, der Gradient wird mit μ multipliziert (g = Differenz / λ)."""
    rng = np.random.default_rng(2)
    C = rng.uniform(0.5, 4, size=(40, 40))
    Chat = C + rng.normal(0, 1.0, size=C.shape)
    g1 = BB.bb_gradient(grid5, Chat, C, 2.0)
    g2 = BB.bb_gradient(grid5, Chat, 4.0 * C, 0.5)
    assert np.array_equal(g2, 4.0 * g1)


def test_at_the_truth_without_noise_the_regret_gradient_is_zero(grid5):
    """ĉ = c: ĉ_λ = (1 + λ) c ist ein positives Vielfaches von c, der Weg ändert sich nicht (stetige Kosten, keine Gleichstände)."""
    rng = np.random.default_rng(3)
    C = rng.uniform(0.5, 4, size=(60, 40))
    for lam in (0.5, 1.0, 3.0, 10.0):
        assert np.all(BB.bb_gradient(grid5, C, C, lam) == 0), lam


def test_ties_follow_the_fixed_rule():
    """Zwei Wege, ĉ = 0 (Gleichstand: Weg 1) und c = (3, 5): ĉ_λ = λ (3, 5) wählt Weg 1 für jedes λ > 0, der Gradient ist null.
    Mit ĉ = (1, 1) und dL/dy = (0, -1): ĉ_λ = (1, 1 - λ), Weg 2 für jedes λ > 0; die Wahl auf ĉ ist Weg 1 (Gleichstand): g = ((0,1) - (1,0)) / λ."""
    two = PathSet(np.eye(2))
    assert BB.bb_gradient(two, np.zeros((1, 2)), np.array([[3.0, 5.0]]), 1.0).tolist() == [[0.0, 0.0]]
    g = BB.bb_gradient(two, np.ones((1, 2)), np.array([[0.0, -1.0]]), 0.5)
    assert g[0].tolist() == pytest.approx([-2.0, 2.0])


def test_all_equal_costs_give_zero_gradient_on_the_grid(grid5):
    assert np.all(BB.bb_gradient(grid5, np.ones((3, 40)), np.zeros((3, 40)), 1.0) == 0)


def test_hamming_distance_and_its_derivative_by_hand(two):
    """y = (1, 0), y* = (0, 1): Abstand 2. dL/dy = 1 - 2 y* = (1, -1). Für y = y*: Abstand 0."""
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    Ystar = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert BB.hamming_distance(W, Ystar).tolist() == [2.0, 0.0]
    assert BB.dldy_hamming(Ystar).tolist() == [[1.0, -1.0], [1.0, -1.0]]


def test_hand_example_with_the_hamming_loss(two, hand_chat):
    """Zielweg y* = Weg 1 = (1, 0): dL/dy = (-1, 1), ĉ_λ = ĉ + λ (-1, 1) = (4 - λ, 3 + λ): Gleichstand bei λ = 0,5 (Weg 1 nach der Regel), Gradient bei λ = 1: Weg 1 gegen Weg 2: (1, -1)."""
    dldy = BB.dldy_hamming(np.array([[1.0, 0.0]]))
    assert dldy.tolist() == [[-1.0, 1.0]]
    assert BB.bb_gradient(two, hand_chat, dldy, 0.25)[0].tolist() == [0.0, 0.0]       # ĉ_λ = (3,75; 3,25): Weg 2 wie zuvor
    assert BB.bb_gradient(two, hand_chat, dldy, 0.5)[0].tolist() == pytest.approx([2.0, -2.0])  # (3,5; 3,5): Gleichstand, Weg 1
    assert BB.bb_gradient(two, hand_chat, dldy, 1.0)[0].tolist() == pytest.approx([1.0, -1.0])


def test_objective_gradient_is_the_chain_rule_and_the_observed_loss_is_the_mean_regret(grid5):
    data = make_dataset(WorldSpec(deg=4, noise=0.5), grid5.d, 60, 10, seed=2)
    rng = np.random.default_rng(0)
    coef = rng.normal(0, 0.3, size=(6, 40)) + T.fit_least_squares(data.X_train, data.C_train)
    loss, grad = BB.bb_objective(grid5, data.X_train, data.C_train, 2.0)(coef)
    Xa = T.add_intercept(data.X_train)
    g, W1, _ = BB.bb_parts(grid5, Xa @ coef, data.C_train, 2.0)
    assert np.allclose(grad, Xa.T @ g / 60)
    _, zstar = grid5.solve(data.C_train)
    assert loss == pytest.approx(np.mean(np.sum(data.C_train * W1, axis=1) - zstar))


def test_objective_hamming_default_target_is_the_best_path_to_the_observed_costs(grid5):
    data = make_dataset(WorldSpec(deg=4, noise=0.5), grid5.d, 40, 10, seed=3)
    coef = T.fit_least_squares(data.X_train, data.C_train)
    Ystar, _ = grid5.solve(data.C_train)
    a = BB.bb_objective(grid5, data.X_train, data.C_train, 1.0, "hamming")(coef)
    b = BB.bb_objective(grid5, data.X_train, data.C_train, 1.0, "hamming", Ystar)(coef)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])
    with pytest.raises(ValueError):
        BB.bb_objective(grid5, data.X_train, data.C_train, 1.0, "quadrat")


def test_regret_with_mean_costs_as_dldy_equals_the_default_without_noise(grid5):
    data = make_dataset(WorldSpec(deg=4, noise=0.0), grid5.d, 40, 10, seed=3)
    coef = T.fit_least_squares(data.X_train, data.C_train)
    a = BB.bb_objective(grid5, data.X_train, data.C_train, 1.0)(coef)
    b = BB.bb_objective(grid5, data.X_train, data.C_train, 1.0, dldy_costs=data.M_train)(coef)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])


def test_train_bb_with_zero_steps_returns_the_start_and_last_is_the_final_parameter(grid5):
    data = make_dataset(WorldSpec(deg=4, noise=0.0), grid5.d, 60, 10, seed=1)
    kq = T.fit_least_squares(data.X_train, data.C_train)
    res0 = BB.train_bb(grid5, data.X_train, data.C_train, 1.0, steps=0, init=kq)
    assert np.array_equal(res0.coef, kq) and res0.losses.size == 0
    res = BB.train_bb(grid5, data.X_train, data.C_train, 1.0, steps=5, init=kq, snapshot_steps=(0, 5))
    assert np.array_equal(res.coef, res.snapshots[5]) and res.best_step == 5 and np.array_equal(res.snapshots[0], kq)


def test_bb_training_lowers_the_regret_of_kq_in_the_world_without_noise(grid5):
    """Band, kein Beleg für die Zahl: Welt Grad 6 ohne Rauschen, n = 300, 60 Schritte, λ = 1: das Testregret liegt unter dem von KQ (die Messung der README: 20 von 20 Seeds mit n = 1000)."""
    from bbl_losses import normalized_regret

    data = make_dataset(WorldSpec(deg=6, noise=0.0), grid5.d, 300, 500, seed=0)
    kq = T.fit_least_squares(data.X_train, data.C_train)
    res = BB.train_bb(grid5, data.X_train, data.C_train, 1.0, steps=60, init=kq)
    assert normalized_regret(grid5, data.C_test, T.predict(res.coef, data.X_test)) < normalized_regret(grid5, data.C_test, T.predict(kq, data.X_test))
