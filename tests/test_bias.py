"""Der Teilgradient an der Wahrheit (bbl_bias): Maß von Hand, Spiegelung bei SPO+, saubere Kosten, Reproduzierbarkeit."""

import numpy as np
import pytest

import bbl_bias as Bi
from bbl_world import WorldSpec


def test_relative_mean_gradient_by_hand():
    """Form (nx, ndraw, d). Zwei Ziehungen (1, 0) und (-1, 0): Mittel 0, Verhältnis 0. Beide (1, 0): Mittel (1, 0), Norm 1, mittlere Norm 1: 1.
    (1, 0) und (0, 1): Mittel (0,5; 0,5), Norm sqrt(0,5), mittlere Norm 1: 0,7071. Lauter Nullen: Verhältnis 0 (kein 0/0)."""
    G = np.array([[[1.0, 0.0], [-1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]])
    assert Bi.relative_mean_gradient(G).tolist() == pytest.approx([0.0, 1.0, np.sqrt(0.5), 0.0])


def test_draws_are_reproducible_and_have_the_documented_shape(grid5):
    a = Bi.draw_truth(WorldSpec(noise=0.5), grid5.d, seed=3, nx=7, ndraw=11)
    b = Bi.draw_truth(WorldSpec(noise=0.5), grid5.d, seed=3, nx=7, ndraw=11)
    assert a.M.shape == (7, 40) and a.C.shape == (7, 11, 40) and a.Z.shape == (7, 11, 40)
    assert np.array_equal(a.C, b.C) and np.array_equal(a.Z, b.Z)
    ratio = a.C / a.M[:, None, :]
    assert ratio.min() >= 0.5 and ratio.max() <= 1.5
    assert not np.array_equal(Bi.draw_truth(WorldSpec(noise=0.5), grid5.d, seed=4, nx=7, ndraw=11).C, a.C)


def test_without_noise_the_observed_costs_equal_the_mean_costs(grid5):
    d = Bi.draw_truth(WorldSpec(noise=0.0), grid5.d, seed=1, nx=5, ndraw=4)
    assert np.array_equal(d.C, np.repeat(d.M[:, None, :], 4, axis=1))


def test_spo_plus_gradient_at_the_truth_is_antisymmetric_under_mirrored_noise(grid5):
    """Mit ĉ = m und c = m ε ist 2m - c = m (2 - ε): die Ziehung ε' = 2 - ε (gespiegelt) vertauscht die beiden Wege w*(c) und w(2m - c); der Teilgradient 2 (w*(c) - w(2ĉ - c)) kehrt das Vorzeichen um.
    Mittel über ein gespiegeltes Paar: exakt null, also ist der Mittelwert an der Wahrheit bei symmetrischem Rauschen null."""
    d = Bi.draw_truth(WorldSpec(noise=0.5), grid5.d, seed=0, nx=6, ndraw=30)
    eps = d.C / d.M[:, None, :]
    mirrored = Bi.TruthDraws(d.M, d.M[:, None, :] * (2.0 - eps), d.Z)
    both = Bi.TruthDraws(d.M, np.concatenate([d.C, mirrored.C], axis=1), np.concatenate([d.Z, d.Z], axis=1))
    G = Bi.gradients_at_truth(grid5, both, "spo")
    assert np.abs(G.sum(axis=1)).max() < 1e-12
    assert np.abs(G[:, :30]).sum() > 0                            # die einzelnen Gradienten sind nicht null
    assert Bi.relative_mean_gradient(G).max() < 1e-12


def test_interpolation_with_the_mean_costs_as_dldy_is_exactly_zero_at_the_truth(grid5):
    """dL/dy = m (saubere Kosten) und ĉ = m: ĉ_λ = (1 + λ) m, derselbe Weg wie unter m: Gradient null (stetige Kosten)."""
    d = Bi.draw_truth(WorldSpec(noise=0.5), grid5.d, seed=0, nx=20, ndraw=3)
    from bbl_blackbox import bb_gradient

    G = bb_gradient(grid5, d.M, d.M, 1.0)
    assert np.all(G == 0)


def test_fenchel_young_without_perturbation_is_the_difference_of_the_two_best_paths(grid5):
    """σ = 0: Gradient w*(c) - w(m) (die Störung fällt weg); ohne Rauschen ist das null, mit Rauschen nicht."""
    d0 = Bi.draw_truth(WorldSpec(noise=0.0), grid5.d, seed=0, nx=8, ndraw=5)
    assert np.all(Bi.gradients_at_truth(grid5, d0, "fy", sigma=0.0) == 0)
    d1 = Bi.draw_truth(WorldSpec(noise=0.5), grid5.d, seed=0, nx=8, ndraw=50)
    G = Bi.gradients_at_truth(grid5, d1, "fy", sigma=0.0)
    assert np.abs(G).sum() > 0 and np.abs(G.sum(axis=2)).max() < 1e-12      # beide Wege haben je 8 Bögen


def test_unknown_kind_is_rejected(grid5):
    d = Bi.draw_truth(WorldSpec(noise=0.5), grid5.d, seed=0, nx=2, ndraw=2)
    with pytest.raises(ValueError):
        Bi.gradients_at_truth(grid5, d, "adam")


def test_kinds_are_the_five_columns_of_the_table():
    assert [name for name, _, _ in Bi.KINDS] == ["SPO+", "Fenchel-Young (σ = 0,3)", "Fenchel-Young (σ = 1)", "Interpolation (λ = 1)", "Interpolation (λ = 10)"]
    assert (Bi.N_X, Bi.N_DRAW) == (60, 400)


def test_bias_table_is_reproducible_and_has_all_columns(grid5):
    a = Bi.bias_table(grid5, WorldSpec(noise=0.5), 0, nx=6, ndraw=20)
    assert a == Bi.bias_table(grid5, WorldSpec(noise=0.5), 0, nx=6, ndraw=20)
    assert set(a) == {name for name, _, _ in Bi.KINDS} and all(0.0 <= v <= 1.0 + 1e-12 for v in a.values())
