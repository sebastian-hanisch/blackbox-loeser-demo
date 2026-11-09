"""Kostenwelt (bbl_world) und Lauf (bbl_runs): Formeln von Hand, Reproduzierbarkeit, Rauschbereich, Zusammenfassungen."""

import numpy as np
import pytest

import bbl_runs as R
from bbl_world import TAG_TEST, TAG_TRAIN, WorldSpec, draw_data, draw_weights, make_dataset, mean_costs


def test_mean_costs_by_hand():
    """p = 1, ein Bogen mit Gewicht 1: m = (x / 1 + 3)^deg / 3,5^deg + 1. Bei x = 0.5 und deg 1: 3,5 / 3,5 + 1 = 2; deg 2: (3,5/3,5)^2 + 1 = 2.
    Bei x = -0.5 und deg 2: (2,5 / 3,5)^2 + 1 = 1 + 25/49. Gewicht 0 (zweiter Bogen): m = (3/3,5)^deg + 1."""
    B = np.array([[1.0], [0.0]])
    X = np.array([[0.5], [-0.5]])
    m1 = mean_costs(B, X, WorldSpec(p=1, deg=1, noise=0.0))
    assert m1[0, 0] == pytest.approx(2.0) and m1[1, 0] == pytest.approx(2.5 / 3.5 + 1)
    m2 = mean_costs(B, X, WorldSpec(p=1, deg=2, noise=0.0))
    assert m2[0, 0] == pytest.approx(2.0) and m2[1, 0] == pytest.approx(1 + 25 / 49)
    assert m2[0, 1] == pytest.approx((3 / 3.5) ** 2 + 1)


def test_weights_are_zero_or_one_with_about_half_ones():
    B = draw_weights(40, WorldSpec(p=5), 11)
    assert B.shape == (40, 5) and set(np.unique(B)) <= {0.0, 1.0}
    assert 0.3 < B.mean() < 0.7


def test_same_seed_same_data_and_train_differs_from_test(grid5):
    spec = WorldSpec(deg=6, noise=0.5)
    a, b = make_dataset(spec, 40, 50, 30, 5), make_dataset(spec, 40, 50, 30, 5)
    assert np.array_equal(a.C_train, b.C_train) and np.array_equal(a.X_test, b.X_test) and np.array_equal(a.B, b.B)
    assert not np.array_equal(a.X_train[:30], a.X_test)
    assert not np.array_equal(make_dataset(spec, 40, 50, 30, 6).C_train, a.C_train)


def test_without_noise_observed_equals_mean_and_with_noise_ratio_is_inside_the_band():
    B = draw_weights(40, WorldSpec(), 3)
    _, C0, M0 = draw_data(B, WorldSpec(noise=0.0), 100, 3, TAG_TRAIN)
    assert np.array_equal(C0, M0)
    _, C1, M1 = draw_data(B, WorldSpec(noise=0.5), 500, 3, TAG_TRAIN)
    ratio = C1 / M1
    assert ratio.min() >= 0.5 and ratio.max() <= 1.5 and abs(ratio.mean() - 1.0) < 0.01
    _, C2, M2 = draw_data(B, WorldSpec(noise=1.0), 500, 3, TAG_TEST)
    assert (C2 / M2).min() >= 0.0


def test_costs_are_positive_in_the_standard_worlds(grid5):
    for deg in (1, 2, 4, 6):
        data = make_dataset(WorldSpec(deg=deg, noise=0.5), grid5.d, 300, 300, 2)
        assert data.C_train.min() > 0 and data.C_test.min() > 0




def test_dataset_holds_the_mean_costs_of_the_training_cases(grid5):
    """M_train ist m(x) der Trainingsfälle: ohne Rauschen gleich den beobachteten Kosten, mit Rauschen liegt das Verhältnis im Band [1 - e, 1 + e]."""
    d0 = make_dataset(WorldSpec(noise=0.0), grid5.d, 50, 10, 4)
    assert np.array_equal(d0.M_train, d0.C_train)
    d1 = make_dataset(WorldSpec(noise=0.5), grid5.d, 200, 10, 4)
    ratio = d1.C_train / d1.M_train
    assert ratio.min() >= 0.5 and ratio.max() <= 1.5
    assert np.allclose(d1.M_train, mean_costs(d1.B, d1.X_train, d1.spec))


def test_the_four_streams_are_separate(grid5):
    """Gewichte, Training, Test und die Messung an der Wahrheit bekommen getrennte Teilströme aus einem Zufalls-Seed (verschiedene erste Ausgabewerte)."""
    from bbl_rng import derive_seed
    from bbl_world import TAG_TRUTH, TAG_WEIGHTS

    firsts = {derive_seed(7, tag) for tag in (TAG_WEIGHTS, TAG_TRAIN, TAG_TEST, TAG_TRUTH)}
    assert len(firsts) == 4
