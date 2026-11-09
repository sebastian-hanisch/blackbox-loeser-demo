"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, keine Seeds. Damit prüft der Test die Grenzen (gleiche angezeigte Werte, Gleichstand, Ausnahmezweige) und jede genannte Zahl direkt."""

import bbl_evaluation as ev


def test_de_formats_with_decimal_comma_and_no_negative_zero():
    assert ev.de(1.234) == "1,2" and ev.de(-0.04) == "0,0" and ev.de(-2.0, 2) == "-2,00" and ev.de(0.0) == "0,0"
    assert ev.de_g(0.25) == "0,25" and ev.de_g(10.0) == "10" and ev.de_g(0.5) == "0,5" and ev.de_g(100.0) == "100"


# --- Experiment 1 ----------------------------------------------------------------------------------------------------------------------------

def test_hand_verdict_below_the_crossing_is_a_warning_with_the_crossing_value():
    kind, text = ev.verdict_hand(ev.hand_numbers(4.0, 3.0, 0.25))
    assert kind == "warning"
    assert "λ = 0,25" in text and "(4,75; 4,25)" in text and "Weg 2" in text and "(0,00; 0,00)" in text and "erst ab λ = 0,5" in text


def test_hand_verdict_at_the_tie_names_the_fixed_rule_and_the_jump():
    kind, text = ev.verdict_hand(ev.hand_numbers(4.0, 3.0, 0.5))
    assert kind == "info"
    assert "(5,50; 5,50)" in text and "kleinster Weg" in text and "Weg 1" in text and "(2,00; -2,00)" in text and "genau bei diesem λ springt er" in text


def test_hand_verdict_above_the_crossing_names_both_paths_and_the_gradient():
    kind, text = ev.verdict_hand(ev.hand_numbers(4.0, 3.0, 1.0))
    assert kind == "success" and "von Weg 2 auf Weg 1" in text and "(1,00; -1,00)" in text
    kind, text = ev.verdict_hand(ev.hand_numbers(4.0, 3.0, 10.0))
    assert kind == "success" and "(0,10; -0,10)" in text


def test_hand_verdict_with_the_right_path_already_chosen_says_nothing_to_improve():
    for chat in ((2.0, 5.0), (3.0, 3.0), (0.0, 8.0)):       # auch der Gleichstand ĉ₁ = ĉ₂ wählt Weg 1
        kind, text = ev.verdict_hand(ev.hand_numbers(*chat, 1.0))
        assert kind == "info" and "schon den richtigen Weg (Weg 1)" in text and "(0,00; 0,00)" in text and "nichts zu verbessern" in text


def test_hand_numbers_fields_by_hand():
    h = ev.hand_numbers(4.0, 3.0, 0.5)
    assert (h["chosen"], h["chosen_lam"], h["best"], h["tie"], h["changed"]) == (2, 1, 1, True, True)
    assert h["chat_lam"] == (5.5, 5.5) and h["gradient"] == (2.0, -2.0) and h["crossing"] == 0.5 and h["regret"] == 2.0
    h2 = ev.hand_numbers(4.0, 3.0, 0.25)
    assert (h2["changed"], h2["tie"], h2["chosen_lam"]) == (False, False, 2)


def test_hand_curve_contains_the_app_lambdas_and_the_zero_then_one_over_lambda_shape():
    cu = ev.hand_curve(4.0, 3.0)
    lam, g1 = cu["lam"], cu["g1"]
    assert all(v in set(lam) for v in (0.25, 0.5, 1.0, 10.0))
    assert all(g == 0 for g, v in zip(g1, lam) if v < 0.5)
    assert all(abs(g - 1.0 / v) < 1e-12 for g, v in zip(g1, lam) if v >= 0.5)


# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------------

BB = {1.0: 20.0, 10.0: 21.0, 100.0: 22.0}


def test_noise_verdict_with_noise_is_a_warning_with_all_numbers():
    kind, text = ev.verdict_noise(0.5, 13.0, 12.0, BB, {1.0: 20, 10.0: 19, 100.0: 20}, 20)
    assert kind == "warning"
    assert "λ = 1 / 10 / 100" in text and "20,0 / 21,0 / 22,0 %" in text and "13,0 %" in text and "SPO+: 12,0 %" in text
    assert "Rauschstufe 0,5" in text and "20 / 19 / 20 von 20 Seeds" in text and "Median über 20 Zufalls-Seeds" in text


def test_noise_verdict_without_noise_is_a_success():
    kind, text = ev.verdict_noise(0.0, 2.0, 0.5, {1.0: 0.6, 10.0: 1.0, 100.0: 1.0}, {1.0: 0, 10.0: 0, 100.0: 0}, 20)
    assert kind == "success" and "Ohne Rauschen" in text and "0,6 / 1,0 / 1,0 %" in text and "2,0 %" in text and "0 / 0 / 0 von 20 Seeds" in text


def test_noise_verdict_equal_shown_values_are_not_counted_as_worse_or_better():
    kind, text = ev.verdict_noise(0.5, 13.0, 12.0, {1.0: 13.04, 10.0: 14.0, 100.0: 15.0}, {1.0: 9, 10.0: 12, 100.0: 15}, 20)
    assert kind == "info" and "Gemischtes Bild" in text and "13,0 / 14,0 / 15,0 %" in text


# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------

def test_bias_verdict_with_noise_names_the_three_numbers():
    vals = {"SPO+": 0.047, "Interpolation (λ = 1)": 0.713, "Interpolation (λ = 10)": 0.701}
    kind, text = ev.verdict_bias(0.5, vals, 20)
    assert kind == "warning"
    assert "0,713 (λ = 1) und 0,701 (λ = 10)" in text and "bei SPO+ bei 0,047" in text and "nicht stationär" in text and "Median über 20 Zufalls-Seeds, Rauschstufe 0,5" in text


def test_bias_verdict_without_noise_says_both_are_zero_and_an_odd_case_is_neutral():
    kind, text = ev.verdict_bias(0.0, {"SPO+": 0.0, "Interpolation (λ = 1)": 0.0, "Interpolation (λ = 10)": 0.0}, 20)
    assert kind == "info" and "beide genau null" in text and "stationär" in text and "Verhältnis 0,000" in text
    kind, text = ev.verdict_bias(0.5, {"SPO+": 0.4, "Interpolation (λ = 1)": 0.3, "Interpolation (λ = 10)": 0.2}, 20)
    assert kind == "info" and "SPO+ 0,400" in text


# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------

def test_loss_verdict_with_noise_compares_all_six_noisy_variants_against_kq():
    kind, text = ev.verdict_loss(0.5, 13.0, [18.0, 20.0, 21.0, 21.0, 22.0, 22.5], [11.0, 11.4, 11.6], 20)
    assert kind == "warning"
    assert "alle sechs Varianten mit verrauschtem Ziel liegen bei 18,0 bis 22,5 % gegen 13,0 % für KQ" in text and "bei 11,0 bis 11,6 %" in text
    assert "Median über 20 Zufalls-Seeds, Rauschstufe 0,5" in text


def test_loss_verdict_without_noise_is_a_success_and_a_mixed_case_is_neutral():
    kind, text = ev.verdict_loss(0.0, 2.0, [0.5, 0.6, 1.0, 1.1, 0.9, 1.0], [0.6, 1.0], 20)
    assert kind == "success" and "0,5 bis 1,1 % gegen 2,0 % für KQ" in text and "0,6 bis 1,0 %" in text
    kind, text = ev.verdict_loss(0.5, 13.0, [12.0, 20.0], [11.0], 20)
    assert kind == "info" and "verrauschtes Ziel 12,0 bis 20,0 %" in text
