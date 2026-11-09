"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Adresszeile (Begrenzen und Einrasten)."""

import bbl_constants as C
import bbl_evaluation as ev
import bbl_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_choices():
    for preset in C.PRESETS.values():
        assert preset["noise"] in C.NOISE_CHOICES and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Ohne Rauschen (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_the_hand_example_sliders():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)
    assert {"lam_slider", "c1_slider", "c2_slider"} <= set(P.SETTING_SPECS)


def test_snap_limits_and_snaps_to_the_allowed_values():
    assert P.snap(P.SETTING_SPECS["noise_slider"], 0.3) == 0.5 and P.snap(P.SETTING_SPECS["noise_slider"], -4.0) == 0.0
    assert P.snap(P.SETTING_SPECS["lam_slider"], 0.7) == 0.5 and P.snap(P.SETTING_SPECS["lam_slider"], 1e9) == 100.0
    assert P.snap(P.SETTING_SPECS["c1_slider"], 3.3) == 3.5 and P.snap(P.SETTING_SPECS["c1_slider"], 99.0) == 8.0 and P.snap(P.SETTING_SPECS["c2_slider"], -1.0) == 0.0
    assert P.snap(P.SETTING_SPECS["seed_input"], 10 ** 9) == C.SEED_MAX


def test_lambda_choices_contain_the_values_of_the_hand_example_in_the_plan():
    assert {0.25, 0.5, 1.0, 10.0} <= set(C.LAMBDA_HAND) and C.DEFAULT_LAMBDA_HAND in C.LAMBDA_HAND
