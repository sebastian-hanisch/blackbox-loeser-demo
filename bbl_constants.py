"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Das Gitter und die Welt (aus Stück 3 der Reihe übernommen) -----------------------------------------------------------------------------
GRID_H = 5                       # 5 x 5 Knoten: 40 Bögen, 70 Wege, je 8 Bögen
N_FEATURES = 5                   # Merkmale je Fall
DEG = 6                          # Grad der Wahrheit (fest): ein lineares Modell kann sie nur annähern
N_TRAIN = 1000                   # Trainingsfälle je Welt (fest)
N_TEST = 1000                    # Testfälle je Welt
STEPS, LR = 300, 0.05            # Adam: Schritte und Lernrate
BAND_SEEDS = tuple(range(20))    # Zufalls-Seeds der vorgerechneten Bänder (bbl_bands.json)

NOISE_CHOICES = (0.0, 0.5)       # Rauschstufe e
DEFAULT_NOISE, DEFAULT_SEED = 0.0, 0
SEED_MAX = 999999

# Live gerechnete Interpolationsläufe in Experiment 2 (Regret-Verlust)
LAMBDA_LIVE = (1.0, 10.0, 100.0)

# --- Das Handbeispiel (zwei Wege mit je einem Bogen) ---------------------------------------------------------------------------------------
HAND_C = (3.0, 5.0)              # wahre Kosten
CHAT_MIN, CHAT_MAX, CHAT_STEP = 0.0, 8.0, 0.5
DEFAULT_C1HAT, DEFAULT_C2HAT = 4.0, 3.0
LAMBDA_HAND = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0)   # Auswahl für das Handbeispiel (die Werte des Plans: 0,25 / 0,5 / 1 / 10)
DEFAULT_LAMBDA_HAND = 1.0
LAMBDA_CURVE = (0.05, 200.0, 241)  # Bereich und Zahl der Punkte der Kurve "Gradient gegen lam" (logarithmisch)

# --- Anzeige ------------------------------------------------------------------------------------------------------------------------------
COLOR_KQ = "#4c78a8"
COLOR_SPO = "#2F6B65"
COLOR_ORACLE = "#7f7f7f"
COLOR_BB = {1.0: "#d62728", 10.0: "#e8943a", 100.0: "#8c564b"}
COLOR_FY = "#b8860b"
COLOR_REGION_1 = "rgba(76,120,168,0.10)"
COLOR_REGION_2 = "rgba(214,39,40,0.08)"

PRESETS = {
    "Ohne Rauschen (Voreinstellung)": {"noise": 0.0, "seed": 0},
    "Mit Rauschen, Stufe 0,5": {"noise": 0.5, "seed": 0},
    "Mit Rauschen, anderer Zufalls-Seed": {"noise": 0.5, "seed": 1},
}
PRESET_HELP = {
    "Ohne Rauschen (Voreinstellung)": "Rauschstufe 0, Zufalls-Seed 0: beobachtete Kosten und Mittelkosten sind gleich; hier arbeitet die Interpolation wie SPO+.",
    "Mit Rauschen, Stufe 0,5": "Rauschstufe 0,5, Zufalls-Seed 0: jeder Bogen kostet das Mittel mal einen Faktor aus [0,5; 1,5]; hier liegt die Interpolation in der Messung dieser Demo über Kleinsten Quadraten.",
    "Mit Rauschen, anderer Zufalls-Seed": "Rauschstufe 0,5, Zufalls-Seed 1: dieselbe Lage mit einer anderen Welt; eine einzelne Welt streut, die Aussage steht in den 20 Zufalls-Seeds der Bänder.",
}
