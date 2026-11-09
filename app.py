"""Der Löser als Black Box: Interpolation statt Ableitung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Man kann einen Löser auch ableiten, ohne ihn zu öffnen: ein zweiter Löseraufruf mit verschobenen Kosten, die Differenz der beiden Lösungen ist der Gradient (Interpolation mit dem
Parameter lam). Im 5 x 5 Gitter ohne Rauschen funktioniert das wie SPO+, mit verrauschten Zielen liegt es in diesem Aufbau über Kleinsten Quadraten (Negativbefund im eigenen Aufbau).
Aufbau nach DEMO-PLAYBOOK §2. Kern ohne Streamlit: bbl_rng, bbl_grid, bbl_world, bbl_losses, bbl_training, bbl_blackbox, bbl_bias, bbl_runs.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import bbl_bands as B
import bbl_constants as C
from bbl_bias import bias_table
from bbl_evaluation import (
    LAM_KEYS,
    Settings,
    de,
    de_g,
    hand_curve,
    hand_numbers,
    loss_rows,
    noise_rows,
    verdict_bias,
    verdict_hand,
    verdict_loss,
    verdict_noise,
)
from bbl_grid import Grid
from bbl_layout import dunkles_design, kopf_css
from bbl_pdf_export import build_pdf
from bbl_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from bbl_runs import run_world
from bbl_visualization import build_bias_figure, build_hand_figure, build_loss_figure, build_noise_figure
from bbl_world import WorldSpec

st.set_page_config(page_title="Der Löser als Black Box – Sebastian Hanisch", layout="wide")

LIVE_KEYS = ("spo",) + tuple(LAM_KEYS.values())


@st.cache_data(show_spinner="Training läuft: KQ, SPO+ und drei Läufe der Interpolation mit je 300 Adam-Schritten (je Schritt zwei Löseraufrufe auf allen 1000 Fällen) …")
def _world_run(noise: float, seed: int):
    return run_world(Settings(noise=noise, seed=seed).run_spec(), LIVE_KEYS, snapshot_steps=())


@st.cache_data(show_spinner=False)
def _bias_live(noise: float, seed: int) -> dict:
    return bias_table(Grid(C.GRID_H), WorldSpec(p=C.N_FEATURES, deg=C.DEG, noise=noise), seed)


def _pct(x: float) -> str:
    return f"{de(100 * x)} %"


def _thou(n: int) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{n:,}".replace(",", ".")


def _range(r: dict) -> str:
    return f"{de(r['median'])} % ({de(r['min'])} bis {de(r['max'])})"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in bbl_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Der Löser als Black Box: Interpolation statt Ableitung")
st.markdown(
    """
Wenn ein Löser eine **Entscheidung** liefert (hier: den kürzesten Weg im Netz), ist sie als Funktion der Kosten **stückweise konstant**: fast überall ist ihr Gradient null. Man kann sie trotzdem ableiten, **ohne den Löser zu öffnen**:
einmal lösen mit den vorhergesagten Kosten, ein zweites Mal mit **verschobenen** Kosten, und die Differenz der beiden Lösungen, geteilt durch den Verschiebungsparameter **λ**, ist der Gradient (eine Interpolation).
Im 5 × 5 Gitter ohne Rauschen funktioniert das wie SPO+. **Mit verrauschten Zielwerten liegt es in meinem Nachbau deutlich über Kleinsten Quadraten**: ein **Negativbefund im eigenen Aufbau**, kein Urteil über das Verfahren der Quelle.

Weiter unten: **🔬 vier Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Interpolation", expanded=True):
    st.markdown(
        """
1. **Den Löser fragen:** zu den vorhergesagten Kosten $\\hat c$ liefert er den Weg $w(\\hat c)$; das Modell ist linear in den Merkmalen und liefert $\\hat c$ für alle 40 Bögen.
2. **Die Kosten verschieben:** $\\hat c_\\lambda = \\hat c + \\lambda\\, \\mathrm{d}L/\\mathrm{d}y$, wobei $L$ der Verlust der Entscheidung ist (beim Regret-Verlust ist $\\mathrm{d}L/\\mathrm{d}y = c$, die beobachteten Kosten). Ein zweiter Löseraufruf liefert $w(\\hat c_\\lambda)$.
3. **Die Differenz ist der Gradient:** $g = \\big(w(\\hat c_\\lambda) - w(\\hat c)\\big) / \\lambda$. Ändert sich der Weg nicht, ist $g = 0$; ein kleines λ bleibt nah an der stückweise konstanten Funktion (oft null), ein großes λ macht den Gradienten aussagekräftiger, aber gröber.
4. **Trainieren:** der Gradient nach den Parametern des linearen Modells ist $X^\\top g / n$ (Kettenregel); Adam wie bei SPO+, **zwei Löseraufrufe je Schritt** und Fall. Die Form des Gradienten steht bei Vlastelica u. a. (ICLR 2020) im Haupttext (Algorithmus 1), nicht im arXiv-Abstract; den Volltext habe ich selbst nicht gelesen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# --- Seitenleiste: Schnellstart und Kostenwelt -------------------------------------------------------------------------------------------------

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Kostenwelt**")
    noise = st.select_slider("Rauschstufe", options=list(C.NOISE_CHOICES), key="noise_slider", format_func=lambda v: de_g(v),
                             help="Jeder Bogen kostet das Mittel mal einen Faktor aus [1 − e, 1 + e]. Bei 0 sind beobachtete und mittlere Kosten gleich, bei 0,5 liegt auch das Orakel beim Regret deutlich über null.")
    seed = st.number_input("Zufalls-Seed der Welt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Welt generieren", width="stretch", on_click=randomize_seed)
    st.caption(f"Fest: 5 × 5 Gitter, Grad {C.DEG} der Wahrheit, {C.N_TRAIN} Trainingsfälle, lineares Modell, 300 Adam-Schritte. λ und die Vorhersage des Handbeispiels (Experiment 1) haben ihre Regler beim Experiment.")

run = _world_run(float(noise), int(seed))
n_seeds = B.n_seeds()

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis dieser Welt")
st.caption(f"Eine Instanz (Zufalls-Seed {int(seed)}): {C.N_TRAIN} Trainingsfälle, {C.N_TEST} Testfälle, Regret-Verlust; normierter Regret = Summe der Regrets geteilt durch die Summe der besten Wegkosten.")
m1, m2, m3, m4, m5, m6 = st.columns(6)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Regret Orakel", _pct(run.regret["orakel"]), help="Das Orakel kennt die mittleren Kosten der Testfälle; sein Regret ist der unvermeidbare Teil (Rauschen).")
m2.metric("Regret KQ", _pct(run.regret["kq"]), delta=f"{de(100 * (run.regret['kq'] - run.regret['orakel']))} Punkte über dem Orakel", delta_color="off")
m3.metric("Regret SPO+", _pct(run.regret["spo"]), delta=f"{de(100 * (run.regret['spo'] - run.regret['kq']))} Punkte gegen KQ", delta_color="inverse")
for col, (lam, key) in zip((m4, m5, m6), LAM_KEYS.items()):
    col.metric(f"Interpolation, λ = {de_g(lam)}", _pct(run.regret[key]), delta=f"{de(100 * (run.regret[key] - run.regret['kq']))} Punkte gegen KQ", delta_color="inverse")

settings = Settings(float(noise), int(seed), float(st.session_state.get("lam_slider", C.DEFAULT_LAMBDA_HAND)),
                    float(st.session_state.get("c1_slider", C.DEFAULT_C1HAT)), float(st.session_state.get("c2_slider", C.DEFAULT_C2HAT)))
if st.button("📄 PDF dieser Welt erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, run))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="blackbox-loeser.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Trägt die Interpolation auch mit verrauschten Zielen?")
bb_med = {lam: B.median(float(noise), key) for lam, key in LAM_KEYS.items()}
bb_worse = {lam: B.count_worse(float(noise), key) for lam, key in LAM_KEYS.items()}
kind, text = verdict_noise(float(noise), B.median(float(noise), "kq"), B.median(float(noise), "spo"), bb_med, bb_worse, n_seeds)
getattr(st, kind)(text)
st.caption(
    f"Das Verdikt gilt für die gewählte Rauschstufe ({de_g(noise)}) in diesem Aufbau: Grad {C.DEG}, {C.N_TRAIN} Trainingsfälle, lineares Modell, Regret-Verlust, Median über {n_seeds} Zufalls-Seeds, "
    "vorgerechnet mit `tools/measure_bands.py`. Es sagt nichts darüber, wie das Verfahren der Quelle in deren Experimenten abschneidet."
)

st.markdown("---")

# --- Experiment 1 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: die Interpolation von Hand, zwei Wege")
st.markdown(
    "Zwei Wege mit je einem Bogen, Regret-Verlust. Die **wahren Kosten** sind $c = (3, 5)$: Weg 1 ist der beste. Stellen Sie eine **Vorhersage** $\\hat c$ und die **Verschiebung** $\\lambda$ ein: "
    "$\\hat c_\\lambda = \\hat c + \\lambda c$, der Gradient ist $(w(\\hat c_\\lambda) - w(\\hat c)) / \\lambda$. Bei Gleichstand gilt die feste Regel „kleinster Weg“ (Weg 1)."
)
h1, h2, h3 = st.columns(3)
with h1:
    c1 = st.slider("Vorhersage ĉ₁ (Weg 1)", C.CHAT_MIN, C.CHAT_MAX, key="c1_slider", step=C.CHAT_STEP, format="%.1f", help="Vorhergesagte Kosten von Weg 1 (wahr: 3).")
with h2:
    c2 = st.slider("Vorhersage ĉ₂ (Weg 2)", C.CHAT_MIN, C.CHAT_MAX, key="c2_slider", step=C.CHAT_STEP, format="%.1f", help="Vorhergesagte Kosten von Weg 2 (wahr: 5).")
with h3:
    lam = st.select_slider("Verschiebung λ", options=list(C.LAMBDA_HAND), key="lam_slider", format_func=de_g,
                           help="Wie weit die Kosten für den zweiten Löseraufruf in Richtung der wahren Kosten verschoben werden. Die Werte 0,25 / 0,5 / 1 / 10 sind die des Handbeispiels in der README.")
hn = hand_numbers(float(c1), float(c2), float(lam))
curve = hand_curve(float(c1), float(c2))
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Weg unter ĉ", f"Weg {hn['chosen']}", help="Der Weg mit den kleineren vorhergesagten Kosten; bei Gleichstand Weg 1.")
k2.metric("ĉ_λ = ĉ + λ c", f"({de(hn['chat_lam'][0], 2)}; {de(hn['chat_lam'][1], 2)})", help="Die verschobenen Kosten für den zweiten Löseraufruf.")
k3.metric("Weg unter ĉ_λ", f"Weg {hn['chosen_lam']}", help="Gleichstand in ĉ_λ: die feste Regel wählt Weg 1." if hn["tie"] else None)
k4.metric("Gradient nach ĉ", f"({de(hn['gradient'][0], 2)}; {de(hn['gradient'][1], 2)})", help="(Lösung auf ĉ_λ − Lösung auf ĉ) / λ.")
k5.metric("Kreuzung bei λ*", "–" if hn["crossing"] is None else de_g(hn["crossing"]), help="Das λ, ab dem ĉ_λ auf der anderen Seite liegt. Wählt ĉ schon Weg 1, gibt es keine Kreuzung.")
st.plotly_chart(build_hand_figure(hn, curve, (float(c1), float(c2))), width="stretch", key="hand_chart")
st.caption("Unterhalb der Kreuzung ist der Gradient null, darüber wird er mit wachsendem λ kleiner (hier 1/λ mal (1; −1)): die Interpolation wird gröber. Rechts steht die erste Komponente des Gradienten.")
kind, text = verdict_hand(hn)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: ohne und mit Rauschen")
st.markdown(
    f"Testregret nach t Adam-Schritten, **Median über {n_seeds} Zufalls-Seeds** (vorgerechnet), links ohne Rauschen, rechts mit Rauschstufe 0,5: KQ (gestrichelt), SPO+ und die Interpolation mit "
    "λ = 1 / 10 / 100. Die Interpolation nimmt den Parameter nach dem letzten Schritt, SPO+ den mit dem kleinsten Trainingsverlust (wie in der Demo `spo-kuerzester-weg-demo`)."
)
curves = {}
for e in C.NOISE_CHOICES:
    entry = {"kq": B.median(e, "kq"), "orakel": B.median(e, "orakel"), "spo": B.curve_median(e, "spo")}
    entry.update({key: B.curve_median(e, key) for key in LAM_KEYS.values()})
    curves[e] = entry
st.plotly_chart(build_noise_figure(curves), width="stretch", key="noise_chart")
rows2 = noise_rows(float(noise))
table2 = pd.DataFrame([
    {"Verfahren": r["Verfahren"], "Regret (Median)": f"{de(r['median'])} %", "Spanne über Seeds": f"{de(r['min'])} bis {de(r['max'])} %",
     f"Schlechter als KQ (von {n_seeds} Seeds)": "–" if r["schlechter"] is None else str(r["schlechter"])} for r in rows2
])
st.markdown(f"Rauschstufe {de_g(noise)} (Seitenleiste), nach 300 Schritten:")
st.dataframe(table2, hide_index=True, width="stretch")
kind, text = verdict_noise(float(noise), B.median(float(noise), "kq"), B.median(float(noise), "spo"), bb_med, bb_worse, n_seeds)
getattr(st, kind)(text)
solved = B.own()["meta"]["gelöste_instanzen_300_schritte"]
st.caption(
    f"Aufwand je Training (n = {C.N_TRAIN}, 300 Schritte): SPO+ löst {_thou(solved['spo'])} Instanzen, die Interpolation {_thou(solved['bb_regret_1'])} (zwei Löseraufrufe je Schritt und Fall, das sind 600.000; dazu kommt die einmalige Lösung auf den "
    "Trainingskosten, die nur der Beobachtung des Trainingsverlusts dient). Das ist eine Zählung, keine Zeitmessung."
)

st.markdown("---")

# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: warum? Der Gradient an der Wahrheit")
st.markdown(
    "Gesetzt, das Modell trifft **genau die wahren Mittelkosten** $m(x)$. Dann sollte ein Verlustgradient im **Mittel über das Rauschen null** sein: das Modell hat nichts mehr zu lernen. "
    "Gemessen wird für 60 Merkmalsvektoren mit je 400 Rauschziehungen der Betrag des Mittels des Gradienten geteilt durch die mittlere Größe der einzelnen Gradienten (0 = im Mittel null, 1 = alle Ziehungen zeigen in dieselbe Richtung). "
    "Fenchel-Young (Gauß-Störung, eine Störung je Fall) ist zum Vergleich mitgerechnet; ich habe es aus dem Gedächtnis nach Abstract-Ebene nachgebaut und prüfe es hier nicht gegen die Quelle."
)
live_bias = _bias_live(float(noise), int(seed))
bias_bands = {e: {k: v["median"] for k, v in B.bias(e)["summary"].items()} for e in C.NOISE_CHOICES}
st.plotly_chart(build_bias_figure(bias_bands), width="stretch", key="bias_chart")
table3 = pd.DataFrame([
    {"Verfahren": name, "Diese Welt (Zufalls-Seed)": de(live_bias[name], 3),
     "Median über Seeds, Rauschstufe 0": de(bias_bands[0.0][name], 3), "Median über Seeds, Rauschstufe 0,5": de(bias_bands[0.5][name], 3)} for name in bias_bands[0.0]
])
st.dataframe(table3, hide_index=True, width="stretch")
kind, text = verdict_bias(float(noise), bias_bands[float(noise)], n_seeds)
getattr(st, kind)(text)
st.caption(
    "Deutung (nicht bewiesen): Der Gradient der Interpolation an der Wahrheit ist (Regret-Verlust) der Mittelwert von $w(m + \\lambda c) - w(m)$ über die Rauschziehungen $c$, geteilt durch λ; ein Mittel über Wege, die zu verrauschten Kosten gewählt wurden, "
    "ist nicht der Weg zu den mittleren Kosten. Setzt man in $\\mathrm{d}L/\\mathrm{d}y$ die mittleren Kosten statt der beobachteten ein (Experiment 4), ist er an der Wahrheit exakt null. "
    "Bei SPO+ ist der Mittelwert an der Wahrheit bei symmetrischem Rauschen ebenfalls null (zu jeder Ziehung gehört eine gespiegelte mit entgegengesetztem Gradienten); die gemessenen 0,05 sind die Streuung der 400 Ziehungen."
)

st.markdown("---")

# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: anderer Verlust, anderes Ziel")
st.markdown(
    f"Zwei Verluste: der **Regret-Verlust** ($\\mathrm{{d}}L/\\mathrm{{d}}y = c$) und der **Hamming-Abstand** zum Zielweg $y^*$ ($\\mathrm{{d}}L/\\mathrm{{d}}y = 1 - 2y^*$; diesen Verlust nutzt nach dem Stand meiner Websuche die Quelle). "
    f"Der Zielweg ist **verrauscht**, wenn er der beste Weg zu den beobachteten Kosten ist; als **Zusatzmessung** dieser Demo ist er **sauber** (bester Weg zu den mittleren Kosten) bzw. der Regret-Verlust nimmt die mittleren Kosten statt der beobachteten. "
    f"Sauber geht nur in der Simulation: im Alltag kennt man die mittleren Kosten nicht. Median über {n_seeds} Zufalls-Seeds."
)
rows4 = {e: loss_rows(e) for e in C.NOISE_CHOICES}
st.plotly_chart(build_loss_figure(rows4), width="stretch", key="loss_chart")
table4 = pd.DataFrame([
    {"Verfahren": r0["Verfahren"], "Ziel": {"KQ": "–", "verrauscht": "verrauscht", "sauber": "sauber (Zusatz)"}[r0["gruppe"]],
     "Rauschstufe 0": _range(r0), "Rauschstufe 0,5": _range(r5),
     f"Rauschstufe 0,5: schlechter als KQ (von {n_seeds})": "–" if r5["schlechter"] is None else str(r5["schlechter"])}
    for r0, r5 in zip(rows4[0.0], rows4[0.5])
])
st.dataframe(table4, hide_index=True, width="stretch")
cur = rows4[float(noise)]
kq_med = next(r["median"] for r in cur if r["key"] == "kq")
kind, text = verdict_loss(float(noise), kq_med, [r["median"] for r in cur if r["gruppe"] == "verrauscht"], [r["median"] for r in cur if r["gruppe"] == "sauber"], n_seeds)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
_c5 = B.cell(0.5)["summary"]
_noisy5 = [B.median(0.5, k) for k in ("bb_regret_0.3", "bb_regret_1", "bb_regret_10", "bb_regret_100", "bb_hamming_0.3", "bb_hamming_1", "bb_hamming_10")]
_clean5 = [B.median(0.5, k) for k in ("bb_regret_clean_1", "bb_regret_clean_10", "bb_hamming_clean_0.3", "bb_hamming_clean_1", "bb_hamming_clean_10")]
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist |
|---|---|
| **Die Ziele sind sauber** (der Zielweg bzw. dL/dy kennt keine Rauschziehung) | Mit verrauschten Zielen liegt die Interpolation bei Rauschstufe 0,5 im Median bei {de(min(_noisy5))} bis {de(max(_noisy5))} % gegen {de(_c5['kq']['median'])} % für KQ ({n_seeds} Zufalls-Seeds, Grad {C.DEG}, n = {C.N_TRAIN}); mit sauberem Ziel (nur in der Simulation bekannt) bei {de(min(_clean5))} bis {de(max(_clean5))} %. Ob die Experimente der Quelle mit sauberen Zielwegen arbeiten, habe ich nicht gelesen. |
| **Ein passendes λ** | Ohne Rauschen liegt λ = 1 bei {de(B.median(0.0, 'bb_regret_1'))} %, λ = 10 und 100 bei {de(B.median(0.0, 'bb_regret_10'))} und {de(B.median(0.0, 'bb_regret_100'))} % (SPO+ {de(B.median(0.0, 'spo'))} %, KQ {de(B.median(0.0, 'kq'))} %): ein großes λ macht die Interpolation gröber. Ein λ unterhalb der Kreuzung liefert den Gradienten null (Experiment 1). λ wurde nicht abgestimmt. |
| **Genug Rechenbudget** | Die Interpolation braucht zwei Löseraufrufe je Schritt und Fall. Die Zählung ist belegt; eine Zeitaussage für teure Löser (etwa ein gemischt-ganzzahliges Programm) ist hier nicht gemessen, der Gitterlöser ist billig. |
| **Eine Welt, ein Gitter** | Nur 5 × 5, Grad {C.DEG}, ein lineares Modell, ein multiplikatives gleichverteiltes Rauschen, Adam mit fester Lernrate 0,05, Parameter nach dem letzten Schritt. Aussagen über andere Gitter, Modelle, Rauschformen oder Trainer sind nicht gemessen. |
| **Das Verfahren der Quelle** | Vlastelica u. a. (ICLR 2020) habe ich selbst nicht im Volltext gelesen. Der Befund heißt „in meinem Nachbau mit verrauschten Zielen“, nicht „das Verfahren scheitert“. |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben sind Fälle $(x_i, c_i)$ mit Merkmalen $x_i \in \mathbb{R}^p$ und beobachteten Bogenkosten $c_i \in \mathbb{R}^d$ und ein Löser, der zu Kosten $c$ einen kürzesten Weg $w(c)$ liefert (Minimierung). Gesucht ist ein lineares Modell $\hat c(x) = \theta^\top (1, x)$, dessen gewählter Weg $w(\hat c(x))$ unter den wahren Kosten möglichst gut ist. Der Löser ist eine Black Box: nur $w(\cdot)$ darf aufgerufen werden.

| Symbol | Bedeutung |
|---|---|
| $p = 5$, $d = 40$ | Zahl der Merkmale, Zahl der Bögen im 5 × 5 Gitter |
| $c$, $\hat c$ | beobachtete (verrauschte) bzw. vorhergesagte Bogenkosten |
| $m(x)$ | wahre mittlere Kosten bei Merkmalen $x$ |
| $w(c)$ | der Weg (0/1-Vektor über die Bögen), den der Löser bei Kosten $c$ wählt; bei Gleichstand rechts vor unten |
| $w^*(c) = w(c)$ | der beste Weg unter den beobachteten Kosten |
| $L(y)$, $\mathrm{d}L/\mathrm{d}y$ | Verlust der Entscheidung $y$ und seine Ableitung nach $y$ |
| $\lambda > 0$ | Verschiebungsparameter der Interpolation |
| $\hat c_\lambda$ | verschobene Kosten $\hat c + \lambda\, \mathrm{d}L/\mathrm{d}y$ |
| $e$ | Rauschstufe |

**Der Gradient der Interpolation.** Die Funktion $\hat c \mapsto w(\hat c)$ ist stückweise konstant. Die Interpolation zwischen ihr und einer linearen Funktion hat den Gradienten

$$g(\hat c) = \frac{w(\hat c_\lambda) - w(\hat c)}{\lambda}, \qquad \hat c_\lambda = \hat c + \lambda\,\frac{\mathrm{d}L}{\mathrm{d}y}\bigl(w(\hat c)\bigr).$$

Er braucht zwei Löseraufrufe. Ändert sich der Weg nicht, ist $g = 0$; beide Lösungen sind Wege mit gleich vielen Bögen, die Komponenten von $g$ summieren sich zu null. Für ein kleines $\lambda$ ist $g$ fast überall null, für ein großes $\lambda$ ist die Interpolation grob (der Betrag von $g$ fällt wie $1/\lambda$).

**Regret-Verlust:** $L(y) = c^\top y$, also $\mathrm{d}L/\mathrm{d}y = c$ und $\hat c_\lambda = \hat c + \lambda c$. **Hamming-Verlust:** $L(y) = \sum_j |y_j - y^*_j| = \sum_j y_j (1 - 2 y^*_j) + \sum_j y^*_j$ für 0/1-Vektoren, also $\mathrm{d}L/\mathrm{d}y = 1 - 2y^*$.

**Training.** Für das lineare Modell folgt der Gradient nach $\theta$ aus der Kettenregel: $\frac{1}{n}\sum_i (1, x_i)\, g_i^\top$. Adam, Lernrate 0,05, 300 Schritte, Start bei der Kleinste-Quadrate-Lösung, Parameter nach dem letzten Schritt.

**Handbeispiel.** Zwei Wege mit je einem Bogen, $c = (3, 5)$, $\hat c = (4, 3)$: $w(\hat c) = (0, 1)$. $\lambda = 0{,}25$: $\hat c_\lambda = (4{,}75; 4{,}25)$, Wahl wie zuvor, $g = (0, 0)$. $\lambda = 0{,}5$: $\hat c_\lambda = (5{,}5; 5{,}5)$, Gleichstand, die feste Regel wählt Weg 1, $g = ((1,0) - (0,1))/0{,}5 = (2, -2)$. $\lambda = 1$: $\hat c_\lambda = (7, 8)$, $g = (1, -1)$. $\lambda = 10$: $g = (0{,}1; -0{,}1)$.

**Messung an der Wahrheit.** Bei $\hat c = m(x)$ und Rauschziehungen $c = m(x) \odot \varepsilon$ ist das Maß $\|\mathbb{E}\,g\| / \mathbb{E}\|g\|$. Für SPO+ (Teilgradient $2\,(w^*(c) - w(2\hat c - c))$) ist es bei symmetrischem Rauschen null, weil $c$ und $2m - c$ gleich verteilt sind.

**Literatur.** Vlastelica, Paulus, Musil, Martius und Rolínek, *Differentiation of Blackbox Combinatorial Solvers*, ICLR 2020 (arXiv 1912.02175): Gradient der Interpolation $-\frac{1}{\lambda}\,[\,y(w) - y_\lambda(w)\,]$ mit $y_\lambda$ aus dem Löser auf verändertem Eingang; λ wägt die Treue zur stückweise konstanten Funktion gegen die Aussagekraft des Gradienten ab; die Rückwärtsrechnung kostet so viel wie die Vorwärtsrechnung (Haupttext, nicht der arXiv-Abstract; Satzbedingungen und Experimente nicht im Primärtext geprüft). Elmachtoub und Grigas, *Smart „Predict, then Optimize“*, Management Science 68(1), 2022: SPO+ (Abstract-Ebene). Das Gitter und die Kostenformel sind aus dem Gedächtnis nach dem Schema dieser Literatur nachgebaut, nicht gegen sie geprüft.
        """
    )

sync_query_params({
    "noise_slider": float(noise), "seed_input": int(seed), "lam_slider": float(lam), "c1_slider": float(c1), "c2_slider": float(c2),
})

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
