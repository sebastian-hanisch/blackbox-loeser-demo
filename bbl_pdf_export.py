"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import bbl_bands as B
import bbl_constants as C
from bbl_evaluation import LAM_KEYS, Settings, de, de_g, hand_numbers, noise_rows, verdict_noise

_REPLACE = {"→": "->", "–": "-", "—": "-", "ĉ": "c^", "−": "-", "≥": ">=", "·": "*", "×": "x", "σ": "sigma", "λ": "lambda", "€": "EUR", "²": "^2", "ˆ": "^", "₁": "1", "₂": "2", "ₗ": "l"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, run) -> bytes:
    """`run`: das `WorldRun` zu den Einstellungen (Live-Lauf einer Instanz)."""
    h = hand_numbers(settings.c1, settings.c2, settings.lam)
    rows = noise_rows(settings.noise)
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Der Löser als Black Box: Interpolation statt Ableitung"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"Welt: 5 x 5 Gitter, 5 Merkmale, Grad {C.DEG}, Rauschstufe {de_g(settings.noise)}, {C.N_TRAIN} Trainingsfälle, Zufalls-Seed {settings.seed}, "
        f"{run.spec.n_test} Testfälle. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Diese Instanz (ein Zufalls-Seed), Regret-Verlust"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    parts = [f"Orakel {de(100 * run.regret['orakel'])} %", f"KQ {de(100 * run.regret['kq'])} %", f"SPO+ {de(100 * run.regret['spo'])} %"]
    parts += [f"Interpolation (lambda = {de_g(lam)}) {de(100 * run.regret[key])} %" for lam, key in LAM_KEYS.items()]
    pdf.multi_cell(0, 5, _s("Normierter Regret: " + ", ".join(parts) + "."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s(f"Median über {B.n_seeds()} Zufalls-Seeds (vorgerechnet, 300 Adam-Schritte)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in rows:
        extra = "" if r["schlechter"] is None else f", schlechter als KQ in {r['schlechter']} von {B.n_seeds()} Seeds"
        pdf.multi_cell(0, 5, _s(f"{r['Verfahren']}: Regret {de(r['median'])} % (Spanne {de(r['min'])} bis {de(r['max'])} %){extra}"), new_x="LMARGIN", new_y="NEXT")
    bb = {lam: B.median(settings.noise, key) for lam, key in LAM_KEYS.items()}
    worse = {lam: B.count_worse(settings.noise, key) for lam, key in LAM_KEYS.items()}
    kind, text = verdict_noise(settings.noise, B.median(settings.noise, "kq"), B.median(settings.noise, "spo"), bb, worse, B.n_seeds())
    pdf.ln(1)
    pdf.multi_cell(0, 5, _s(text.replace("**", "")), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Handbeispiel: zwei Wege"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"Wahre Kosten c = (3, 5), Vorhersage ĉ = ({de(settings.c1)}, {de(settings.c2)}), lambda = {de_g(h['lam'])}: unter ĉ wird Weg {h['chosen']} gewählt, unter ĉ_lambda = "
        f"({de(h['chat_lam'][0], 2)}, {de(h['chat_lam'][1], 2)}) Weg {h['chosen_lam']}; Gradient ({de(h['gradient'][0], 2)}; {de(h['gradient'][1], 2)})."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Der Löser als Black Box: Interpolation statt Ableitung, Negativbefund im eigenen Aufbau mit verrauschten Zielen. Quelle: blackbox-loeser-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
