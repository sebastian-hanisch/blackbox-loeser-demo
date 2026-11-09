"""Plotly-Abbildungen (alle Achsen gesperrt, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import bbl_constants as C
from bbl_evaluation import de, de_g

_LOCK = dict(fixedrange=True)


def _lock(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(**_LOCK)
    fig.update_yaxes(**_LOCK)
    return fig


# --- Experiment 1: die Ebene der Kosten und der Gradient gegen λ --------------------------------------------------------------------------

PLANE_MAX = 12.0   # Achsenlänge der Ebene; ĉ_λ außerhalb wird nicht gezeichnet, sondern vermerkt


def build_hand_figure(h: dict, curve: dict, chat: tuple[float, float]) -> go.Figure:
    """Links: die Ebene der Kosten (ĉ₁, ĉ₂); die Diagonale trennt Weg 1 (unten rechts liegt Weg 2 billiger) von Weg 2; Pfeil von ĉ nach ĉ_λ = ĉ + λ c.
    Rechts: erste Komponente des Gradienten gegen λ (logarithmisch), Marke beim gewählten λ."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Wo liegen ĉ und ĉ_λ? (Ebene der Kosten)", "Gradient (erste Komponente) gegen λ"), horizontal_spacing=0.17)
    m = PLANE_MAX
    fig.add_trace(go.Scatter(x=[0, m, m], y=[0, m, 0], fill="toself", fillcolor=C.COLOR_REGION_2, line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, m, 0], y=[0, m, m], fill="toself", fillcolor=C.COLOR_REGION_1, line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", line=dict(color="#555", width=1, dash="dot"), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_annotation(x=0.68 * m, y=0.2 * m, text="hier wählt der Löser Weg 2", showarrow=False, font=dict(size=11), row=1, col=1)
    fig.add_annotation(x=0.25 * m, y=0.8 * m, text="hier wählt er Weg 1", showarrow=False, font=dict(size=11), row=1, col=1)
    c = C.HAND_C
    fig.add_trace(go.Scatter(x=[c[0]], y=[c[1]], mode="markers", marker=dict(color="gray", size=9), name="wahre Kosten c", hovertemplate="c = (%{x}; %{y})<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[chat[0]], y=[chat[1]], mode="markers", marker=dict(color="black", size=11, symbol="diamond"), name="Vorhersage ĉ", hovertemplate="ĉ = (%{x}; %{y})<extra></extra>"), row=1, col=1)
    lx, ly = h["chat_lam"]
    inside = 0 <= lx <= m and 0 <= ly <= m
    if inside:
        fig.add_trace(go.Scatter(x=[lx], y=[ly], mode="markers", marker=dict(color=C.COLOR_BB[1.0], size=11, symbol="x"), name="verschoben: ĉ_λ", hovertemplate="ĉ_λ = (%{x}; %{y})<extra></extra>"), row=1, col=1)
        fig.add_annotation(x=lx, y=ly, ax=chat[0], ay=chat[1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor=C.COLOR_BB[1.0], text="")
    else:
        fig.add_annotation(x=0.5 * m, y=0.5 * m + 0.9, text=f"ĉ_λ = ({de(lx)}; {de(ly)}) liegt außerhalb", showarrow=False, font=dict(size=11, color=C.COLOR_BB[1.0]), row=1, col=1)
    fig.update_xaxes(title_text="Kosten Weg 1", range=[0, m], row=1, col=1)
    fig.update_yaxes(title_text="Kosten Weg 2", range=[0, m], row=1, col=1)

    lam, g1 = curve["lam"], curve["g1"]
    fig.add_trace(go.Scatter(x=lam, y=g1, mode="lines", line=dict(color=C.COLOR_BB[1.0], width=3), showlegend=False, hovertemplate="λ = %{x:.3g}: %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[h["lam"]], y=[h["gradient"][0]], mode="markers", marker=dict(color="black", size=11, symbol="diamond"), showlegend=False,
                             hovertemplate="gewähltes λ = %{x}: %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="λ (logarithmisch)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Gradient (1. Komponente)", row=1, col=2)
    fig.update_layout(height=360, margin=dict(l=40, r=20, t=50, b=50), legend=dict(orientation="h", y=-0.22))
    return _lock(fig)


# --- Experiment 2: Lernkurven --------------------------------------------------------------------------------------------------------------

def build_noise_figure(curves: dict) -> go.Figure:
    """Testregret (Median über die Zufalls-Seeds) nach t Adam-Schritten, je Rauschstufe ein Feld. `curves`: {Rauschstufe: {"kq": Wert, "orakel": Wert, Schlüssel: {Schritt: Wert}}}."""
    noises = sorted(curves)
    fig = make_subplots(rows=1, cols=len(noises), subplot_titles=[f"Rauschstufe {de_g(e)}" for e in noises], horizontal_spacing=0.1)
    series = [("spo", "SPO+", C.COLOR_SPO)] + [(f"bb_regret_{lam:g}", f"Interpolation, λ = {de_g(lam)}", C.COLOR_BB[lam]) for lam in C.LAMBDA_LIVE]
    for col, e in enumerate(noises, start=1):
        cv = curves[e]
        steps = sorted(cv["spo"])
        xs_all = [steps[0], steps[-1]]
        fig.add_trace(go.Scatter(x=xs_all, y=[cv["kq"]] * 2, mode="lines", line=dict(color=C.COLOR_KQ, width=2, dash="dash"), name="KQ", showlegend=col == 1,
                                 hovertemplate="KQ: %{y:.2f} %<extra></extra>"), row=1, col=col)
        fig.add_trace(go.Scatter(x=xs_all, y=[cv["orakel"]] * 2, mode="lines", line=dict(color=C.COLOR_ORACLE, width=1, dash="dot"), name="Orakel", showlegend=col == 1,
                                 hovertemplate="Orakel: %{y:.2f} %<extra></extra>"), row=1, col=col)
        for key, name, color in series:
            fig.add_trace(go.Scatter(x=steps, y=[cv[key][s] for s in steps], mode="lines+markers", line=dict(color=color, width=2), name=name, showlegend=col == 1,
                                     hovertemplate=name + ", Schritt %{x}: %{y:.2f} %<extra></extra>"), row=1, col=col)
        fig.update_xaxes(title_text="Adam-Schritte", row=1, col=col)
        fig.update_yaxes(title_text="Regret auf den Testfällen (%)" if col == 1 else None, rangemode="tozero", row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=40, r=20, t=50, b=60), legend=dict(orientation="h", y=-0.25))
    return _lock(fig)


# --- Experiment 3: der Gradient an der Wahrheit -------------------------------------------------------------------------------------------------

def build_bias_figure(values: dict) -> go.Figure:
    """Verhältnis |Mittel des Gradienten| / Mittel von |Gradient| an der Wahrheit (Median über Seeds): je Verfahren ein Balken je Rauschstufe. `values`: {Rauschstufe: {Name: Wert}}."""
    fig = go.Figure()
    colors = {0.0: "#9ecae1", 0.5: "#c0392b"}
    for e in sorted(values):
        names = list(values[e])
        fig.add_trace(go.Bar(x=names, y=[values[e][n] for n in names], name=f"Rauschstufe {de_g(e)}", marker_color=colors.get(e, "#777"),
                             hovertemplate="%{x}, Rauschstufe " + de_g(e) + ": %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group", height=360, margin=dict(l=40, r=20, t=20, b=90), legend=dict(orientation="h", y=-0.35))
    fig.update_yaxes(title_text="Mittel des Gradienten / mittlere Größe", range=[0, 1])
    return _lock(fig)


# --- Experiment 4: Verlust und Zielweg ---------------------------------------------------------------------------------------------------------

def build_loss_figure(rows_by_noise: dict) -> go.Figure:
    """Regret je Verfahren (Median über Seeds, Balken: Minimum bis Maximum), je Rauschstufe eine Spur. `rows_by_noise`: {Rauschstufe: Zeilen von `loss_rows`}."""
    fig = go.Figure()
    colors = {0.0: "#4c78a8", 0.5: "#c0392b"}
    for e in sorted(rows_by_noise):
        rows = rows_by_noise[e]
        fig.add_trace(go.Scatter(
            x=[r["median"] for r in rows], y=[r["Verfahren"] for r in rows], mode="markers", name=f"Rauschstufe {de_g(e)}", marker=dict(color=colors.get(e, "#777"), size=9),
            error_x=dict(type="data", symmetric=False, array=[r["max"] - r["median"] for r in rows], arrayminus=[r["median"] - r["min"] for r in rows], thickness=1.2),
            hovertemplate="%{y}, Rauschstufe " + de_g(e) + ": %{x:.2f} %<extra></extra>"))
    fig.update_layout(height=520, margin=dict(l=10, r=20, t=20, b=60), legend=dict(orientation="h", y=-0.12))
    fig.update_xaxes(title_text="Regret auf den Testfällen (%), Median über die Zufalls-Seeds, Balken: Minimum bis Maximum", rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    return _lock(fig)
