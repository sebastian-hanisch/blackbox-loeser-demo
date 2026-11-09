"""Kern der Demo: die Black-Box-Interpolation, also ein Gradient aus zwei Löseraufrufen (keine Streamlit-Abhängigkeit, nur numpy).

Der Löser wird nicht geöffnet. Man löst zweimal: einmal mit der Vorhersage ch, einmal mit einer verschobenen Vorhersage ch_lam = ch + lam * dL/dy, wobei L(y) der Verlust
der Entscheidung y ist (dL/dy: wie sich der Verlust ändert, wenn man die Entscheidung ändert). Die Differenz der beiden Lösungen, geteilt durch lam, ist der "Gradient":

    g(ch) = (w(ch + lam dL/dy) - w(ch)) / lam                    (Minimierung; nach ch)

Das ist der Gradient der Interpolation zwischen der stückweise konstanten Funktion ch -> w(ch) und einer linearen Funktion; lam wägt die Treue zur stückweise konstanten
Funktion (kleines lam) gegen die Aussagekraft des Gradienten (großes lam) ab. Herkunft der Form: Vlastelica, Paulus, Musil, Martius, Rolínek, "Differentiation of
Blackbox Combinatorial Solvers", ICLR 2020 (arXiv 1912.02175), Websuche und Nachprüfung am Haupttext 2026-11-09 (Algorithmus 1; im arXiv-Abstract steht die Formel nicht): dort steht der Gradient -(1/lam) [y(w) - y_lam(w)] mit
y_lam aus dem Löser auf verändertem Eingang. Das Vorzeichen ist hier am rauschfreien Fall geprüft (das Training mit diesem Gradienten verbessert ohne Rauschen gegenüber Kleinsten Quadraten, siehe
README), die Satzbedingungen und die Experimente der Quelle sind nicht im Primärtext geprüft.

Zwei Verluste L stehen zur Wahl:

    Regret-Verlust   L(y) = c^T y,                   dL/dy = c            (c: beobachtete Kosten; der Regret des Weges y ist c^T y - z*(c))
    Hamming-Verlust  L(y) = sum |y - y*| (y, y* in {0,1}^d), dL/dy = 1 - 2 y*   (y*: Zielweg; Abstand zum Zielweg, wie ihn die Quelle nach Websuche-Stand verwendet)

Der Zielweg y* ist "verrauscht", wenn er der beste Weg zu den beobachteten Kosten c ist, und "sauber", wenn er der beste Weg zu den mittleren Kosten m(x) ist (nur in
Simulationen bekannt, Zusatzmessung dieser Demo).

Gleichstände im Löser entscheiden nach der festen Regel des Gitters ("rechts vor unten") bzw. der Wegliste (`PathSet`: der erste Weg der Liste). Für lam -> 0 oder bei
unverändertem Weg ist g = 0 (beide Lösungen gleich).
"""

import numpy as np

from bbl_training import add_intercept, adam_minimize, fit_least_squares


def dldy_regret(C: np.ndarray) -> np.ndarray:
    """dL/dy des Regret-Verlustes: die beobachteten Kosten."""
    return np.atleast_2d(np.asarray(C, float))


def dldy_hamming(Ystar: np.ndarray) -> np.ndarray:
    """dL/dy des Hamming-Verlustes zum Zielweg y*: 1 - 2 y* (die Summe |y - y*| ist für 0/1-Vektoren sum y (1 - 2 y*) + sum y*)."""
    return 1.0 - 2.0 * np.atleast_2d(np.asarray(Ystar, float))


def hamming_distance(W: np.ndarray, Ystar: np.ndarray) -> np.ndarray:
    """Hamming-Abstand je Fall: Zahl der Bögen, in denen die beiden Wege verschieden sind."""
    return np.sum(np.abs(np.asarray(W, float) - np.asarray(Ystar, float)), axis=1)


def bb_parts(solver, Chat: np.ndarray, dldy: np.ndarray, lam: float):
    """Gradient der Interpolation je Fall (n x d) mit beiden Lösungen: (g, w(ch), w(ch + lam dL/dy)). Zwei Löseraufrufe auf allen Fällen."""
    if not lam > 0:
        raise ValueError("lam muss größer als 0 sein")
    Chat = np.atleast_2d(np.asarray(Chat, float))
    W1, _ = solver.solve(Chat)
    W2, _ = solver.solve(Chat + lam * np.asarray(dldy, float))
    return (W2 - W1) / lam, W1, W2


def bb_gradient(solver, Chat: np.ndarray, dldy: np.ndarray, lam: float) -> np.ndarray:
    """Nur der Gradient der Interpolation (n x d)."""
    return bb_parts(solver, Chat, dldy, lam)[0]


def bb_objective(solver, X: np.ndarray, C: np.ndarray, lam: float, loss: str = "regret", Ystar: np.ndarray | None = None, dldy_costs: np.ndarray | None = None):
    """Zielfunktion des Trainings als `loss_and_grad(coef)` für `adam_minimize`: der Gradient nach den Parametern ist X_a^T g / n (Kettenregel über das lineare Modell).

    Der zurückgegebene "Verlust" ist nur die Beobachtungsgröße des Trainings (kein Verlust, dessen Gradient g wäre): der mittlere Regret der gewählten Wege auf den Trainingsfällen
    (loss = "regret") bzw. der mittlere Hamming-Abstand zum Zielweg (loss = "hamming"). Bei "hamming" ohne `Ystar` ist der Zielweg der beste Weg zu C (verrauschter Zielweg);
    bei "regret" ersetzt `dldy_costs` (z. B. die mittleren Kosten) die beobachteten Kosten in dL/dy (saubere Zusatzmessung)."""
    Xa = add_intercept(X)
    C = np.asarray(C, float)
    n = len(Xa)
    if loss == "regret":
        dldy = dldy_regret(C if dldy_costs is None else dldy_costs)   # sauber: die mittleren Kosten statt der beobachteten
        _, zstar = solver.solve(C)
        target = None
    elif loss == "hamming":
        target = solver.solve(C)[0] if Ystar is None else np.asarray(Ystar, float)
        dldy = dldy_hamming(target)
        zstar = None
    else:
        raise ValueError(f"loss muss \"regret\" oder \"hamming\" sein, nicht {loss!r}")

    def loss_and_grad(coef):
        g, W1, _ = bb_parts(solver, Xa @ coef, dldy, lam)
        observed = np.mean(np.sum(C * W1, axis=1) - zstar) if loss == "regret" else np.mean(hamming_distance(W1, target))
        return float(observed), Xa.T @ g / n

    return loss_and_grad


def train_bb(solver, X: np.ndarray, C: np.ndarray, lam: float, loss: str = "regret", Ystar: np.ndarray | None = None,
             steps: int = 300, lr: float = 0.05, init=None, snapshot_steps=(), dldy_costs: np.ndarray | None = None):
    """Training mit dem Gradienten der Interpolation: Adam, Vollbatch, Start bei Kleinsten Quadraten (oder `init`), zurückgegeben wird der Parameter nach dem letzten Schritt
    (`select="last"`, so wie die Vormessung). Je Schritt zwei Löseraufrufe auf allen Trainingsfällen."""
    coef0 = fit_least_squares(X, C) if init is None else np.array(init, dtype=float)
    return adam_minimize(bb_objective(solver, X, C, lam, loss, Ystar, dldy_costs), coef0, steps, lr, snapshot_steps, select="last")
