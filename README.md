# Der Löser als Black Box: Interpolation statt Ableitung – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-blackbox-loeser-demo.streamlit.app/)**

Demo der Reihe **Entscheidungsorientiertes Lernen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie setzt `spo-kuerzester-weg-demo` fort (dasselbe 5 × 5 Gitter mit 40 Bögen und 70 Wegen, derselbe Löser, dieselbe Kostenwelt; Code kopiert, nicht importiert) und fragt, was passiert, wenn man den **Löser nicht öffnen** darf: Man kennt nur die Funktion „Kosten → Weg“. Ein **Gradient** entsteht dann aus einem zweiten Löseraufruf mit verschobenen Kosten: die Differenz der beiden Lösungen, geteilt durch den Verschiebungsparameter **λ**, ist der Gradient einer Interpolation. Das Modell (linear, 240 Parameter) wird damit wie bei SPO+ mit Adam trainiert, der Löser läuft in jeder Runde zweimal.

## Warum dieses Problem

Die Entscheidung eines Lösers ist als Funktion der Kosten stückweise konstant, ihr Gradient ist fast überall null. SPO+ umgeht das mit einem konvexen Ersatzverlust, der den Löser kennt. Die Black-Box-Interpolation braucht nur den Löseraufruf. Die Kernfrage der Demo: **Trägt das auch, wenn die Zielwerte verrauscht sind?** Antwort in meinem Nachbau: **nein**, und in diesem Aufbau liegt die Ursache nach der Zusatzmessung im Rauschen der Ziele, nicht am Verlust (Regret, Hamming) und nicht an λ (0,3 bis 100).

- **Ohne Rauschen funktioniert es:** Rauschstufe 0, Grad 6, n = 1000: KQ 1,96 % Regret, SPO+ 0,51 %, Interpolation mit λ = 1 0,57 %, mit λ = 10 und mit λ = 100 1,02 bzw. 1,01 % (Median über 20 Zufalls-Seeds; in jedem Seed besser als KQ). Bei λ = 10 und 100 liegt der Regret fast doppelt so hoch wie bei λ = 1 und steigt nach dem Tiefpunkt (bei 25 bzw. 50 Schritten) wieder an (0,64 und 0,76 % nach 25 Schritten).
- **Mit Rauschen scheitert es in diesem Aufbau:** Rauschstufe 0,5: KQ 13,1 %, SPO+ 12,3 %, Interpolation bei λ = 0,3 / 1 / 10 / 100 bei 18,6 / 20,8 / 20,4 / 21,6 % (Median über 20 Zufalls-Seeds, Orakel 10,5 %). In **allen 20 Seeds** ist jede dieser vier Varianten schlechter als KQ.
- **Warum, gemessen:** Trifft die Vorhersage genau die wahren Mittelkosten, ist der Mittelwert des Gradienten über das Rauschen bei SPO+ nahe null (0,047 seiner mittleren Größe, mit mehr Rauschziehungen kleiner), bei der Interpolation aber bei 0,713 (λ = 1) und 0,695 (λ = 10): im Mittel ist die Wahrheit hier kein Ruhepunkt des Trainings (die Lernkurve entfernt sich von KQ).
- **Nicht am Verlust, nicht an λ:** Auch mit dem Hamming-Abstand zum optimalen Weg (der Verlust, den die Quelle nach dem Stand meiner Websuche verwendet) liegt der Regret bei Rauschstufe 0,5 bei 21,0 / 21,9 / 22,5 % (λ = 0,3 / 1 / 10).
- **Am Rauschen der Ziele, gemessen als Zusatz:** Nimmt man stattdessen den Zielweg zu den **mittleren** Kosten (oder die mittleren Kosten in dL/dy), die man nur in der Simulation kennt, liegt dieselbe Interpolation bei 11,1 bis 11,6 % und ist in 19 bis 20 von 20 Seeds besser als KQ. Das ist kein Verfahren für den Alltag, sondern in dieser Simulation ein Beleg dafür, dass der Mangel am verrauschten Ziel liegt.

Der Satz der Demo ist eng: **im eigenen Aufbau mit verrauschten Zielen**, in diesem Gitter, mit diesem Modell und Trainer. Er sagt nicht, dass das Verfahren von Vlastelica u. a. scheitert; deren Experimente habe ich nicht gelesen.

## Modell

**Welt** (aus `spo-kuerzester-weg-demo` übernommen; aus dem Gedächtnis nach dem Schema der SPO-Literatur nachgebaut, nicht gegen sie geprüft): Merkmale $x \sim \mathcal N(0, I_5)$; eine Gewichtsmatrix $B \in \{0,1\}^{40 \times 5}$ mit fairen Münzwürfen; mittlere Kosten $m_j(x) = ((Bx)_j/\sqrt{5} + 3)^{6} / 3{,}5^{6} + 1$ (Grad 6, fest); beobachtete Kosten $c_j = m_j(x)\,\varepsilon_j$ mit $\varepsilon_j \sim U[1 - e, 1 + e]$ und der Rauschstufe $e$ (0 oder 0,5). Das **Orakel** kennt $m(x)$; bei $e = 0$ hat es Regret 0. Je Welt 1000 Trainings- und 1000 Testfälle.

**Löser.** Rückwärtsrechnung über die 25 Knoten, vektorisiert über alle Fälle; Gleichstand: am Knoten „rechts“ vor „unten“. Bei der Wegliste des Handbeispiels gewinnt bei Gleichstand der erste Weg.

**Die Interpolation.** Mit dem Löser $w(\cdot)$ (minimiert), vorhergesagten Kosten $\hat c$ und dem Verlust $L(y)$ der Entscheidung $y$:

$$g(\hat c) = \frac{w(\hat c_\lambda) - w(\hat c)}{\lambda}, \qquad \hat c_\lambda = \hat c + \lambda\,\frac{\mathrm{d}L}{\mathrm{d}y}.$$

Regret-Verlust $L(y) = c^\top y$: $\mathrm{d}L/\mathrm{d}y = c$ (beobachtete Kosten). Hamming-Verlust $L(y) = \sum_j |y_j - y^*_j|$ zum Zielweg $y^*$: $\mathrm{d}L/\mathrm{d}y = 1 - 2y^*$. Der **Zielweg** ist **verrauscht**, wenn er der beste Weg zu den beobachteten Kosten ist, **sauber** (Zusatzmessung dieser Demo), wenn er der beste Weg zu den mittleren Kosten ist. Der Gradient nach den Parametern des linearen Modells ist $X_a^\top g / n$. Die Form des Gradienten steht bei Vlastelica, Paulus, Musil, Martius und Rolínek, *Differentiation of Blackbox Combinatorial Solvers*, ICLR 2020 (arXiv 1912.02175), im Haupttext (Algorithmus 1), nicht im arXiv-Abstract (Websuche und Nachprüfung am Haupttext 2026-11-09): $-\frac{1}{\lambda}[y(w) - y_\lambda(w)]$ mit $y_\lambda$ aus dem Löser auf verändertem Eingang; λ wägt die Treue zur stückweise konstanten Funktion gegen die Aussagekraft des Gradienten ab; die Rückwärtsrechnung kostet so viel wie die Vorwärtsrechnung. Das Vorzeichen ist hier am rauschfreien Fall geprüft (das Training senkt dort den Regret gegenüber KQ). **Den Volltext (Satzbedingungen, Experimente) habe ich selbst nicht gelesen.**

**Handbeispiel** (zwei Wege mit je einem Bogen, Regret-Verlust, wahre Kosten $c = (3, 5)$, Vorhersage $\hat c = (4, 3)$, gewählt wird Weg 2 = $(0, 1)$): $\lambda = 0{,}25$: $\hat c_\lambda = (4{,}75;\, 4{,}25)$, Wahl wie zuvor, Gradient $(0, 0)$. $\lambda = 0{,}5$: $\hat c_\lambda = (5{,}5;\, 5{,}5)$, **exakter Gleichstand**, die feste Regel „kleinster Weg“ wählt Weg 1, Gradient $((1,0) - (0,1))/0{,}5 = (2, -2)$. $\lambda = 1$: $\hat c_\lambda = (7, 8)$, Gradient $(1, -1)$. $\lambda = 10$: Gradient $(0{,}1;\, -0{,}1)$. Unterhalb der Kreuzung $\lambda^* = (\hat c_1 - \hat c_2)/(c_2 - c_1) = 0{,}5$ ist der Gradient null, darüber $(1, -1)/\lambda$: die Interpolation wird mit wachsendem λ gröber. Wählt $\hat c$ schon Weg 1, ist der Gradient für jedes $\lambda > 0$ null.

**Messung an der Wahrheit.** Bei $\hat c = m(x)$ und Rauschziehungen $c = m(x) \odot \varepsilon$ ist das Maß $\|\mathbb{E}\,g\| / \mathbb{E}\|g\|$ (Normen über die 40 Bögen; Mittel über 400 Ziehungen; Median über 60 Merkmalsvektoren je Welt): 0 heißt im Mittel null, 1 heißt, dass alle Ziehungen in dieselbe Richtung zeigen. Verglichen wird mit dem Teilgradienten von SPO+, $2\,(w^*(c) - w(2\hat c - c))$, und, nur als Bezug, mit einem Fenchel-Young-Gradienten mit Gauß-Störung ($w^*(c) - w(\hat c + \sigma z)$, eine Störung je Fall; aus dem Gedächtnis nach Abstract-Ebene nachgebaut, nicht gegen die Quelle geprüft).

## Methodik

- **Training.** Lineares Modell mit Achsenabschnitt (6 × 40 = 240 Parameter). KQ über `numpy.linalg.lstsq`. SPO+ und Interpolation mit **Adam**, Lernrate 0,05, Vollbatch, **300 Schritte**, Start bei der KQ-Lösung. SPO+ gibt den Parameter mit dem kleinsten Trainingsverlust zurück (wie in `spo-kuerzester-weg-demo`), die **Interpolation den Parameter nach dem letzten Schritt** (so wie die Vormessung). Der Löser läuft in jeder Runde auf allen Trainingsfällen: SPO+ einmal, die Interpolation zweimal.
- **Auswertung.** Je Rauschstufe die festen Zufalls-Seeds 0 bis 19, 1000 Testfälle, 5 Merkmale. Angegeben werden **Median und Spanne (Minimum bis Maximum)** über die Seeds, nie ein einzelner Seed. Die Bänder stehen vorgerechnet in `bbl_bands.json` und sind mit `python tools/measure_bands.py` reproduzierbar (einige Minuten; Zufall nur über den ganzzahligen SplitMix64 aus `bbl_rng.py`, nie Pythons `hash()`). Die App rechnet außerdem eine Welt live (KQ, SPO+, drei Interpolationsläufe) und die Messung an der Wahrheit für diese Welt.
- **Trainierte Zahlen sind plattformabhängig** (Adam, Gleitkomma, Löserentscheidungen bei knappen Unterschieden). Die Tests prüfen sie deshalb nur als Bänder und Ungleichungen; exakt geprüft werden nur Rechnungen ohne Training (Handbeispiel, Eigenschaften des Gradienten, Zählungen).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen dieses Abschnitts stammen aus `bbl_bands.json` (eigene Messung: lineares Modell, Grad 6, n = 1000, 300 Adam-Schritte, Zufalls-Seeds 0 bis 19, 1000 Testfälle, Regret in Prozent als Median [Minimum; Maximum]), wenn nichts anderes dabeisteht. Die Tabellen werden von `tests/readme_tables.py` aus der JSON-Datei erzeugt, und `tests/test_claims.py` prüft, dass jede Zeile wörtlich hier steht; die Aussagen darunter stehen als Bänder und Ungleichungen im selben Test.

### Regret je Verfahren, Rauschstufe 0

| Verfahren | Regret (Median [Minimum; Maximum]) | besser als KQ | schlechter als KQ |
|---|---|---|---|
| Orakel (kennt die mittleren Kosten) | 0,00 [0,00; 0,00] % | – | – |
| Kleinste Quadrate (KQ) | 1,96 [1,38; 3,00] % | – | – |
| SPO+ | 0,51 [0,36; 0,83] % | 20 von 20 | 0 von 20 |
| Regret-Verlust, λ = 0,3 | 0,56 [0,37; 0,80] % | 20 von 20 | 0 von 20 |
| Regret-Verlust, λ = 1 | 0,57 [0,38; 0,82] % | 20 von 20 | 0 von 20 |
| Regret-Verlust, λ = 10 | 1,02 [0,68; 1,97] % | 20 von 20 | 0 von 20 |
| Regret-Verlust, λ = 100 | 1,01 [0,52; 1,73] % | 20 von 20 | 0 von 20 |
| Hamming-Abstand, λ = 0,3 | 1,08 [0,54; 2,09] % | 18 von 20 | 2 von 20 |
| Hamming-Abstand, λ = 1 | 0,97 [0,70; 1,75] % | 20 von 20 | 0 von 20 |
| Hamming-Abstand, λ = 10 | 0,92 [0,73; 1,88] % | 20 von 20 | 0 von 20 |
| Regret-Verlust mit mittleren Kosten, λ = 1 | 0,57 [0,38; 0,82] % | 20 von 20 | 0 von 20 |
| Regret-Verlust mit mittleren Kosten, λ = 10 | 1,02 [0,68; 1,97] % | 20 von 20 | 0 von 20 |
| Hamming-Abstand, sauberer Zielweg, λ = 0,3 | 1,08 [0,54; 2,09] % | 18 von 20 | 2 von 20 |
| Hamming-Abstand, sauberer Zielweg, λ = 1 | 0,97 [0,70; 1,75] % | 20 von 20 | 0 von 20 |
| Hamming-Abstand, sauberer Zielweg, λ = 10 | 0,92 [0,73; 1,88] % | 20 von 20 | 0 von 20 |

### Regret je Verfahren, Rauschstufe 0,5

| Verfahren | Regret (Median [Minimum; Maximum]) | besser als KQ | schlechter als KQ |
|---|---|---|---|
| Orakel (kennt die mittleren Kosten) | 10,54 [9,31; 12,32] % | – | – |
| Kleinste Quadrate (KQ) | 13,08 [11,28; 14,48] % | – | – |
| SPO+ | 12,29 [10,72; 13,42] % | 20 von 20 | 0 von 20 |
| Regret-Verlust, λ = 0,3 | 18,61 [16,00; 22,66] % | 0 von 20 | 20 von 20 |
| Regret-Verlust, λ = 1 | 20,80 [16,80; 27,54] % | 0 von 20 | 20 von 20 |
| Regret-Verlust, λ = 10 | 20,44 [14,87; 30,84] % | 0 von 20 | 20 von 20 |
| Regret-Verlust, λ = 100 | 21,59 [16,17; 28,40] % | 0 von 20 | 20 von 20 |
| Hamming-Abstand, λ = 0,3 | 21,01 [14,80; 29,70] % | 0 von 20 | 20 von 20 |
| Hamming-Abstand, λ = 1 | 21,85 [17,81; 24,27] % | 0 von 20 | 20 von 20 |
| Hamming-Abstand, λ = 10 | 22,53 [20,00; 31,15] % | 0 von 20 | 20 von 20 |
| Regret-Verlust mit mittleren Kosten, λ = 1 | 11,13 [9,99; 12,64] % | 20 von 20 | 0 von 20 |
| Regret-Verlust mit mittleren Kosten, λ = 10 | 11,40 [10,16; 13,37] % | 20 von 20 | 0 von 20 |
| Hamming-Abstand, sauberer Zielweg, λ = 0,3 | 11,57 [10,31; 13,45] % | 19 von 20 | 1 von 20 |
| Hamming-Abstand, sauberer Zielweg, λ = 1 | 11,59 [10,29; 13,37] % | 20 von 20 | 0 von 20 |
| Hamming-Abstand, sauberer Zielweg, λ = 10 | 11,59 [10,22; 13,25] % | 20 von 20 | 0 von 20 |

„Besser“ und „schlechter als KQ“ zählen die Seeds (von 20), in denen das Verfahren im Regret streng unter bzw. über KQ liegt. „Mit mittleren Kosten“ und „sauberer Zielweg“ sind Zusatzmessungen dieser Demo: sie benutzen die mittleren Kosten $m(x)$, die es nur in der Simulation gibt. Ohne Rauschen sind sauberes und verrauschtes Ziel dasselbe, die Zeilen stimmen dort überein.

| Frage | Befund |
|---|---|
| **Funktioniert die Interpolation ohne Rauschen?** | Ja: bei λ = 0,3 und λ = 1 im Median 0,56 und 0,57 % gegen 1,96 % (KQ) und 0,51 % (SPO+); in allen 20 Seeds besser als KQ. Bei λ = 10 und 100 1,02 und 1,01 %, ebenfalls in 20 von 20 Seeds besser als KQ, aber klar hinter λ = 1 und SPO+. |
| **Und mit Rauschen?** | Nein: Rauschstufe 0,5, alle vier Interpolationen mit Regret-Verlust liegen im Median bei 18,6 bis 21,6 % gegen 13,1 % für KQ; in 20 von 20 Seeds schlechter als KQ. Im Median der Paare liegt λ = 1 8,0 Punkte über KQ. Der kleinste Regret, den irgendeine der sieben Varianten mit verrauschtem Ziel (vier mit Regret-Verlust, drei mit Hamming-Abstand) in irgendeinem Seed erreicht (14,8 %), liegt über dem größten Regret von KQ in irgendeinem Seed (14,5 %). |
| **Hängt es an λ?** | Ohne Rauschen ja, mäßig: der Regret steigt von 0,57 % (λ = 1) auf 1,02 und 1,01 % (λ = 10 und 100), und die Kurve steigt dort nach dem Tiefpunkt wieder (Tabelle unten). Mit Rauschen nicht entscheidend: das kleinste λ (0,3) ist mit 18,6 % besser als die übrigen drei Regret-Varianten, aber in jedem Seed schlechter als KQ. |
| **Warum? Der Gradient an der Wahrheit** | Rauschstufe 0,5: Mittelwert des Gradienten an der Wahrheit bei 0,713 (λ = 1) und 0,695 (λ = 10) der mittleren Größe, SPO+ 0,047, Fenchel-Young 0,116 (σ = 0,3) und 0,124 (σ = 1); Median über 20 Welten. Ohne Rauschen sind SPO+ und die Interpolation genau 0 (Median und Maximum über die 20 Welten), Fenchel-Young 0,706 und 0,700. |
| **Hängt es am Verlust?** | Nein: mit dem Hamming-Abstand zum verrauschten Zielweg liegt der Regret bei Rauschstufe 0,5 bei 21,0 / 21,9 / 22,5 % (λ = 0,3 / 1 / 10), in 20 von 20 Seeds schlechter als KQ; ohne Rauschen bei 1,1 / 1,0 / 0,9 % (λ = 1 und 10 in 20, λ = 0,3 in 18 von 20 Seeds besser als KQ). |
| **Hängt es am Rauschen der Ziele? (Zusatzmessung)** | Ja, in dieser Welt: mit sauberem Ziel liegt der Regret bei Rauschstufe 0,5 bei 11,1 bis 11,6 % (fünf Varianten), schlechter als KQ in höchstens 1 von 20 Seeds. Gegen die Lücke zwischen KQ und Orakel (13,1 und 10,5 %) schließt SPO+ im Median 31 %, die Interpolation mit mittleren Kosten (λ = 1) 77 %, mit sauberem Zielweg (Hamming, λ = 1) 58 % (aus den ungerundeten Medianwerten berechnet). Das ist kein Vergleich unter gleichen Bedingungen: die sauberen Varianten benutzen $m(x)$, SPO+ nicht. |
| **Was kostet der Löser?** | Gezählt: SPO+ löst 301.000 Instanzen, die Interpolation 601.000 (n = 1000, 300 Schritte; der Gradient selbst braucht 600.000 Aufrufe, dazu kommt die einmalige Lösung auf den Trainingskosten, die nur der Beobachtung des Trainingsverlusts dient). Eine Zählung, keine Zeitmessung; der Gitterlöser ist billig. |

### Lernkurve (Regret des Parameters nach t Adam-Schritten, Median über 20 Seeds; Rauschstufe 0)

| Schritte | SPO+ | Interpolation λ = 1 | Interpolation λ = 10 | Interpolation λ = 100 |
|---|---|---|---|---|
| 0 | 1,96 % | 1,96 % | 1,96 % | 1,96 % |
| 5 | 0,68 % | 0,99 % | 0,93 % | 1,10 % |
| 10 | 0,61 % | 0,71 % | 0,82 % | 0,88 % |
| 25 | 0,53 % | 0,56 % | 0,64 % | 0,76 % |
| 50 | 0,52 % | 0,57 % | 0,68 % | 0,73 % |
| 100 | 0,53 % | 0,57 % | 0,97 % | 1,02 % |
| 200 | 0,53 % | 0,57 % | 0,90 % | 0,92 % |
| 300 | 0,52 % | 0,57 % | 1,02 % | 1,01 % |

### Lernkurve (Rauschstufe 0,5)

| Schritte | SPO+ | Interpolation λ = 1 | Interpolation λ = 10 | Interpolation λ = 100 |
|---|---|---|---|---|
| 0 | 13,08 % | 13,08 % | 13,08 % | 13,08 % |
| 5 | 12,66 % | 14,06 % | 18,00 % | 19,02 % |
| 10 | 12,30 % | 15,17 % | 24,07 % | 23,96 % |
| 25 | 12,30 % | 19,18 % | 21,33 % | 20,74 % |
| 50 | 12,26 % | 19,85 % | 21,95 % | 21,54 % |
| 100 | 12,31 % | 21,06 % | 21,45 % | 20,05 % |
| 200 | 12,28 % | 20,52 % | 21,46 % | 21,30 % |
| 300 | 12,20 % | 20,80 % | 20,44 % | 21,59 % |

Beim Wert für SPO+ steht der Parameter nach t Schritten (nicht der mit dem kleinsten Trainingsverlust); beide Kurven beginnen bei KQ. Mit Rauschen liegt der Regret der Interpolation schon nach 5 Schritten im Median über dem Startwert (KQ) und bleibt in allen späteren Zeilen darüber.

### Gradient an der Wahrheit (Verhältnis |Mittel| / Mittel von |·|; 60 Merkmalsvektoren, 400 Rauschziehungen je Vektor, Median über die Merkmalsvektoren einer Welt, dann Median [Minimum; Maximum] über 20 Welten)

| Verfahren | Rauschstufe 0 (Median [Minimum; Maximum] über 20 Welten) | Rauschstufe 0,5 (Median [Minimum; Maximum] über 20 Welten) |
|---|---|---|
| SPO+ | 0,000 [0,000; 0,000] | 0,047 [0,044; 0,050] |
| Fenchel-Young (σ = 0,3) | 0,706 [0,688; 0,735] | 0,116 [0,081; 0,165] |
| Fenchel-Young (σ = 1) | 0,700 [0,689; 0,717] | 0,124 [0,113; 0,139] |
| Interpolation (λ = 1) | 0,000 [0,000; 0,000] | 0,713 [0,688; 0,741] |
| Interpolation (λ = 10) | 0,000 [0,000; 0,000] | 0,695 [0,681; 0,715] |

### Gegenprobe mit mehr Rauschziehungen (Rauschstufe 0,5, eine Welt, Zufalls-Seed 0, 60 Merkmalsvektoren)

| Rauschziehungen je Merkmalsvektor | SPO+ | Fenchel-Young (σ = 1) | Interpolation (λ = 1) |
|---|---|---|---|
| 100 | 0,087 | 0,140 | 0,709 |
| 400 | 0,048 | 0,122 | 0,709 |
| 1600 | 0,023 | 0,110 | 0,708 |
| 6400 | 0,011 | 0,111 | 0,709 |

Der Wert für SPO+ fällt mit der Wurzel der Zahl der Ziehungen (von 100 auf 6400 Ziehungen auf etwa ein Achtel), der der Interpolation bleibt bei 0,71, der von Fenchel-Young nähert sich 0,11. Das passt dazu, dass der SPO+-Mittelwert an der Wahrheit bei symmetrischem Rauschen **genau null** ist und die 0,05 bei 400 Ziehungen die Streuung der Stichprobe sind: Zu jeder Ziehung $\varepsilon$ gehört die gespiegelte $2 - \varepsilon$, die die beiden Wege $w^*(c)$ und $w(2m - c)$ vertauscht und den Teilgradienten umkehrt (im Test exakt geprüft, `test_spo_plus_gradient_at_the_truth_is_antisymmetric_under_mirrored_noise`). Dass die Interpolation und Fenchel-Young bei null Rauschen nicht gleich reagieren (die Interpolation 0, Fenchel-Young 0,7), ist gemessen; die Erklärung (Fenchel-Young glättet auch ohne Rauschen) ist eine Deutung.

### Aufwand

| Verfahren | gelöste Instanzen (n = 1000, 300 Schritte) |
|---|---|
| SPO+ | 301.000 |
| Interpolation | 601.000 |

### Diese Messung gegen die Vormessung

| Verfahren | Vormessung (Rauschstufe 0 / 0,5) | diese Demo (Rauschstufe 0 / 0,5) |
|---|---|---|
| Kleinste Quadrate (KQ) | 1,87 / 12,47 | 1,96 / 13,08 |
| Regret-Verlust, λ = 0,3 | 0,60 / 18,06 | 0,56 / 18,61 |
| Regret-Verlust, λ = 1 | 0,54 / 21,48 | 0,57 / 20,80 |
| Regret-Verlust, λ = 10 | 1,06 / 21,03 | 1,02 / 20,44 |
| Hamming-Abstand, λ = 0,3 | 1,01 / 20,93 | 1,08 / 21,01 |
| Hamming-Abstand, λ = 1 | 1,06 / 21,15 | 0,97 / 21,85 |
| Hamming-Abstand, λ = 10 | 0,94 / 21,26 | 0,92 / 22,53 |

### Prüfungen des Lösers und des Gradienten

- **Löser gegen Aufzählung der 70 Wege** (`tests/test_oracle_blackbox_loeser.py`): stetige und ganzzahlige Kosten (viele Gleichstände), dieselben Werte und dieselben Wege inklusive Gleichstandsregel.
- **Gradient der Interpolation gegen die Aufzählung** für beide Verluste und λ = 0,3 / 1 / 10 (/ 100): $(w_{\text{Aufzählung}}(\hat c + \lambda\,\mathrm{d}L/\mathrm{d}y) - w_{\text{Aufzählung}}(\hat c)) / \lambda$, keine Abweichung. **Handbeispiel** gegen die geschlossene Formel auf 9 × 9 Vorhersagen und den 8 λ der App (648 Fälle).
- **Hamming-Identität** $\sum_j |y_j - y^*_j| = y^\top(1 - 2y^*) + \sum_j y^*_j$ über alle 70 Wege für zehn Zielwege.
- **Scratch-Lauf** (nicht im Repo): Löser gegen Aufzählung auf 300 Instanzen mit stetigen Kosten (größte Abweichung im Wert 7·10⁻¹⁵, dieselben Wege) und 300 mit ganzzahligen Kosten (dieselben Werte und Wege); Gradient der Interpolation auf 500 Instanzen, vier λ (0,3 / 1 / 10 / 100) und beide Verluste: größte Abweichung 0. Im Repo läuft die kleine Fassung.

## Befunde und Korrekturen gegenüber dem Plan

- **Die Zahlen der Vormessung reproduzieren sich nicht auf die Stelle.** Die Vormessung benutzte numpy-Zufall (hier der ganzzahlige SplitMix64, andere Welten); beide rechnen 300 Adam-Schritte. Die Tabelle „Diese Messung gegen die Vormessung“ zeigt die Abweichungen: ohne Rauschen höchstens 0,1 Punkte bei den Interpolationen (KQ 1,87 → 1,96 %), mit Rauschen bis 1,3 Punkte (Hamming, λ = 10: 21,26 → 22,53 %; KQ 12,47 → 13,08 %). Es gilt die neue Messung; **Vorzeichen und Größenordnung sind dieselben, die Rangfolge unter den Varianten nicht**. Genauer weichen im Plantext ab: „alle λ bei 20 bis 21 %“ gilt jetzt als 20,4 bis 21,6 % für λ = 1 / 10 / 100, dazu 18,6 % bei λ = 0,3; „0,70 bis 0,72“ für die Interpolation an der Wahrheit ist 0,695 bis 0,713 (Median über 20 Welten; die Vormessung nahm eine Welt); Fenchel-Young „0,10 bis 0,11“ ist 0,116 und 0,124 (σ = 0,3 und 1); „steigt von 0,62 / 0,65 nach 25 Schritten“ ist hier „von 0,64 / 0,76 % nach 25 Schritten“.
- **Zusatzmessung außerhalb der vier geplanten Experimente:** Der Plan (Z3) sah ein Experiment „saubere Ziele“ nur für den Fall vor, dass die Quelle mit sauberen Zielwegen arbeitet; ich habe es ohne Kenntnis der Quelle gemessen, weil es die offene Messlücke („keine Messung mit sauberen Zielwegen“) schließt. Ergebnis: mit sauberem Ziel trägt die Interpolation auch bei Rauschen (Tabelle oben). Der Negativbefund bleibt, wird aber genauer: Er gilt für **verrauschte Ziele**, nicht für die Interpolation als solche, und nicht für den Verlust.
- **Die Erklärung „Teilgradient an der Wahrheit nicht null“ hat zwei Teile.** Gemessen ist, dass er bei der Interpolation mit verrauschtem dL/dy nicht null ist und mit den mittleren Kosten als dL/dy exakt null (dann ist $\hat c_\lambda = (1 + \lambda)\,m$, derselbe Weg; im Test exakt). Warum der Mittelwert nicht null ist (ein Mittel über Wege, die zu verrauschten Kosten gewählt wurden, ist nicht der Weg zu den mittleren Kosten), ist eine Deutung, die die Zusatzmessung stützt, aber nicht beweist.

## Ehrliche Grenzen

- **Negativbefund im eigenen Aufbau, nicht über die Quelle.** Vlastelica u. a. habe ich selbst nicht im Volltext gelesen: Satzbedingungen, Wahl von λ und Experimente sind nicht im Primärtext geprüft. Ob deren Experimente mit sauberen Zielwegen arbeiten, weiß ich nicht. Die Demo sagt nicht „das Verfahren scheitert“.
- **Verrauschte Ziele sind eine Eigenschaft dieses Aufbaus:** der Zielweg ist der beste Weg zu den beobachteten Kosten, das Rauschen multiplikativ gleichverteilt. Andere Rauschformen sind nicht gemessen. Der Befund „sauberes Ziel trägt“ gilt nur in der Simulation; im Alltag kennt man die mittleren Kosten nicht.
- **Eine Welt, ein Gitter:** nur 5 × 5, Grad 6, ein lineares Modell, Adam mit fester Lernrate 0,05 und 300 Schritten, Parameter nach dem letzten Schritt (bei der Interpolation). λ und die Lernrate wurden nicht abgestimmt. Aussagen über andere Gitter, Modelle oder Trainer sind nicht gemessen. Das Gitter und die Kostenformel sind aus dem Gedächtnis nachgebaut.
- **Zahlen sind trainiert und plattformabhängig.** Gemessen unter Windows; unter Linux können einzelne Werte um Zehntelpunkte abweichen. Die Tests prüfen Bänder und Ungleichungen.
- **Median über 20 Seeds, kein Signifikanztest.** Die Spanne (Minimum bis Maximum) ist keine Konfidenzaussage.
- **Ursachen sind Deutungen**, soweit sie nicht in der Tabelle „Befunde“ als gemessen stehen (siehe oben: Warum der Mittelwert nicht null ist).
- **Der Fenchel-Young-Gradient** ist als Bezug mitgerechnet, aus dem Gedächtnis nach Abstract-Ebene nachgebaut und nicht gegen die Quelle geprüft; er ist hier kein Gegenstand.
- **Keine Aussage über Rechenzeit.** Gezählt sind gelöste Instanzen; eine Zeitaussage für teure Löser (etwa ein gemischt-ganzzahliges Programm) ist nicht gemessen, der Löser im Gitter ist billig.

## Bewusst nicht umgesetzt

- **Die Experimente der Quelle** (Volltext von Vlastelica u. a., Originalcode oder eine Bibliothek wie PyEPO als Offline-Referenz).
- **Fenchel-Young und gestörte Optimierer als Trainingsverfahren, Glättung des Regrets, implizite Differentiation:** andere Demos der Reihe bzw. `kegel-ableitung-demo` (Ableitung am Ergebnis).
- **Abstimmung von λ** (etwa per Kreuzvalidierung) und anderer Trainer (Lernrate, Schrittzahl, Regularisierung).
- **Größere Gitter, teure Löser, andere Rauschformen, flexible Modellklassen (Zufallswald, Kern-Ridge).**
- **Eine Zeitmessung des Lösers.**

## Tests

170 Tests (AppTest-Smoke und Layout eingeschlossen). Zusammensetzung: Handbeispiel und Eigenschaften der Interpolation (`test_blackbox.py`), Orakel-Tests gegen die Aufzählung der 70 Wege und die geschlossene Formel (`test_oracle_blackbox_loeser.py`), Gradient an der Wahrheit (`test_bias.py`), Läufe und Zählung der gelösten Instanzen (`test_runs.py`), die kopierten Einheiten des Gitter-Gerüsts (Zufallsstrom, Gitter und Löser, Welt, Verluste, Training), Verdikte an handgesetzten Werten, Randfälle, Presets, Layout, AppTest-Smoke, Footer, Laufzeit-Importe und Claims (`test_claims.py`).

| README-Aussage | Test |
|---|---|
| alle Tabellen („Befunde“) | `test_readme_contains_every_generated_table_line` (wörtlich aus der JSON-Datei erzeugt) |
| Ohne Rauschen funktioniert es; λ-Abhängigkeit; Kurve | `test_without_noise_the_interpolation_beats_kq_in_every_seed`, `test_larger_lambda_is_coarser_without_noise`, `test_learning_curves` |
| Mit Rauschen schlechter als KQ in 20 von 20 Seeds | `test_with_noise_every_regret_interpolation_is_worse_than_kq_in_every_seed` |
| Hamming-Verlust ändert nichts | `test_the_hamming_loss_does_not_change_the_verdict` |
| Saubere Ziele (Zusatzmessung) und Anteil der Lücke | `test_clean_targets_beat_kq_and_close_most_of_the_gap` |
| Gradient an der Wahrheit (Bänder) | `test_gradient_at_the_truth_bands` |
| Gegenprobe mit mehr Ziehungen | `test_more_draws_shrink_spo_plus_but_not_the_interpolation` |
| SPO+-Mittelwert null bei gespiegeltem Rauschen | `test_bias.py::test_spo_plus_gradient_at_the_truth_is_antisymmetric_under_mirrored_noise` |
| Aufwand (301.000, 601.000) | `test_effort_counts`, `test_runs.py::test_solved_rows_by_hand` |
| Abweichungen von der Vormessung | `test_comparison_with_the_pre_measurement` |
| Messaufbau der Bänder | `test_band_file_describes_the_documented_setup`, `test_live_run_reproduces_the_bands` |
| Handbeispiel (0, 0) / (2, −2) / (1, −1) / (0,1; −0,1) | `test_blackbox.py::test_hand_example_all_four_lambdas`, `test_oracle_blackbox_loeser.py::test_hand_example_equals_the_closed_form_on_a_grid_of_forecasts_and_lambdas` |
| Scratch-Läufe (300 bzw. 500 Instanzen) | nicht im Repo; im Repo die kleine Fassung der Orakel-Tests |

Trainierte Zahlen werden nur als Bänder und Ungleichungen geprüft (Plattform); exakt geprüft werden das Handbeispiel, die Eigenschaften des Gradienten, die Zählungen und die Tabellen gegen die JSON-Datei.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Kopf, Ergebnis der Welt, Kernfrage, vier Experimente, Grenzen, 📐 Mathematische Formulierung |
| `bbl_blackbox.py` | Kern: Gradient der Interpolation, beide Verluste, Zielfunktion und Training |
| `bbl_bias.py` | Messung des Gradienten an der Wahrheit (SPO+, Fenchel-Young, Interpolation) |
| `bbl_runs.py` | Läufe: Verfahrensschlüssel, Welt, Training, Zählung der gelösten Instanzen |
| `bbl_rng.py`, `bbl_grid.py`, `bbl_world.py`, `bbl_losses.py`, `bbl_training.py` | Kopien des Gitter-Gerüsts aus `spo-kuerzester-weg-demo` (Herkunft und Änderungen im Modulkopf): Zufallsstrom, Löser, Welt, SPO+ und Verluste, Adam |
| `bbl_bands.py`, `bbl_bands.json` | vorgerechnete Bänder (eigene Messung, `tools/measure_bands.py`) |
| `bbl_evaluation.py` | Kennzahlen, Verdikte, Tabellen |
| `bbl_constants.py`, `bbl_presets.py`, `bbl_layout.py` | Regler-Grenzen und Farben, Permalink und Schnellstart, fester Kopf |
| `bbl_visualization.py`, `bbl_pdf_export.py` | Plotly-Abbildungen (Achsen gesperrt), PDF-Export |
| `tools/measure_bands.py` | misst die Bänder neu (höchstens zwei Prozesse) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit-Bibliotheken (ungepinnt), Laufzeit plus pytest |
| `tests/` | siehe Abschnitt „Tests“ |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Bänder neu messen: `python tools/measure_bands.py --jobs 2`. Tests: `pip install -r requirements-dev.txt`, dann `pytest tests/ -v`.

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
