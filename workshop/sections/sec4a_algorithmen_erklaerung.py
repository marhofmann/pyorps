# ruff: noqa: E501, B028  (long German prose lines in the cells)
"""Abschnitt 4 (erste Hälfte): kurze Erklärung Dijkstra und Delta-Stepping mit einem Spielfeld."""

CELLS = [
    ("md", r"""---
# Teil 4: Dijkstra und Delta-Stepping

**Warum zwei Verfahren?** Die Suche ist der Rechenkern von PYORPS. Auf einem Raster mit Millionen Zellen entscheidet
das Verfahren darüber, wie lange wir warten, vor allem wenn viele Routen zu rechnen sind und mehrere Prozessorkerne zur
Verfügung stehen. Ihr solltet deshalb wissen, was hinter `algorithm="..."` steckt.

Beide Verfahren lösen dieselbe Aufgabe und liefern **dieselben exakten Kosten**: die billigste Route von einer Quelle
zu allen anderen Zellen. Sie unterscheiden sich darin, **in welcher Reihenfolge** sie die Zellen abarbeiten.

## Dijkstra: immer die nächste Zelle

Eine Warteschlange (*Heap*) hält Zellen mit ihrer vorläufigen Distanz von der Quelle. Dijkstra nimmt **immer die Zelle
mit der kleinsten Distanz** heraus. Diese Distanz ist dann endgültig. Danach prüft das Verfahren die Nachbarn: Ist die
Verbindung über die gerade fertige Zelle billiger, wird deren Distanz verbessert (*Relaxieren*).

- Jede Zelle wird **genau einmal** abgeschlossen, streng in der Reihenfolge der Distanz.
- Einfach und exakt, aber **sequenziell**: Welche Zelle als nächste kommt, hängt vom Ergebnis der vorigen ab.
- Gibt es nur ein Ziel, kann die Suche stoppen, sobald dieses Ziel abgeschlossen ist.

## Delta-Stepping: ganze Eimer auf einmal

Die Zellen werden nach Distanz in **Eimer** (*Buckets*) der Breite Δ (`delta`) sortiert: Eimer *k* enthält Zellen mit
vorläufiger Distanz in [*k*·Δ, (*k*+1)·Δ). Das Verfahren leert immer den **kleinsten nicht leeren Eimer** und bearbeitet
**alle Zellen darin gleichzeitig**, also parallel auf mehreren Prozessorkernen oder auf der Grafikkarte.

- *Kurze* Kanten (Gewicht ≤ Δ) können eine Zelle wieder in denselben Eimer legen und werden deshalb wiederholt
  relaxiert, bis der Eimer leer bleibt. *Lange* Kanten werden einmal nach dem Eimer relaxiert.
- Am Ende stehen dieselben exakten Distanzen wie bei Dijkstra, aber viele Zellen pro Schritt.
- **Δ ist ein Stellrad:** sehr kleines Δ verhält sich fast wie Dijkstra (wenig Parallelität), sehr großes Δ gibt viel
  Parallelität, aber mehr Wiederholungsarbeit. In PYORPS ist `delta=100` voreingestellt.

## Die Unterschiede in einer Tabelle

| | Dijkstra | Delta-Stepping |
|---|---|---|
| Reihenfolge | eine Zelle nach der anderen, nach Distanz | ein Eimer nach dem anderen, darin alle Zellen zugleich |
| Parallel | nein | ja (OpenMP auf dem Prozessor, CUDA auf der GPU) |
| Ergebnis | exakt | exakt, dieselben Kosten |
| Stellgröße | keine | Eimerbreite Δ, Threadzahl |
| Eigenart | einfach, keine Einstellung nötig, rechnet auch mit Steigungskosten | bearbeitet viele Zellen zugleich; wie viel das bringt, hängt von Raster, Zahl der Ziele und Zahl der Kerne ab |
| Grenze in PYORPS | – | **keine Steigungskosten** (Teil 5): mit Geländemodell fällt PYORPS auf Dijkstra zurück und warnt |

**Welches Verfahren schneller ist, sagen wir nicht voraus.** Es hängt vom Raster, von der Zahl der Ziele und von der
Zahl der Kerne ab. **Wir messen es unten selbst** und lesen die Antwort aus den Tabellen. Sicher ist nur eines: Mit
Geländemodell läuft PYORPS immer mit Dijkstra."""),

    ("md", r"""### Ein Spielraster zum Anschauen

Wir rechnen auf einem kleinen Raster (25 × 25 Zellen, eine gesperrte Wand, ein teurer Streifen) beide Reihenfolgen nach.
Die **Distanzen sind in beiden Fällen gleich**; die Bilder zeigen, *wann* eine Zelle abgeschlossen wird.
(Vereinfachung: Wir zeigen für Delta-Stepping den Eimer, in dem die Zelle am Ende endgültig ist.)"""),

    ("code", r"""import heapq

import matplotlib.pyplot as plt
import numpy as np

# Kleines Spielraster: 1 = normal, 6 = teurer Streifen, unendlich = gesperrt
n = 25
kosten = np.ones((n, n))
kosten[8:17, 6] = np.inf            # eine gesperrte Wand
kosten[5:20, 12:15] = 6.0           # ein teurer Streifen
quelle = (12, 2)

schritte = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def dijkstra(kosten, quelle):
    # Gibt Distanzfeld und Abschluss-Reihenfolge zurück
    dist = np.full(kosten.shape, np.inf)
    rang = np.full(kosten.shape, -1)
    dist[quelle] = 0.0
    heap = [(0.0, quelle)]
    zaehler = 0
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if d > dist[r, c] or rang[r, c] >= 0:
            continue
        rang[r, c] = zaehler          # Zelle ist jetzt endgültig
        zaehler += 1
        for dr, dc in schritte:
            rr, cc = r + dr, c + dc
            if 0 <= rr < kosten.shape[0] and 0 <= cc < kosten.shape[1] and np.isfinite(kosten[rr, cc]):
                laenge = np.hypot(dr, dc)
                neu = d + laenge * 0.5 * (kosten[r, c] + kosten[rr, cc])
                if neu < dist[rr, cc]:
                    dist[rr, cc] = neu
                    heapq.heappush(heap, (neu, (rr, cc)))
    return dist, rang


dist, rang = dijkstra(kosten, quelle)
delta = 10.0
eimer = np.where(np.isfinite(dist), np.floor(dist / delta), np.nan)

erreichbar = int(np.isfinite(dist).sum())
n_eimer = int(np.nanmax(eimer)) + 1
print(f"Erreichbare Zellen: {erreichbar}")
print(f"Dijkstra:       {erreichbar} Einzelschritte (eine Zelle pro Schritt)")
print(f"Delta-Stepping: mindestens {n_eimer} Runden (Eimer) bei Δ = {delta:g}; im größten Eimer "
      f"{int(np.bincount(eimer[np.isfinite(eimer)].astype(int)).max())} Zellen zugleich "
      "(mit den Wiederholungen innerhalb eines Eimers sind es etwas mehr Runden)")

fig, achsen = plt.subplots(1, 3, figsize=(12, 3.8))
bild = np.where(np.isfinite(kosten), kosten, np.nan)
m0 = achsen[0].imshow(bild, cmap="YlOrBr", vmin=0, vmax=8)
achsen[0].set_title("Kosten je Zelle (weiß = gesperrt)")
plt.colorbar(m0, ax=achsen[0], fraction=0.046, label="Kosten")
r1 = np.where(rang >= 0, rang, np.nan)
m1 = achsen[1].imshow(r1, cmap="viridis")
achsen[1].set_title("Dijkstra: Nummer des Schritts")
plt.colorbar(m1, ax=achsen[1], fraction=0.046, label="Schritt")
m2 = achsen[2].imshow(eimer, cmap="viridis")
achsen[2].set_title(f"Delta-Stepping: Eimer (Δ = {delta:g})")
cb2 = plt.colorbar(m2, ax=achsen[2], fraction=0.046, label="Eimer")
from matplotlib.ticker import MaxNLocator
cb2.locator = MaxNLocator(integer=True)   # Eimer sind ganze Zahlen
cb2.update_ticks()
for a in achsen:
    a.plot(*quelle[::-1], "r*", ms=12)
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
plt.show()"""),

    ("md", r"""Links: das Spielraster mit der Quelle (roter Stern). In der Mitte erscheinen **viele einzelne Schritte** von innen
nach außen: Dijkstra geht Zelle für Zelle. Rechts sind es **wenige Eimer**: Alle Zellen mit ähnlicher Distanz haben
dieselbe Farbe und lassen sich gleichzeitig bearbeiten. Die Wand und der teure Streifen verformen beide Bilder gleich,
denn die Distanzen sind dieselben.

Jetzt rechnen wir mit PYORPS auf dem echten Kostenraster."""),
]
