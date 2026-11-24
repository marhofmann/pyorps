# ruff: noqa: E501, B028  (long German prose lines in the cells)
"""Abschnitt 2: Grundidee (kurz, ohne Rechenzeit)."""

CELLS = [
    ("md", r"""---
# Teil 2: Grundidee

**Warum?** Eine Trasse für Kabel oder Freileitung muss Wiesen, Wald, Straßen, Gewässer und Schutzgebiete kreuzen, und jedes
Gelände kostet etwas anderes. Von Hand findet man dafür selten die billigste Lösung. Ein Rechner kann es, wenn wir das
Problem in eine Form bringen, die er versteht: ein Raster mit Kosten.

PYORPS macht aus der Trassenplanung ein **Kürzeste-Wege-Problem auf einem Graphen**:

1. **Raster:** Jede Zelle trägt einen Kostenwert, zum Beispiel die Baukosten in EUR/m für diese Landnutzung.
2. **Graph:** Jede Zelle ist ein Knoten. Kanten verbinden Nachbarzellen; das Gewicht einer Kante ist *Kosten × Länge*.
3. **Suche:** Ein Algorithmus (Dijkstra oder Delta-Stepping, Teil 4) findet die Verbindung mit den kleinsten Gesamtkosten.
4. **Ergebnis:** Eine georeferenzierte Linie mit Länge, Kosten und exportierbarer Geometrie (GeoJSON, GeoPackage).

**Begriffe in diesem Notebook:** Die berechnete Linie heißt **Route**. **Trasse** sagen wir nur für die Planungsaufgabe
("wir planen eine Trasse"), **Leitung** nur für die Freileitung in Teil 6.

## Kostenwerte

| Wert | Bedeutung |
|---|---|
| 0 | kostenlos (zum Beispiel vorhandenes Leerrohr) |
| 1 bis 65534 | wachsende Kosten (typisch einige hundert EUR/m) |
| **65535** | **gesperrt**: dort darf die Route nicht verlaufen |

Die Werte liegen als `uint16` im Raster. **Niedriger Wert = billig**, gesperrt = unendlich teuer.

## Nachbarschaften

Die *Nachbarschaft* bestimmt, in wie viele Richtungen die Suche von einer Zelle aus gehen darf.

| Nachbarschaft | Richtungen | größte Längenüberschätzung | Beschreibung |
|:---:|:---:|:---:|---|
| `r0` | 4 | 41,4 % | nur Nord, Süd, Ost, West |
| `r1` | 8 | 8,2 % | zusätzlich die Diagonalen |
| **`r2`** | **16** | **2,7 %** | zusätzlich Springerzüge (Standard) |
| `r3` | 32 | 1,3 % | erweiterte Springerzüge |

Ein *Springerzug* geht wie der Springer im Schach: zwei Zellen in eine Richtung und eine zur Seite.
Die Längenüberschätzung sagt, um wie viel länger als die Luftlinie eine Route im ungünstigsten Winkel höchstens wird,
wenn sie nur diese Schritte benutzen darf (Rechnung für lange Strecken auf einem leeren Raster).
Mehr Richtungen ergeben glattere, genauere Routen, kosten aber Rechenzeit. Für die meisten Aufgaben ist `r2` richtig.
Wir setzen sie in Teil 3 selbst ein (`neighborhood_str="r2"`)."""),
]
