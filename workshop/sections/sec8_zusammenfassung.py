# ruff: noqa: E501, B028  (long German prose lines in the cells)
"""Abschnitt 8: Zusammenfassung, Entscheidungshilfe, weiterführende Quellen."""

CELLS = [
    ("md", r"""---
# Teil 8: Zusammenfassung

## Was wir gemacht haben

| Teil | Kernaussage |
|---|---|
| 1 Installation | `pip install "pyorps[gui,examples]"` in einer eigenen Umgebung; keine Compiler nötig |
| 2 Grundidee | Raster → Graph → kürzeste Route; niedriger Kostenwert = billig, 65535 = gesperrt |
| 3 Kostenraster | Ein Basislayer (Landnutzung) plus überlagernde Layer (sperren, Zuschläge, Abschläge); andere Annahmen ergeben andere Trassen |
| 4 Algorithmen | Dijkstra und Delta-Stepping liefern dieselben Kosten; Delta-Stepping arbeitet Eimer parallel ab |
| 5 Gelände | Mit Geländemodell zählen Steigung und 3D-Länge; Delta-Stepping unterstützt das nicht (Rückfall auf Dijkstra) |
| 6 Freileitung | `ConstrainedPathFinder` sucht Trasse und Maststandorte gemeinsam, mit Winkel- und Spannfeldgrenzen |
| 7 GUI | Dieselbe Planung im Browser, lokal auf dem eigenen Rechner |

## Welches Verfahren wofür?

| Aufgabe | Werkzeug |
|---|---|
| Ein Erdkabel von A nach B | `PathFinder`, Nachbarschaft `r2`; `algorithm="dijkstra"` oder `"delta-stepping"` (gleiche Kosten, Zeiten selbst messen) |
| Viele Ziele oder Quellen, großes Raster | `PathFinder` oder Kostenfeld, beide Algorithmen vergleichen (Teil 4) |
| Hügeliges Gelände, Steigungsgrenze | `PathFinder` mit Geländemodell und `Objective` (Dijkstra) |
| Freileitung mit Masten und Spannfeldern | `ConstrainedPathFinder` mit `InfrastructureProfile` |
| Planung ohne Code, mit Karte | `python -m pyorps.gui` |

## Weiterführend

- **Dokumentation:** [pyorps.readthedocs.io](https://pyorps.readthedocs.io), darin *Release Status*, *Neighborhoods*,
  *Cost semantics* und die API-Seiten zu `PathFinder`, `ConstrainedPathFinder` und der GUI
- **Quellcode und Fehlermeldungen:** [github.com/marhofmann/pyorps](https://github.com/marhofmann/pyorps)
- **Weitere Funktionen, die wir nicht behandelt haben:** GPU-Rechnung (`pip install "pyorps[gpu]"`, Backends
  `raster_gpu` und `raster_fim`), Kostenfelder und Suchsitzungen für viele Endpunktpaare, Korridorgraphen für gemeinsame
  Gräben, vorberechnete Mastfelder
- **Zitieren:** Hofmann, M., Stetz, T., Kammer, F., Repo, S.: „PYORPS: An Open-Source Tool for Automated Power Line Routing“,
  CIRED 2025 – 28th Conference and Exhibition on Electricity Distribution, 16.–19. Juni 2025, Genf

**Kontakt:** Martin Hofmann, THM – Technische Hochschule Mittelhessen, Gießen

---
*Datenquellen dieses Workshops: © LGL-BW (ALKIS, Tatsächliche Nutzung; DGM1) und © LUBW (Schutzgebiete, Biotope,
Überschwemmungsgebiete), Datenlizenz Deutschland – Namensnennung – Version 2.0 (www.govdata.de/dl-de/by-2-0). Die Daten
wurden zugeschnitten, gerastert, gemittelt und umklassifiziert. Das Geländemodell steht mit seiner Quelle in Teil 5.*"""),
]
