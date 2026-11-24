# PYORPS-Workshop 

Trassenplanung mit PYORPS, vom Kostenraster bis zur Freileitung.

**Notebook:** `pyorps_workshop_de.ipynb`

| Teil | Thema |
|:---:|---|
| 1 | Installation und Prüfung (`pip install "pyorps[gui,examples]"`) |
| 2 | Grundidee: Raster, Graph, kürzester Weg, Nachbarschaften |
| 3 | Kostenraster aus mehreren überlappenden Layern (ALKIS, Schutzgebiete, eigene Layer), zwei Kostenvarianten |
| 4 | Dijkstra und Delta-Stepping: kurze Erklärung; 1→1, 1→N, M→N, paarweise, Kostenfeld |
| 5 | Geländemodell (DGM1) und 3D-Routing |
| 6 | Freileitung mit `ConstrainedPathFinder` (Masten, Spannfelder, Abstand zum Boden) |
| 7 | Die GUI lokal betreiben |

**Dauer:** grob 3,5 bis 4,5 Stunden ohne Übungen (Schätzung für Einsteiger), mit Übungen deutlich mehr. Vorschlag: zwei
Blöcke, Teil 1 bis 4 und Teil 5 bis 7. Entbehrliche Abschnitte sind im Notebook mit „Vertiefung (optional)“ markiert.
Empfohlen sind 8 GB Arbeitsspeicher.

## Benutzen

```bash
pip install "pyorps[gui,examples]"      # Version 0.5.0 oder neuer
cd workshop                             # Jupyter im Ordner workshop/ starten, nicht im Hauptordner
jupyter notebook pyorps_workshop_de.ipynb
```

Das Notebook läuft nach dem ersten Mal **ohne Internet**: Die Layer (`data/layers/`), die Kostenraster und das
Geländemodell (`data/dem_clip.tif`) liegen im Ordner. Fehlt eine Layer-Datei, lädt das Notebook sie vom WFS-Dienst nach.

## Daten und Lizenzen

| Datei | Inhalt | Quelle und Lizenz |
|---|---|---|
| `data/layers/alkis_*.gpkg` | Tatsächliche Nutzung (ALKIS) | LGL-BW, Datenlizenz Deutschland – Namensnennung – Version 2.0 |
| `data/layers/{biotop,ffh,wasserschutz,ueberschwemmung,naturschutz}_*.gpkg` | Schutz- und Biotopflächen | LUBW, Datenlizenz Deutschland – Namensnennung – Version 2.0 |
| `data/dem_clip.tif` | DGM1, auf 5 m gemittelt | LGL-BW, Datenlizenz Deutschland – Namensnennung – Version 2.0 |
| `data/cost_raster_a.tiff`, `data/cost_raster_b.tiff` | Kostenraster, vom Notebook (Teil 3) erzeugt | abgeleitet aus den Layern oben |
| `data/profiles/overhead_line_*.yaml` | Freileitungsprofile 110, 220 und 380 kV | Kopie von `profiles/` im Repository |

**Namensnennung:** © LGL-BW (ALKIS, DGM1) und © LUBW (Schutzgebiete, Biotope, Überschwemmungsgebiete),
[Datenlizenz Deutschland – Namensnennung – Version 2.0](https://www.govdata.de/dl-de/by-2-0). Die Dateien in `data/` sind
**verändert**: zugeschnitten, gerastert, auf 5 m gemittelt oder umklassifiziert. Die Kostenwerte stammen aus dem Tutorial
zum OSMSES-Workshop 2026 (`tutorial/` in diesem Repository).

Untersuchungsgebiet: 5 × 5 km im Kinzigtal bei Biberach (EPSG:25832, Rechtswert 427000 bis 432000, Hochwert 5354000 bis 5359000).
Die Oberfläche mit Baumhöhen in Teil 6 ist **eine Annahme für den Unterricht** (Wald plus 25 m), kein Messwert.

## Notebook neu bauen

Das Notebook entsteht aus den Dateien in `sections/`. Nicht im Notebook von Hand ändern, sondern in `sections/*.py`:

```bash
python workshop/build_workshop_notebook.py            # schreibt pyorps_workshop_de.ipynb (ohne Ausgaben)
python workshop/build_workshop_notebook.py --check    # prüft, ob Notebook und Abschnitte übereinstimmen
```

Alle Code-Zellen laufen von oben nach unten in einem frischen Kernel (geprüft mit `jupyter nbconvert --execute`).

**Daten neu erzeugen:** Die Layer-Dateien in `data/layers/` lädt Teil 3 vom WFS-Dienst, sobald sie fehlen; die Kostenraster
baut Teil 3 aus den Layern. Das Geländemodell `data/dem_clip.tif` holt `hole_dgm()` in Teil 5, wenn die Datei fehlt
(einmaliger Download von etwa 235 MB beim LGL-BW; ohne Netz entsteht ein künstliches Gelände, das im Notebook als solches
benannt wird).
