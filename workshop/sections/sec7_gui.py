# ruff: noqa: E501, B028  (long German prose lines in the cells)
"""Abschnitt 7: die GUI lokal betreiben."""

CELLS = [
    ("md", r"""---
# Teil 7: Die GUI lokal betreiben

Alles, was wir im Code gemacht haben, geht auch **ohne Code** in der grafischen Oberfläche von PYORPS
(`pyorps.gui`, eine Dash-Anwendung mit Karte). Sie läuft **auf eurem eigenen Rechner**: Ein kleiner Webserver
startet lokal, und ihr bedient ihn im Browser. Eure Daten verlassen den Rechner nicht.

## 7.1 Starten

Im Terminal, in der Umgebung, in der PYORPS installiert ist (Teil 1):

```bash
python -m pyorps.gui                      # öffnet die Adresse im Browser (Standard: http://127.0.0.1:8050)
python -m pyorps.gui --desktop            # eigenes Fenster (pywebview) statt Browser-Tab
python -m pyorps.gui --port 8060          # anderer Port; ist er belegt, nimmt PYORPS den nächsten freien
python -m pyorps.gui --work-dir gui_arbeit             # Ordner für Raster und temporäre Dateien
python -m pyorps.gui --project mein_gebiet/project.json  # gespeichertes Projekt direkt öffnen
```

| Option | Wirkung |
|---|---|
| `--desktop` | Fenster statt Browser (braucht `pywebview`, im Zusatz `[gui]` enthalten; unter Linux kann `pywebview` zusätzlich GTK oder Qt verlangen, dann hilft der Browser) |
| `--host`, `--port` | Adresse und Port; Standard `127.0.0.1` und `8050` |
| `--allow-remote` | erlaubt eine andere Adresse als `127.0.0.1` |
| `--work-dir` | Arbeitsordner; wird nie automatisch gelöscht |
| `--project DATEI` | öffnet ein gespeichertes Projekt (`project.json` oder der Ordner darin) |
| `--threads N` | Threads des Webservers (Standard 8) |

> **Sicherheit:** Die GUI hat **keine Anmeldung**, und ihre Pfadfelder erreichen jede Datei des Benutzers.
> Deshalb hört sie nur auf `127.0.0.1`. `--allow-remote` nur in einem vertrauenswürdigen Netz verwenden, nie im Internet.
> Für einen gemeinsamen Rechner im Seminarraum startet jede Person ihre **eigene** GUI auf dem eigenen Laptop."""),

    ("md", r"""## 7.2 Aus dem Notebook starten (optional)

Das Starten ist **nicht** Teil des normalen Laufs: Der Schalter `GUI_STARTEN` steht auf `False`. Setzt ihn auf `True`,
startet die nächste Zelle die GUI als Hintergrundprozess und schreibt die Meldungen in `data/gui_log.txt`.
Die Zelle danach beendet sie wieder. Ein Notebook-Lauf von oben nach unten startet also nie ungewollt einen Server."""),

    ("code", r"""import subprocess
import sys
import time
from pathlib import Path

GUI_STARTEN = False      # auf True setzen, um die GUI aus dem Notebook zu starten
gui_prozess = None

if GUI_STARTEN:
    arbeitsordner = Path("data/gui_arbeit").resolve()
    arbeitsordner.mkdir(parents=True, exist_ok=True)
    log = open("data/gui_log.txt", "w", encoding="utf-8")
    gui_prozess = subprocess.Popen(
        [sys.executable, "-m", "pyorps.gui", "--port", "8050", "--work-dir", str(arbeitsordner)],
        stdout=log, stderr=subprocess.STDOUT,
    )
    time.sleep(8)
    print("GUI läuft (Prozess", gui_prozess.pid, ") – im Browser öffnen: http://127.0.0.1:8050")
    print("(Ist der Port belegt, nimmt PYORPS den nächsten freien: die richtige Adresse steht in der Log-Datei.)")
    print("Meldungen:", Path("data/gui_log.txt").resolve())
else:
    print("GUI_STARTEN ist False – es wurde nichts gestartet.")"""),

    ("code", r"""# GUI beenden (nur nötig, wenn sie oben gestartet wurde)
if gui_prozess is not None:
    gui_prozess.terminate()
    gui_prozess.wait(timeout=15)
    print("GUI beendet.")
else:
    print("Keine GUI aus dem Notebook gestartet.")"""),

    ("md", r"""## 7.3 Notebook und GUI im Vergleich

Die GUI ist englisch beschriftet und hat sechs Reiter. Jeder entspricht einem Schritt dieses Workshops:

| Reiter in der GUI | Was dort passiert | Entsprechung im Notebook |
|---|---|---|
| **Data** | Gebiet auf der Karte zeichnen; Daten laden (lokale Datei, WFS, OpenStreetMap, Geländemodell) | Teil 3: `lade_layer`, WFS-Abfragen; Teil 5: DGM |
| **Cost** | Kostentabelle je Layer erzeugen (*Seed*) und bearbeiten | Teil 3: `CostAssumptions` |
| **Raster** | Vektordaten in ein Kostenraster umwandeln (*Build cost raster*) | Teil 3: `GeoRasterizer` |
| **Routes** | Quelle und Ziel setzen, Algorithmus und Nachbarschaft wählen, Routen berechnen (*Run routing*) und bearbeiten | Teil 4: `PathFinder.find_route` |
| **Layers** | Tabelle der Kartenebenen: Namen ändern, ein- und ausblenden | – |
| **Attrs** | Ein Objekt auf der Karte anklicken, seine Attribute erscheinen hier | Blick in die Ergebnistabellen |

Die GUI führt euch durch fünf Schritte und hakt sie ab, sobald sie erledigt sind:
**Gebiet zeichnen → Daten laden → Kosten festlegen → Kostenraster bauen → Routen planen.**

Im Reiter **Routes** liegen eingeklappte Karten für die Themen aus den Teilen 5 und 6:

- **Objective & terrain:** Gewichte für Kosten, Länge und Steigung, Geländemodell (DGM) – wie in Teil 5.
  *Die GUI lehnt Geländemodell zusammen mit Delta-Stepping ab* und weist auf das Feld „Algorithm“ hin; die Bibliothek
  fällt dagegen mit einer Warnung auf Dijkstra zurück (Teil 5).
- **Constrained routing (overhead line / towers):** Profil als Formular (Winkel, Spannfeld, Mastkosten), wie in Teil 6.
- **Cost field:** die Kosten von den Quellen zu jeder Zelle als Rasterebene – das Ergebnis der Mehr-Quellen-Rechnung aus Teil 4.
- **Corridor graph:** gemeinsame Gräben für mehrere Routen.

Mit **Save project** speichert die GUI Gebiet, Kostentabelle, Raster, Routen und alle Einstellungen in einen Ordner
(darin liegt die `project.json`); mit **Open project** oder `--project` öffnet ihr es wieder.

### Übung Teil 7

1. Startet die GUI im Terminal (`python -m pyorps.gui`) und zeichnet ein kleines Gebiet (etwa 2 × 2 km) in dem Gebiet
   aus Teil 3 (Kinzigtal bei Biberach im Ortenaukreis, Baden-Württemberg).
2. Ladet die ALKIS-Landnutzung über WFS (die Adresse steht in Teil 3), prüft die Kostentabelle im Reiter **Cost**,
   baut das Raster und berechnet eine Route. Vergleicht sie mit der Route aus dem Notebook.
3. Speichert das Projekt, beendet die GUI und öffnet es mit `--project` wieder."""),
]
