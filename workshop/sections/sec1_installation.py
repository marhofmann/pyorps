# ruff: noqa: E501, B028  (long German prose lines in the cells)
"""Abschnitt 0 und 1: Titel, Agenda, Installation, Prüfung der Umgebung."""

CELLS = [
    ("md", r"""# PYORPS-Workshop: Trassenplanung mit Python

**Vom Kostenraster zur Freileitung** – Workshop für Studierende und Mitarbeitende

Wir planen Kabel- und Freileitungstrassen mit **PYORPS** (*Python for Optimal Routes in Power Systems*):
Aus Geodaten entsteht ein Kostenraster, darauf suchen wir kostenoptimale Routen.
Alles läuft als Python-Code in diesem Notebook; am Ende starten wir die grafische Oberfläche (GUI) lokal.

| Teil | Thema | Was ihr danach könnt | Zeit (grobe Schätzung ohne Übung) |
|:---:|---|---|:---:|
| 1 | Installation und Prüfung | PYORPS und die GUI auf dem eigenen Rechner einrichten | 15–25 min |
| 2 | Grundidee | Raster, Graph, kürzeste Route, Nachbarschaften | 10 min |
| 3 | Kostenraster aus mehreren Layern | Geodaten laden, Kosten zuweisen, Layer überlagern, Annahmen variieren | 35–45 min |
| 4 | Dijkstra und Delta-Stepping | Algorithmen verstehen; 1→1, 1→N, M→N und paarweise rechnen | 45–60 min |
| 5 | Geländemodell (DGM) und 3D-Routing | Steigung als Kosten, 2D und 3D vergleichen | 40–50 min |
| 6 | Freileitung | Masten, Spannfelder, Abstände zum Boden mit `ConstrainedPathFinder` | 50–60 min |
| 7 | GUI lokal betreiben | Dieselbe Planung ohne Code im Browser | 20–30 min |
| 8 | Zusammenfassung | Welches Werkzeug wofür, weiterführende Quellen | 5 min |

Die Zeiten sind eine Schätzung für Einsteiger. Mit den Übungen wird es ein langer Tag; sinnvoll sind zwei Blöcke
(Teil 1 bis 4, dann Teil 5 bis 7). Abschnitte, die für den ersten Kontakt entbehrlich sind, tragen in der Überschrift
**„Vertiefung (optional)“**.

**So arbeitet ihr mit dem Notebook:** Zellen von oben nach unten ausführen (`Umschalt + Enter`).
Die meisten Teile enden mit einer kurzen **Übung**. Die Daten liegen im Ordner `data/`; nach dem ersten Lauf
braucht das Notebook kein Internet mehr.

**Empfohlen:** 8 GB Arbeitsspeicher; ein Durchlauf aller Zellen dauert je nach Rechner etwa 15 bis 30 Minuten (Schätzung).

*Quellen:* Die Kostenwerte und Teile der Erklärungen beruhen auf dem Tutorial zum OSMSES-Workshop 2026
(Ordner `tutorial/` im Repository, [github.com/marhofmann/pyorps](https://github.com/marhofmann/pyorps))."""),

    ("md", r"""---
# Teil 1: Installation

## 1.1 Voraussetzungen

- **Python 3.12 bis 3.15** (64 Bit). Prüfen mit `python --version`.
- **Windows, Linux oder macOS.** Fertige Programmpakete (*Wheels*) gibt es für alle drei; ihr braucht **keinen C++-Compiler**.
  macOS braucht Version 15 oder neuer (Apple Silicon).
- Ein Webbrowser (für die GUI) und eine Internetverbindung für die Installation.

## 1.2 Virtuelle Umgebung und Installation

Wir installieren in eine eigene Umgebung, damit nichts mit anderen Projekten kollidiert.
Die folgenden Befehle gebt ihr in einem **Terminal** ein (nicht in dieser Zelle).

**Windows (PowerShell):**
```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install "pyorps[gui,examples]"
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "pyorps[gui,examples]"
```

Statt `-3.13` geht jede installierte Version von 3.12 bis 3.15 (`py -0` listet die installierten Versionen auf).
Der Zusatz in eckigen Klammern (`[gui,examples]`, englisch *extras*) holt zusätzliche Pakete: die Oberfläche (Dash, Karte, Tabellen) und Jupyter (`notebook`).
Weitere Zusätze: `[gpu]` (NVIDIA-Grafikkarte, CuPy), `[graph]` (weitere Graph-Bibliotheken), `[simplify]` (Routenvereinfachung).

## 1.3 Workshop-Dateien holen und Jupyter starten

```bash
git clone https://github.com/marhofmann/pyorps.git
cd pyorps/workshop
jupyter notebook
```

Der Ordner `workshop/` liegt im Repository. Ohne Git geht auch der ZIP-Download: auf der GitHub-Seite
*Code → Download ZIP*, entpacken und im Ordner `workshop/` weiterarbeiten.

> **Wichtig:** Die Pfade im Notebook sind relativ (`data/...`). Das Notebook muss deshalb in seinem Ordner `workshop/`
> laufen. Die Prüfzelle unten meldet, ob der Ordner `data/` gefunden wird.

**Nur für Entwickler:** Wer den neuesten Stand aus GitHub braucht, installiert mit `pip install -e ".[dev,full]"` aus dem Repository;
dafür sind ein C++-Compiler und `python setup.py build_ext --inplace` nötig (siehe die Installationsseite der Dokumentation)."""),

    ("md", r"""## 1.4 Prüfen, ob alles da ist

Die nächste Zelle meldet Python, Betriebssystem, die Version von PYORPS und ob die übersetzten Erweiterungen
(Dijkstra, Delta-Stepping) geladen werden. Sie zeigt auch, welche optionalen Teile vorhanden sind."""),

    ("code", r"""import platform
from pathlib import Path

import pyorps

version = pyorps.__version__
ort = Path(pyorps.__file__).resolve().parent
print(f"Python  {platform.python_version()}  auf {platform.system()} ({platform.machine()})")
print(f"pyorps  {version}")
print(f"Ort     {ort}")

# Mindestversion: die Funktionen dieses Workshops gibt es ab 0.5.0
teile = []
for zahl in version.replace("rc", ".").replace("a", ".").replace("b", ".").split(".")[:3]:
    teile.append(int(zahl) if zahl.isdigit() else 0)
if tuple(teile) < (0, 5, 0):
    print("\n⚠ Version zu alt: bitte  pip install --upgrade \"pyorps[gui,examples]\"  ausführen.")
if "site-packages" not in str(ort):
    print("\nHinweis: pyorps kommt nicht aus site-packages, sondern aus einem Quellordner "
          "(Entwicklerinstallation).")
if not Path("data/profiles").is_dir():
    print(f"\n⚠ Arbeitsordner {Path.cwd()} enthält keinen Ordner data/: "
          "bitte das Notebook im Ordner workshop/ ausführen.")

# Übersetzte Erweiterungen (Dijkstra, Delta-Stepping)
try:
    from pyorps.utils import path_algorithms  # noqa: F401
    print("\nErweiterungen:  ok (Dijkstra und Delta-Stepping geladen)")
except ImportError as fehler:
    print("\nErweiterungen:  FEHLT –", fehler)

# Optionale Teile
import warnings

for name, zusatz in (("dash", "gui"), ("cupy", "gpu"), ("yaml", "gui")):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # CuPy meldet ohne CUDA-Pfad eine Warnung
            __import__(name)
        print(f"{name:<14}  ok")
    except ImportError:
        print(f"{name:<14}  nicht installiert (Zusatz [{zusatz}])")"""),

    ("md", r"""Sieht die Ausgabe gut aus (Version ab 0.5.0, Erweiterungen `ok`), kann es losgehen.
`cupy` ist nur mit NVIDIA-Grafikkarte nötig; wir rechnen im Workshop auf dem Prozessor.

### Übung Teil 1

1. Welche Python-Version und welche PYORPS-Version habt ihr? Notiert beides – bei Fragen hilft das.
2. Führt im Terminal `python -m pyorps.gui --help` aus und lest, welche Optionen es gibt (Teil 7 braucht sie)."""),
]
