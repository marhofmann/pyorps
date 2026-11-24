# ruff: noqa: E501, B028  (long German prose lines in the cells)
"""Abschnitt 4 (zweite Hälfte): die Läufe. Dijkstra und Delta-Stepping, eine Quelle und mehrere Quellen, Kostenfelder."""

CELLS = [
    # ------------------------------------------------------------------ 1: Modi
    ("md", r"""---
## Die Läufe mit PYORPS

Jetzt rechnen wir mit PYORPS auf dem Kostenraster `RASTER_A` aus Teil 3, **immer mit beiden Algorithmen**, und vergleichen
Kosten, Route und Rechenzeit. Alle Tabellen entstehen beim Ausführen der Zellen. Die Sekunden hängen von eurem Rechner ab:
von der Zahl der Kerne, davon, was nebenher läuft, und von der Rastergröße. Zählen sollen **eure** Zahlen auf **eurem** Laptop.

**Zeit:** Dieser Teil rechnet je nach Rechner einige Minuten. Die Zellen für Δ und Threads (Abschnitt „“) könnt ihr überspringen;
die Zusammenfassung am Ende läuft auch ohne sie.

## Welche Routing-Arten gibt es?

PYORPS wählt die Art automatisch aus der Zahl der Quellen und Ziele. Entscheidend für die Rechenzeit ist, **wie viele Suchen** dahinterstehen.
Eine Suche liefert die billigsten Kosten von **einer** Quelle zu allen Zellen. Das heißt *Single-Source Shortest Path* (SSSP).

| Art | Eingabe | Ergebnis | Was PYORPS rechnet |
|---|---|---|---|
| 1→1 | eine Quelle, ein Ziel | `Path` | **eine** Suche von der Quelle; sie hört auf, sobald das Ziel feststeht |
| 1→N | eine Quelle, N Ziele | `PathCollection` mit N Routen | **eine** Suche von der Quelle (SSSP); alle N Ziele werden aus ihr abgelesen |
| M→1 | M Quellen, ein Ziel | M Routen | **M** Suchen, eine pro Quelle |
| M→N | M Quellen, N Ziele | M·N Routen, Quelle für Quelle | **M** Suchen, eine pro Quelle; jede liest N Ziele ab |
| paarweise | `pairwise=True`, gleich viele Quellen und Ziele | N Routen (Quelle *i* zu Ziel *i*) | eine Suche **pro Paar** (Dijkstra fasst Paare zusammen, die einen Punkt teilen) |

**„“ heißt hier:** mehrere Quellen, aber *getrennte* Suchen. `find_route` startet nie eine einzige Suche, die von allen Quellen
zugleich losläuft. Die Mehr-Quellen-Rechnung bauen wir weiter unten selbst: für jede Zelle die Kosten zur **billigsten** Quelle.

Beide Algorithmen sind exakt. Die Kosten sollten also gleich sein. Das prüfen wir in jeder Zelle und **drucken das Ergebnis der Prüfung** (`gleiche Kosten: ja` oder `nein`).
Nur wenn es dort `nein` heißt, erklärt die Zelle, woran das liegen kann."""),

    # ------------------------------------------------------------------ 2: Setup
    ("code", r'''import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.transform import rowcol

from pyorps import PathFinder, full_window_buffer_m

# Raster und Punkte aus Teil 3 (falls Teil 3 nicht ausgeführt wurde)
if not all(n in globals() for n in ("RASTER_A", "RASTER_B", "SOURCE", "TARGET", "SOURCES", "TARGETS")):
    RASTER_A = "data/cost_raster_a.tiff"
    RASTER_B = "data/cost_raster_b.tiff"
    SOURCE = (427558, 5357403)
    TARGET = (430288, 5355723)
    SOURCES = [SOURCE, (428238, 5358313), (428663, 5356873)]
    TARGETS = [TARGET, (431613, 5355083), (429853, 5355148), (431273, 5356048)]
VERBOTEN = 65535                      # Zellwert für gesperrte Zellen


def de(x, nd=0):
    # Zahl im deutschen Format für Sätze: 3.206 m, 3,5 s
    return f"{x:,.{nd}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def dz(x, nd=0):
    # Zahl für Tabellen: Dezimalkomma, keine Tausenderpunkte
    return f"{x:.{nd}f}".replace(".", ",")


def de_e(x):
    # Kleine Zahlen in Schreibweise 1,6e-15 (deutsches Komma)
    return f"{x:.1e}".replace(".", ",")


# Threads für Delta-Stepping: höchstens drei Viertel der Kerne, damit der Rechner nebenbei benutzbar bleibt
KERNE = os.cpu_count() or 2
THREADS = max(1, int(KERNE * 0.75))
OMP_GRENZE = os.environ.get("OMP_NUM_THREADS", "")
if OMP_GRENZE.isdigit():              # eine vorgegebene Obergrenze gilt
    THREADS = min(THREADS, int(OMP_GRENZE))

TOLERANZ = 1e-6                       # erlaubter relativer Kostenunterschied bei Routen (Kosten aus den Zellen der Route)
TOLERANZ_FELD = 1e-5                  # bei Kostenfeldern größer, siehe unten
ALGOS = ("dijkstra", "delta-stepping")
ZEITEN, CPU, ERGEBNISSE = {}, {}, {}  # Sekunden, CPU-Zeit / Uhrzeit und Ergebnisse, gefüllt von den Zellen unten

ERKLAERUNG = ("  Erklärung: Dijkstra ist die Referenz. Delta-Stepping mit mehreren Threads kann in seltenen Fällen eine etwas teurere Route\n"
              "  liefern (ein Kommentar im Quellcode von PYORPS nennt diese Möglichkeit für den Kern von 1→1 bei zwei und mehr Threads). Das ist hier kein Fehler des Notebooks.\n"
              "  Rechnet mit `num_threads=1` nach, dann gleichen sich die Kosten.")

# Das Kostenraster einmal für die Bilder lesen
with rasterio.open(RASTER_A) as src:
    kosten_a = src.read(1)
    transform_a = src.transform
    EXTENT = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)


def neuer_finder(raster, nachbarschaft="r2"):
    # Das ganze Raster ist das Suchfenster (wie in Teil 3)
    return PathFinder(dataset_source=raster, source_coords=SOURCE, target_coords=TARGET,
                      search_space_buffer_m=full_window_buffer_m(raster), neighborhood_str=nachbarschaft)


FINDER = neuer_finder(RASTER_A)


def lauf(algo, source, target, finder=None, **kw):
    # Ein find_route-Aufruf; Quelle, Ziel und Einstellungen werden pro Aufruf übergeben
    finder = FINDER if finder is None else finder
    if algo == "delta-stepping":
        kw.setdefault("num_threads", THREADS)
    return finder.find_route(source=source, target=target, algorithm=algo, **kw)


def zeit(text, aufruf, key=None, wiederholungen=3):
    # Ein Aufwärm-Lauf, dann das Minimum aus drei Läufen. Der erste Aufruf kann langsamer sein: er lädt das Raster,
    # legt den Suchspeicher an und füllt die Caches des Prozessors (kompiliert wird nichts, die Kerne sind fertig übersetzt).
    t0 = time.perf_counter()
    aufruf()
    erst = time.perf_counter() - t0
    zeiten = []
    c0, w0 = time.process_time(), time.perf_counter()
    for _ in range(wiederholungen):
        t0 = time.perf_counter()
        ergebnis = aufruf()
        zeiten.append(time.perf_counter() - t0)
    cpu_faktor = (time.process_time() - c0) / (time.perf_counter() - w0)   # etwa 1 bei einem Thread, etwa 2 bei zwei
    sek = min(zeiten)
    print(f"{text:<46s} {de(sek, 3):>7s} s   (erster Aufruf {de(erst, 3)} s)")
    if key is not None:
        ZEITEN[key], CPU[key] = sek, cpu_faktor
    return sek, ergebnis


def als_liste(ergebnis):
    # Path (eine Route) oder PathCollection (mehrere Routen) als Liste von Path
    return [ergebnis] if hasattr(ergebnis, "total_cost") else list(ergebnis)


def pruefe_reihenfolge(routen, quellen, ziele, paarweise=False):
    # PYORPS liefert die Routen in Eingabereihenfolge, bei M→N Quelle für Quelle. Das prüfen wir an den Endpunkten.
    # (Diese Prüfung hängt nicht von Threads ab, deshalb darf sie hart sein.)
    erwartet = list(zip(quellen, ziele)) if paarweise else [(s, z) for s in quellen for z in ziele]
    assert len(routen) == len(erwartet), "falsche Zahl von Routen"
    for p, (s, z) in zip(routen, erwartet):
        assert np.allclose(p.source, s, atol=5) and np.allclose(p.target, z, atol=5), "Reihenfolge der Routen stimmt nicht"


def vergleiche(routen_d, routen_s):
    # Kosten beider Algorithmen: Arrays, größter relativer Unterschied, Zahl zellgleicher Routen
    kd = np.array([p.total_cost for p in routen_d])
    ks = np.array([p.total_cost for p in routen_s])
    rel = float(np.max(np.abs(kd - ks) / kd))
    gleich = sum(np.array_equal(a.path_indices, b.path_indices) for a, b in zip(routen_d, routen_s))
    return kd, ks, rel, gleich


def pruefe_kosten(name, kd, ks, tol=TOLERANZ):
    # Sichtbare Prüfung statt assert: Dijkstra (kd) ist die Referenz, Delta-Stepping (ks) wird verglichen
    kd, ks = np.asarray(kd, float), np.asarray(ks, float)
    diff = np.abs(kd - ks)
    rel = float(np.max(diff / kd))
    ok = rel < tol
    print(f"{name}: gleiche Kosten: {'ja' if ok else 'nein'}, größter Unterschied {de_e(float(diff.max()))} EUR "
          f"(relativ {de_e(rel)}, Toleranz {de_e(tol)})")
    if not ok:
        print(ERKLAERUNG)
    return ok


def hintergrund(ax):
    # Kostenraster in Grau, gesperrte Zellen rot
    bild = np.where(kosten_a == VERBOTEN, np.nan, kosten_a.astype(float))
    ax.imshow(bild, cmap="Greys", vmin=0, vmax=800, extent=EXTENT, alpha=0.55)
    gesperrt = np.ma.masked_where(kosten_a != VERBOTEN, kosten_a)
    ax.imshow(gesperrt, cmap=ListedColormap(["#b03030"]), extent=EXTENT, alpha=0.6)
    ax.set_xlabel("Rechtswert [m]", fontsize=9)
    ax.set_ylabel("Hochwert [m]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(style="plain", useOffset=False)


print(f"Rechner: {KERNE} logische Kerne.")
print(f"Threads für Delta-Stepping: {THREADS} (drei Viertel von {KERNE} Kernen = {int(KERNE * 0.75)}; "
      f"Obergrenze durch OMP_NUM_THREADS: {OMP_GRENZE if OMP_GRENZE else 'nicht gesetzt'}).")
print(f"Raster: {kosten_a.shape[1]} × {kosten_a.shape[0]} Zellen, {de(100 * np.mean(kosten_a == VERBOTEN), 1)} % gesperrt.")'''),

    # ------------------------------------------------------------------ 3: 1->1
    ("md", r"""## 1→1: ein Paar, zwei Algorithmen

`PathFinder` hält das Raster und die Einstellungen. Quelle, Ziel und Algorithmus geben wir bei jedem `find_route`-Aufruf neu an
(`source=`, `target=`, `algorithm=`); das Raster wird dabei nicht neu geladen.
Eigene Einstellungen des Algorithmus (`delta`, `num_threads`) gehen ebenfalls als Schlüsselwort an `find_route`.
Einen Tippfehler wie `detla=3` zeigen wir unten: PYORPS bricht mit einer Fehlermeldung ab. Sie ist englisch; „“ heißt „“.

Die Hilfsfunktion `zeit` misst jeden Aufruf nach einem Aufwärm-Lauf dreimal und nimmt das Minimum.
Der erste Aufruf kann langsamer sein, weil er das Raster lädt, Speicher für die Suche anlegt und die Caches des Prozessors füllt (kompiliert wird nichts).
Die Angabe „“ zeigt, wie groß dieser Unterschied bei euch ist."""),

    ("code", r'''zeilen, routen = [], {}
for algo in ALGOS:
    sek, p = zeit(f"1→1 {algo}", lambda a=algo: lauf(a, SOURCE, TARGET), key=("1→1", algo))
    routen[algo] = p
    zeilen.append((algo, dz(p.total_cost, 2), dz(p.total_length, 1), dz(sek, 3)))
ERGEBNISSE["1→1"] = {a: [routen[a]] for a in ALGOS}

print()
print(pd.DataFrame(zeilen, columns=["Algorithmus", "Kosten [EUR]", "Länge [m]", "Sekunden"]).to_string(index=False))
print()

d, s = routen["dijkstra"], routen["delta-stepping"]
pruefe_kosten("1→1", [d.total_cost], [s.total_cost])

gleich = np.array_equal(d.path_indices, s.path_indices)
abstand = d.path_geometry.hausdorff_distance(s.path_geometry)
print(f"Dieselben Zellen: {'ja' if gleich else 'nein'}; größter Abstand der beiden Linien: {de(abstand, 1)} m")

# Ein Tippfehler im Namen einer Einstellung: PYORPS meldet ihn (die Meldung ist englisch)
print("\nTippfehler-Test mit `detla=3`. Die englische Meldung nennt den falschen Namen und schlägt den richtigen vor:")
try:
    FINDER.find_route(source=SOURCE, target=TARGET, algorithm="delta-stepping", detla=3)
    print("  keine Fehlermeldung (in dieser PYORPS-Version wird der Name nicht geprüft)")
except TypeError as fehler:
    print(f"  {type(fehler).__name__}: {fehler}")'''),

    # ------------------------------------------------------------------ 4: Deutung + 1->N
    ("md", r"""**Wie lesen wir das?** Die Kosten einer Route berechnet `find_route` am Ende aus den Zellen der Route, bei beiden Algorithmen gleich.
Deshalb stimmen sie so genau überein. Steht oben bei „“ ein *nein*, dann haben die Algorithmen **zwei gleich teure Routen** gefunden.
Das kommt auf einem Raster mit großen gleichförmigen Flächen (Wald, Acker) oft vor: Dort kosten viele Folgen von Schritten genau gleich viel.
Welche von zwei gleich billigen Routen zuerst gefunden wird, hängt von der Reihenfolge ab, in der die Nachbarzellen abgearbeitet werden.
Eine Route ist also nicht eindeutig, **ihre Kosten schon**.

Steht oben bei „“ ein *nein*, dann ist Dijkstra die Referenz (siehe die Erklärung, die die Zelle dann druckt).

## 1→N: eine Quelle, vier Ziele

Hier lohnt sich die Idee der Single-Source-Suche: **Eine** Suche von `SOURCE` liest alle vier Ziele ab. Die Tabelle zeigt die Kosten pro Ziel
für beide Algorithmen. Das Bild zeigt die vier Routen auf dem Kostenraster (grau, gesperrte Zellen rot): Dijkstra durchgezogen,
Delta-Stepping gestrichelt darüber."""),

    ("code", r'''sek = {}
pc = {}
for algo in ALGOS:
    sek[algo], pc[algo] = zeit(f"1→N {algo}", lambda a=algo: lauf(a, SOURCE, TARGETS), key=("1→N", algo))
    pruefe_reihenfolge(als_liste(pc[algo]), [SOURCE], TARGETS)
ERGEBNISSE["1→N"] = {a: list(pc[a]) for a in ALGOS}

kd, ks, rel, gleich = vergleiche(ERGEBNISSE["1→N"]["dijkstra"], ERGEBNISSE["1→N"]["delta-stepping"])
tabelle = pd.DataFrame({
    "Ziel": [f"Ziel {j + 1} (TARGETS[{j}])" for j in range(len(TARGETS))],
    "Länge [m]": [dz(p.total_length, 1) for p in ERGEBNISSE["1→N"]["dijkstra"]],
    "Kosten Dijkstra [EUR]": [dz(x, 2) for x in kd],
    "Kosten Delta-Stepping [EUR]": [dz(x, 2) for x in ks],
})
print()
print(tabelle.to_string(index=False))
print()
pruefe_kosten("1→N", kd, ks)
print(f"zellgleiche Routen: {gleich} von {len(kd)}")

# Verhältnis: eine Suche für vier Ziele gegenüber einer Suche für ein Ziel
for algo in ALGOS:
    print(f"{algo}: 1→N dauert das {de(ZEITEN[('1→N', algo)] / ZEITEN[('1→1', algo)], 1)}-fache von 1→1 (vier Ziele, eine Suche)")

fig, ax = plt.subplots(figsize=(8, 6))
hintergrund(ax)
for j, (pd_, ps_) in enumerate(zip(ERGEBNISSE["1→N"]["dijkstra"], ERGEBNISSE["1→N"]["delta-stepping"])):
    farbe = f"C{j}"
    ax.plot(*pd_.path_geometry.xy, color=farbe, lw=3, label=f"Ziel {j + 1}: {de(pd_.total_length)} m, {de(pd_.total_cost / 1000)} kEUR")
    ax.plot(*ps_.path_geometry.xy, color="black", lw=0.8, ls="--")
    ax.plot(*TARGETS[j], "s", ms=8, mfc="white", mec="black")
    ax.annotate(str(j + 1), TARGETS[j], xytext=(7, 7), textcoords="offset points", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.5))
ax.plot(*SOURCE, "o", ms=10, mfc="white", mec="black", label="SOURCE")
ax.set_title("1→N: vier Routen von SOURCE (durchgezogen: Dijkstra, gestrichelt: Delta-Stepping)", fontsize=9)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)   # unter der Karte, verdeckt keine Route
plt.tight_layout()
plt.show()'''),

    # ------------------------------------------------------------------ 5: M->N Text
    ("md", r"""## M→N und paarweise: viele Quellen

Mit drei Quellen (`SOURCES`) und vier Zielen (`TARGETS`) entstehen **12 Routen**. PYORPS startet drei Suchen, eine pro Quelle, und liest jeweils vier Ziele ab.
Die Routen kommen Quelle für Quelle zurück; `pruefe_reihenfolge` bestätigt das an den Endpunkten der Routen. Die Matrix unten hat deshalb
eine Zeile pro Quelle und eine Spalte pro Ziel.

Danach folgen zwei kleinere Fälle: **M→1** (drei Quellen, ein Ziel) und **paarweise** (die drei Quellen mit den ersten drei Zielen, also nur drei Routen).
Bei „“ geben wir `num_threads` mit, PYORPS **ignoriert** es dort aber; die Zelle zeigt das am Ende mit der Spalte *CPU/Uhr*."""),

    ("code", r'''pc = {}
for algo in ALGOS:
    _, pc[algo] = zeit(f"M→N {algo} (12 Routen)", lambda a=algo: lauf(a, SOURCES, TARGETS), key=("M→N", algo))
    pruefe_reihenfolge(list(pc[algo]), SOURCES, TARGETS)
ERGEBNISSE["M→N"] = {a: list(pc[a]) for a in ALGOS}

kd, ks, rel, gleich = vergleiche(ERGEBNISSE["M→N"]["dijkstra"], ERGEBNISSE["M→N"]["delta-stepping"])
form = (len(SOURCES), len(TARGETS))
zeilen_namen = [f"SOURCES[{i}]" for i in range(form[0])]
spalten_namen = [f"TARGETS[{j}]" for j in range(form[1])]
for name, werte in (("Dijkstra", kd), ("Delta-Stepping", ks)):
    print(f"\nKosten {name} [EUR]")
    print(pd.DataFrame(werte.reshape(form), index=zeilen_namen, columns=spalten_namen).to_string(float_format=lambda x: f"{x:.0f}"))

print()
pruefe_kosten("M→N (12 Routen)", kd, ks)
print(f"zellgleiche Routen: {gleich} von {len(kd)}")'''),

    ("code", r'''def beide(modus, text, quellen, ziele, paarweise=False):
    # Einen Fall mit beiden Algorithmen rechnen; Reihenfolge der Routen prüfen, Kosten sichtbar vergleichen
    ERGEBNISSE[modus] = {}
    ziel_liste = [ziele] if isinstance(ziele, tuple) else list(ziele)
    for algo in ALGOS:
        _, ergebnis = zeit(f"{modus} {algo} ({text})", lambda a=algo: lauf(a, quellen, ziele, pairwise=paarweise),
                           key=(modus, algo))
        ERGEBNISSE[modus][algo] = list(ergebnis)
        pruefe_reihenfolge(ERGEBNISSE[modus][algo], quellen, ziel_liste, paarweise=paarweise)
    kd, ks, rel, _ = vergleiche(ERGEBNISSE[modus]["dijkstra"], ERGEBNISSE[modus]["delta-stepping"])
    return kd, ks


kd1, ks1 = beide("M→1", "3 Routen", SOURCES, TARGET)                                   # drei Quellen, ein Ziel
kdp, ksp = beide("paarweise", "3 Routen", SOURCES, TARGETS[:3], paarweise=True)        # SOURCES[i] nach TARGETS[i]

print()
print(pd.DataFrame({
    "M→1: Quelle → TARGET": [f"SOURCES[{i}]" for i in range(len(SOURCES))],
    "Kosten Dijkstra [EUR]": [f"{x:.0f}" for x in kd1], "Kosten Delta-Stepping [EUR]": [f"{x:.0f}" for x in ks1],
}).to_string(index=False))
pruefe_kosten("M→1", kd1, ks1)
print()
print(pd.DataFrame({
    "paarweise: Paar": [f"SOURCES[{i}] → TARGETS[{i}]" for i in range(len(SOURCES))],
    "Kosten Dijkstra [EUR]": [f"{x:.0f}" for x in kdp], "Kosten Delta-Stepping [EUR]": [f"{x:.0f}" for x in ksp],
}).to_string(index=False))
pruefe_kosten("paarweise", kdp, ksp)

# Hat num_threads bei „“ eine Wirkung? CPU/Uhr zeigt, wie viele Threads wirklich arbeiten.
print()
zeit("paarweise delta-stepping, num_threads=1 verlangt",
     lambda: lauf("delta-stepping", SOURCES, TARGETS[:3], pairwise=True, num_threads=1), key=("paarweise", "1 Thread verlangt"))
print(f"CPU/Uhr bei verlangtem 1 Thread: {de(CPU[('paarweise', '1 Thread verlangt')], 1)}; "
      f"mit {THREADS} verlangten Threads: {de(CPU[('paarweise', 'delta-stepping')], 1)}. "
      f"Wäre `num_threads` wirksam, läge die erste Zahl nahe 1,0.")'''),

    # ------------------------------------------------------------------ 7: Kostenfeld Text
    ("md", r"""## Das Kostenfeld: Kosten von der Quelle zu **jeder** Zelle

Bisher haben wir nur Routen gelesen. Die Suche selbst kennt aber mehr: den Preis jeder Zelle. Dieses *Kostenfeld* liefert `finder.cost_field(punkt, algorithm=...)`
(Kostenfelder gibt es in PYORPS seit Version 0.4, sie sind als *experimentell* markiert).
`field_array()` liefert das Feld als Array in der Form des Suchfensters, hier das ganze Raster (1000 × 1000 Zellen). Zellen, die nicht erreichbar sind, stehen auf `inf`.
Beide Algorithmen können das Feld rechnen, auch Dijkstra: Er läuft dafür bis zum Ende und hört nicht beim ersten Ziel auf.

Die **Mehr-Quellen-Rechnung** bauen wir selbst: ein Feld pro Quelle, dann in jeder Zelle das **Minimum**. Das ist genau das Ergebnis einer Suche,
die von allen Quellen zugleich startet. Die Isolinien verbinden Zellen mit gleichen Kosten. Darüber liegt die Route `SOURCE`→`TARGET` aus der ersten Rechnung.
Die Kosten sind in EUR wie bei den Routen.

**Warum ist die erlaubte Abweichung bei Feldern größer als bei Routen?** Die Kosten einer Route werden aus ihren Zellen neu berechnet. Ein Feld wird nicht neu berechnet,
es enthält die Werte der Suche selbst. Die PYORPS-Dokumentation zu Kostenfeldern nennt für Delta-Stepping 32-Bit-Zahlen als Grund für kleine Abweichungen gegenüber Dijkstra.
Wir erwarten bei Feldern deshalb einen kleinen Unterschied (relativ etwa 1e-6 bis 1e-5). Für Felder gilt die Toleranz 1e-5, für Routen 1e-6.
Die Zelle druckt zur Einordnung den Abstand zweier 32-Bit-Zahlen in dieser Größe. Er allein erklärt den größten Unterschied nicht ganz: Der Unterschied ist ein Vielfaches davon.
Die Reihenfolge der Summation entlang der Route kommt vermutlich dazu (nicht geprüft)."""),

    ("code", r'''def feld(punkt, algo):
    # Kostenfeld ab einem Punkt: (Zeilen, Spalten), in EUR, inf = nicht erreichbar
    kw = {"num_threads": THREADS} if algo == "delta-stepping" else {}
    with FINDER.cost_field(punkt, algorithm=algo, **kw) as cf:
        return cf.field_array(np.float64)


einzel, mehr = {}, {}
for algo in ALGOS:
    _, einzel[algo] = zeit(f"Kostenfeld ab SOURCE {algo}", lambda a=algo: feld(SOURCE, a), key=("Kostenfeld", algo))
    t0 = time.perf_counter()
    mehr[algo] = np.minimum.reduce([feld(q, algo) for q in SOURCES])
    print(f"{'Minimum über 3 Quellen ' + algo:<46s} {de(time.perf_counter() - t0, 3):>7s} s   (drei Felder, ein Lauf)")

# Beide Algorithmen: dasselbe Feld? (sichtbare Prüfung, kein assert)
print()
for name, fd, fs in (("ab SOURCE", einzel["dijkstra"], einzel["delta-stepping"]),
                     ("Minimum über SOURCES", mehr["dijkstra"], mehr["delta-stepping"])):
    beide_erreichbar = np.isfinite(fd) & np.isfinite(fs)
    verschieden = int((np.isfinite(fd) != np.isfinite(fs)).sum())
    max_abs = float(np.max(np.abs(fd[beide_erreichbar] - fs[beide_erreichbar])))
    kmax = float(fd[beide_erreichbar].max())
    rel = max_abs / kmax
    ok = verschieden == 0 and rel < TOLERANZ_FELD
    print(f"Feld {name}: {de(int(beide_erreichbar.sum()))} erreichbare Zellen, {verschieden} Zellen nur von einem Algorithmus erreicht; "
          f"gleiche Kosten: {'ja' if ok else 'nein'}, größter Unterschied {de(max_abs, 2)} EUR bei Kosten bis {de(kmax)} EUR "
          f"(relativ {de_e(rel)}, Toleranz {de_e(TOLERANZ_FELD)})")
    if not ok:
        print(ERKLAERUNG)
vmax = float(np.max(einzel["dijkstra"][np.isfinite(einzel["dijkstra"])]))
print(f"Zur Einordnung: Zwei benachbarte 32-Bit-Zahlen in der Größe {de(vmax)} liegen etwa {de(float(np.spacing(np.float32(vmax))), 3)} auseinander.")

# Das Feld am Ziel ist der Preis der Route
zeile, spalte = rowcol(transform_a, *TARGET)
print(f"Feld am Ziel: {de(einzel['dijkstra'][zeile, spalte], 2)} EUR; Routenkosten (1→1): {de(ERGEBNISSE['1→1']['dijkstra'][0].total_cost, 2)} EUR")

route = ERGEBNISSE["1→1"]["dijkstra"][0]
stufen = np.arange(200_000, vmax, 200_000)
fig, achsen = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)
for ax, f, titel in ((achsen[0], einzel["dijkstra"], "ab SOURCE"), (achsen[1], mehr["dijkstra"], "ab der billigsten der 3 Quellen")):
    bild = ax.imshow(np.where(np.isfinite(f), f, np.nan) / 1000, cmap="YlOrRd", vmin=0, vmax=vmax / 1000, extent=EXTENT)
    linien = ax.contour(np.where(np.isfinite(f), f, np.nan) / 1000, levels=stufen / 1000, colors="black", linewidths=0.6, extent=EXTENT, origin="upper")
    ax.clabel(linien, fmt="%d", fontsize=8)
    ax.plot(*route.path_geometry.xy, color="tab:blue", lw=2, label="Route SOURCE→TARGET")
    ax.plot(*zip(*SOURCES), "o", ms=7, mfc="white", mec="black", label="Quellen")
    ax.plot(*TARGET, "s", ms=7, mfc="white", mec="black", label="TARGET")
    ax.set_title(f"Kosten {titel} (Isolinien alle 200 kEUR)", fontsize=9)
    ax.set_xlabel("Rechtswert [m]", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(style="plain", useOffset=False)
    balken = plt.colorbar(bild, ax=ax, fraction=0.046, pad=0.02)
    balken.set_label("Kosten [kEUR]", fontsize=9)
    balken.ax.tick_params(labelsize=8)
achsen[0].set_ylabel("Hochwert [m]", fontsize=9)
achsen[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.show()'''),

    # ------------------------------------------------------------------ 9: Vertiefung Text
    ("md", r"""## Vertiefung (optional): die Stellschrauben `delta` und `num_threads`

Dieser Abschnitt ist freiwillig. Die beiden Zellen brauchen am meisten Zeit; wer sie überspringt, kann bei der Zusammenfassung direkt weitermachen.

Beide Einstellungen gehen als Schlüsselwort an `find_route`. Wir nehmen den M→N-Fall (12 Routen) und ändern jeweils **eine** Größe.

**Was ist Δ (`delta`) genau?** Die Einheit sind die **Zellwerte des Rasters** (hier EUR pro Meter), nicht Meter und nicht EUR.
Das Kantengewicht eines Schritts ist der **Mittelwert der Zellwerte entlang des Schritts mal die Länge des Schritts in Zellen**:
ein waagerechter Schritt zwischen zwei Zellen mit dem Wert 365 wiegt 365, ein diagonaler Schritt 365·√2.
Die gemeldeten Kosten in EUR sind dieses Gewicht mal die Zellgröße (5 m). **Δ = 100 sind also 500 EUR Routenkosten**, und das ist die Voreinstellung.
Eine Kante gilt als *kurz* (wird im Eimer wiederholt relaxiert), wenn ihr Gewicht höchstens Δ beträgt, sonst als *lang*.
Die Zelle druckt zuerst Median und Mittelwert der freien Zellen von `RASTER_A`. Die Tabelle zeigt in der zweiten Spalte, welcher Anteil der freien Zellen überhaupt höchstens Δ kostet
(eine Faustregel, denn ein Schritt über mehrere Zellen wiegt mehr).

`num_threads` ist die Zahl der Threads (Rechenstränge) auf dem Prozessor. Die Spalte *CPU/Uhr* ist die verbrauchte Rechenzeit geteilt durch die Uhrzeit:
etwa 1 heißt, es arbeitet ein Kern; etwa 2 heißt, es arbeiten zwei. So sehen wir **unabhängig vom Zeitrauschen**, ob die Threads wirklich laufen."""),

    ("code", r'''frei = kosten_a[kosten_a != VERBOTEN]
print(f"Freie Zellen von RASTER_A: Median {de(np.median(frei))}, Mittelwert {de(frei.mean(), 1)}.")
referenz = np.array([p.total_cost for p in ERGEBNISSE["M→N"]["dijkstra"]])


def kosten_von(pc):
    return np.array([p.total_cost for p in pc])


# 1) Eimerbreite Δ, feste Threadzahl
zeilen = []
for delta in (10, 100, 1000, 10000):
    key = ("delta", delta)
    sek, pc = zeit(f"Δ = {delta:>5}, {THREADS} Threads", lambda d=delta: lauf("delta-stepping", SOURCES, TARGETS, delta=d), key=key)
    rel = float(np.max(np.abs(kosten_von(pc) - referenz) / referenz))
    zeilen.append((delta, f"{dz(100 * np.mean(frei <= delta), 1)} %", dz(sek, 3), dz(CPU[key], 2), de_e(rel), "ja" if rel < TOLERANZ else "nein"))
print()
print(pd.DataFrame(zeilen, columns=["Δ", "Zellen mit Wert ≤ Δ", "Sekunden", "CPU/Uhr", "max. rel. Kostenabweichung", "gleiche Kosten"]).to_string(index=False))
if any(z[-1] == "nein" for z in zeilen):
    print(ERKLAERUNG)

# 2) Threadzahl, Δ = 100
print()
zeilen = [("Dijkstra (ein Thread)", dz(ZEITEN[("M→N", "dijkstra")], 3), dz(CPU[("M→N", "dijkstra")], 2), "-", "Referenz")]
for n in sorted({t for t in (1, 2, THREADS) if t <= KERNE}):
    key = ("threads", n)
    sek, pc = zeit(f"Delta-Stepping, Δ = 100, {n} Thread(s)", lambda n=n: lauf("delta-stepping", SOURCES, TARGETS, num_threads=n), key=key)
    rel = float(np.max(np.abs(kosten_von(pc) - referenz) / referenz))
    zeilen.append((f"Delta-Stepping, {n} Thread(s)", dz(sek, 3), dz(CPU[key], 2), de_e(rel), "ja" if rel < TOLERANZ else "nein"))
ZEITEN[("M→N", "delta-1thread")] = ZEITEN[("threads", 1)]
print()
print(pd.DataFrame(zeilen, columns=["Variante", "Sekunden", "CPU/Uhr", "max. rel. Kostenabweichung", "gleiche Kosten"]).to_string(index=False))
if any(z[-1] == "nein" for z in zeilen):
    print(ERKLAERUNG)'''),

    # ------------------------------------------------------------------ 11: Lesehilfe + Zusammenfassung Text
    ("md", r"""**Wie lesen wir diese Tabellen?**

- **Die Ergebnisse sollten gleich sein.** In der Spalte „“ sollte bei jeder Einstellung *ja* stehen. `delta` und `num_threads` ändern **nur die Zeit**, nicht die Kosten.
  Steht dort *nein*, gilt Dijkstra als Referenz (siehe die Erklärung, die die Zelle dann druckt).
- **Wie viel sich die Zeit ändert, hängt von der Maschine ab:** von der Zahl der Kerne, davon, ob andere Programme Kerne belegen, und von der Größe des Rasters.
- **Allgemeine Erfahrung, hier nicht gemessen:** Auf einem Raster mit vielen Millionen Zellen lohnt sich der Aufwand für Eimer und Threads eher als auf diesem mit 1 Million Zellen.
- **Die Δ-Tabelle zeigt den Einfluss von Δ auf eurem Rechner:** Vergleicht die Sekunden bei Δ = 10 und Δ = 10000 mit denen bei Δ = 100 und Δ = 1000. Gemessen ist nur das.
  **Allgemeine Erfahrung, hier nicht gemessen** ist die Begründung: Bei sehr kleinem Δ gibt es viele Eimer mit wenigen Zellen (kaum Parallelität), bei sehr großem Δ wird viel doppelt relaxiert.
  Die Voreinstellung 100 ist ein Anfang, kein Optimum für jedes Raster.
- **Allgemeine Erfahrung, hier nicht gemessen:** Mehr Threads sind nicht immer schneller, weil sie in jedem Eimer aufeinander warten. Wir nehmen deshalb höchstens drei Viertel der Kerne (so macht es PYORPS bei Kostenfeldern selbst).
  Die Thread-Tabelle zeigt nur die Zahlen 1, 2 und `THREADS`.
- **Messt selbst:** Eure Zahlen auf eurem Laptop sind die, die zählen. Kleine Unterschiede können Rauschen sein; wiederholt die Messung, bevor ihr etwas daraus schließt.

## Zusammenfassung und Grenzen

Die Tabelle unten wird aus den Zeiten der Zellen oben gebaut. Die letzte Spalte ist **gerechnet**, nicht behauptet: Sie sagt, welcher Algorithmus auf **diesem** Rechner schneller war.
Die Zeile „“ gibt es nur, wenn ihr die Vertiefung ausgeführt habt; sie zeigt, wie viel vom Vorsprung nur von den Threads kommt und wie viel vom Verfahren selbst. Liegt der Faktor in dieser Zeile deutlich über 1, kommt der Vorsprung nicht nur von den Threads.
Die Zeile „“ ist mit einem Stern markiert: Dort ignoriert PYORPS `num_threads`, die Zeile ist darum **nicht vergleichbar** und zählt in der Auswertung nicht mit.

**Grenzen, die ihr kennen solltet:**

- **Keine Steigungskosten.** Delta-Stepping kennt nur Kosten aus dem Raster. Mit Geländemodell (Teil 5) läuft PYORPS stattdessen mit Dijkstra und warnt (`AlgorithmFallbackWarning`).
- **`pairwise=True` ignoriert `num_threads`.** Die Zelle zu „“ zeigt das mit der Spalte *CPU/Uhr*. Delta-Stepping läuft dort mit der Voreinstellung (alle Kerne, höchstens 12, oder die Obergrenze `OMP_NUM_THREADS`).
- **Threads und Betriebssystem.** Delta-Stepping nutzt mehrere Threads über OpenMP, auch unter Windows: Die Spalte *CPU/Uhr* oben zeigt, wie viele Kerne wirklich arbeiten.
  Unter macOS ist OpenMP laut `setup.py` nur dabei, wenn die Erweiterung mit Homebrew `libomp` gebaut wurde (von uns nicht getestet); wer einen Mac hat, prüft die Spalte *CPU/Uhr* selbst.
- **Mehr Quellen sind getrennte Suchen** (siehe Tabelle am Anfang), außer ihr baut das Minimum aus Kostenfeldern wie oben."""),

    ("code", r'''modi = [("1→1", "1→1: SOURCE → TARGET"), ("1→N", "1→N: SOURCE → 4 Ziele"), ("M→1", "M→1: 3 Quellen → TARGET"),
        ("M→N", "M→N: 3 Quellen → 4 Ziele (12)"), ("paarweise", "paarweise: 3 Paare *"), ("Kostenfeld", "Kostenfeld ab SOURCE")]
zeilen, vergleichbar = [], []
for key, text in modi:
    d, s = ZEITEN[(key, "dijkstra")], ZEITEN[(key, "delta-stepping")]
    threads = f"Standard (CPU/Uhr {de(CPU[(key, 'delta-stepping')], 1)})" if key == "paarweise" else str(THREADS)
    zeilen.append((text, dz(d, 3), dz(s, 3), dz(d / s, 2), threads, "Delta-Stepping" if s < d else "Dijkstra"))
    if key != "paarweise":
        vergleichbar.append(s < d)
if ("M→N", "delta-1thread") in ZEITEN:        # nur, wenn die Vertiefung ausgeführt wurde
    d, s = ZEITEN[("M→N", "dijkstra")], ZEITEN[("M→N", "delta-1thread")]
    zeilen.append(("M→N mit einem Thread", dz(d, 3), dz(s, 3), dz(d / s, 2), "1", "Delta-Stepping" if s < d else "Dijkstra"))
    vergleichbar.append(s < d)

tabelle = pd.DataFrame(zeilen, columns=["Modus", "Dijkstra [s]", "Delta-Stepping [s]", "Faktor Dijkstra/Delta", "Threads Delta-Stepping", "schneller"])
print(tabelle.to_string(index=False))
print("* paarweise: PYORPS ignoriert dort num_threads und nimmt die Voreinstellung (CPU/Uhr zeigt die wirkliche Zahl). "
      "Die Zeile ist darum nicht mit den anderen vergleichbar und zählt unten nicht mit.")
print(f"\nRechner: {KERNE} logische Kerne, Delta-Stepping mit {THREADS} Threads (Obergrenze OMP_NUM_THREADS: {OMP_GRENZE if OMP_GRENZE else 'nicht gesetzt'}), "
      f"Δ = 100, Raster {de(kosten_a.size)} Zellen.")
print(f"Auf diesem Rechner war Delta-Stepping in {sum(vergleichbar)} von {len(vergleichbar)} vergleichbaren Zeilen schneller.")'''),

    # ------------------------------------------------------------------ 13: Übung
    ("md", r"""### Übung Teil 4

1. **Nachbarschaft.** Baut mit `neuer_finder(RASTER_A, "r1")` und `neuer_finder(RASTER_A, "r3")` zwei neue `PathFinder` und rechnet `SOURCE` → `TARGET` mit beiden Algorithmen
   (`lauf(algo, SOURCE, TARGET, finder=...)`). Vergleicht Kosten, Länge und Zeit mit `"r2"`. Wie verändert die Nachbarschaft die Route, und wie die Zeit von Dijkstra im Vergleich zu Delta-Stepping?
2. **Zweites Raster.** Rechnet dasselbe Paar auf `RASTER_B` (`neuer_finder(RASTER_B)`) mit beiden Algorithmen. Stimmen die Kosten wieder überein? Vergleicht die Route mit Variante B aus Teil 3.
3. **Eigene Δ-Werte.** Rechnet den M→N-Fall mit `delta=30` und `delta=3000` und mit `num_threads=1` und `num_threads=2`. Bleiben die Kosten gleich? Wo liegt auf eurem Rechner das beste Δ?
   *Vorsicht:* Ein sehr großes Δ (zum Beispiel 100000) kann den Lauf deutlich verlangsamen. Probiert es mit einem einzelnen Aufruf, nicht mit `zeit`."""),
]
