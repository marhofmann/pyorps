# ruff: noqa: E501, B028  (long German prose lines in the cells)
"""Abschnitt 5: Geländemodell (DGM) und 3D-Routing."""

CELLS = [
    # ------------------------------------------------------------------ 1
    ("md", r"""---
# Teil 5: Gelände berücksichtigen (DGM, 3D-Routing)

Bisher war die Welt flach: Ein Meter Strecke kostete überall gleich viel, egal wie steil es ist. Für Kabel und Leitungen stimmt das nicht.

**Was ist ein DGM?** Ein *digitales Geländemodell* (DGM) gibt für jede Rasterzelle die Höhe des nackten Erdbodens an. Ein *digitales Oberflächenmodell* (DOM, englisch DSM) enthält zusätzlich Bäume und Gebäude. Zum Verlegen eines Kabels brauchen wir den Boden, deshalb nehmen wir ein DGM. In pyorps heißt das Argument `dem=` (englisch *digital elevation model*); gemeint ist dasselbe.

**Warum ist die Neigung wichtig?**
- Am Hang ist der Graben schwerer zu bauen, und der Boden kann abrutschen oder erodieren.
- Steile Hänge sind für Baumaschinen schwer zugänglich.
- Eine Route, die über den Berg geht, ist länger als die Karte zeigt.

**Neigung in Prozent:** Hangneigung in % = Höhenunterschied / Strecke × 100. 20 % bedeutet 2 m Anstieg auf 10 m Strecke. In Formeln rechnen wir mit der Neigung als Verhältnis (0,2 = 20 %).

**2D- gegen 3D-Länge:** Die 2D-Länge misst nur im Grundriss. Die 3D-Länge misst am Boden entlang: Pro Schritt gilt `sqrt(Strecke² + Höhenunterschied²)`.

In diesem Teil lernen wir, wie `PathFinder` ein DGM nutzt, wie ihr das Gewicht der Neigung wählt, wie wir eine harte Grenze setzen und welche Fallen es gibt.

*Daten:* Die Datei `data/dem_clip.tif` (das DGM für unser Gebiet plus 300 m Rand, 5 m Zellen, EPSG:25832) liegt dem Repository bei. Nur wenn sie fehlt, lädt die Funktion `hole_dgm()` das DGM beim LGL herunter (etwa 235 MB in 16 Kacheln, das dauert länger als eine Minute); ohne Netz nimmt sie ein künstliches Gelände.

*Datenquelle:* Geländemodell: DGM1 Baden-Württemberg, © LGL, www.lgl-bw.de, Datenlizenz Deutschland – Namensnennung – Version 2.0 (dl-de/by-2-0, www.govdata.de/dl-de/by-2-0); Datensatz: https://opengeodata.lgl-bw.de (DGM1); von 1 m auf 5 m gemittelt."""),

    # ------------------------------------------------------------------ 2
    ("code", r'''# Setup für Teil 5: Importe und Variablen aus Teil 3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from pyorps import PathFinder, Objective, GradientOptions, AlgorithmFallbackWarning, NoPathFoundError, route_metrics

def de(x, nd=0):
    # Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen
    return f"{x:,.{nd}f}".replace(",", "§").replace(".", ",").replace("§", ".")

def tab_de(df, nd=1):
    # Tabelle mit Dezimalkomma und ohne Tausendertrenner ausgeben (nur Kommazahlen, ganze Zahlen bleiben)
    return df.apply(lambda spalte: spalte.map(lambda x: f"{x:.{nd}f}".replace(".", ",")) if spalte.dtype.kind == "f" else spalte)

# (falls Teil 3 nicht ausgeführt wurde): dieselben Werte wie dort
if "RASTER_A" not in globals():
    BBOX = (427000, 5354000, 432000, 5359000)   # (min Rechtswert, min Hochwert, max Rechtswert, max Hochwert)
    CRS = "EPSG:25832"
    AUFLOESUNG = 5                               # Meter pro Zelle
    RASTER_A = "data/cost_raster_a.tiff"
    RASTER_B = "data/cost_raster_b.tiff"
    SOURCE = (427558, 5357403)                   # Kinzigtal-Talboden
    TARGET = (430288, 5355723)                   # westliches Ende des Harmersbachtals
print("Kostenraster A:", Path(RASTER_A).resolve(), "| vorhanden:", Path(RASTER_A).exists())'''),

    # ------------------------------------------------------------------ 3
    ("md", r"""## Das DGM beschaffen

Die nächste Zelle enthält die Funktion `hole_dgm()` und ein paar Hilfsfunktionen. Ihr müsst sie nur ausführen, nicht lesen.
`hole_dgm()` gibt den Pfad zu einer DGM-Datei zurück: Liegt `data/dem_clip.tif` schon da (so ist es im Repository), ist das sofort fertig. Nur wenn die Datei fehlt, lädt die Funktion das DGM1 beim LGL herunter (etwa 235 MB, 16 Kacheln, mehr als eine Minute) und mittelt es auf 5 m. Gelingt das nicht, nimmt sie ein künstliches Gelände und sagt es deutlich."""),

    ("code", r'''# Hilfsfunktionen für das DGM (Erklärung in der Zelle davor)
import io
import os
import urllib.request
import warnings
import zipfile

import numpy as np

# Quellenvermerk (Pflicht nach dl-de/by-2-0)
DEM_QUELLE = (
    "Geländemodell: DGM1 Baden-Württemberg, © LGL, www.lgl-bw.de, "
    "Datenlizenz Deutschland – Namensnennung – Version 2.0 (dl-de/by-2-0, "
    "www.govdata.de/dl-de/by-2-0); Datensatz: https://opengeodata.lgl-bw.de (DGM1); "
    "von 1 m auf 5 m gemittelt"
)
DEM_QUELLE_SYNTHETISCH = "SYNTHETISCHES Gelände (numpy-Hügel), keine echten Höhen"

BBOX = (427000, 5354000, 432000, 5359000)  # EPSG:25832: xmin, ymin, xmax, ymax (Kinzigtal bei Biberach)
RAND_M = 300
AUFLOESUNG_M = 5
NODATA = -9999.0
LGL_URL = "https://opengeodata.lgl-bw.de/data/dgm/dgm1_32_{e}_{n}_2_bw.zip"
TIMEOUT_S = 30


def _ziel_gitter(bbox, rand_m, aufloesung_m):
    xmin, ymin, xmax, ymax = bbox
    xmin, ymin, xmax, ymax = xmin - rand_m, ymin - rand_m, xmax + rand_m, ymax + rand_m
    breite = int(round((xmax - xmin) / aufloesung_m))
    hoehe = int(round((ymax - ymin) / aufloesung_m))
    return xmin, ymax, breite, hoehe


def _schreibe_tif(pfad, z, xmin, ymax, aufloesung_m, quelle):
    import rasterio
    from rasterio.transform import from_origin

    os.makedirs(os.path.dirname(os.path.abspath(pfad)), exist_ok=True)
    profil = dict(
        driver="GTiff", height=z.shape[0], width=z.shape[1], count=1, dtype="float32",
        crs="EPSG:25832", transform=from_origin(xmin, ymax, aufloesung_m, aufloesung_m),
        nodata=NODATA, compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256,
    )
    tmp = pfad + ".part"
    with rasterio.open(tmp, "w", **profil) as dst:
        dst.write(z.astype("float32"), 1)
        dst.update_tags(QUELLE=quelle)
    os.replace(tmp, pfad)


def dgm_quelle(pfad):
    """Quellenvermerk einer mit hole_dgm() geschriebenen Datei (Tag QUELLE)."""
    import rasterio

    with rasterio.open(pfad) as src:
        return src.tags().get("QUELLE", "unbekannt")


def _lade_zip(url, versuche):
    for i in range(versuche):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
                return r.read()
        except Exception:  # Netzfehler: einmal erneut versuchen, dann aufgeben
            if i == versuche - 1:
                raise


def _dgm1_lgl(bbox, rand_m, aufloesung_m, url=LGL_URL):
    """DGM1-Kacheln laden und auf das Zielgitter mitteln (Blockmittel, aufloesung_m ganzzahlig)."""
    xmin, ymax, breite, hoehe = _ziel_gitter(bbox, rand_m, aufloesung_m)
    a = int(aufloesung_m)
    # 1-m-Mosaik des Zielfensters; NaN = keine Messung
    mosaik = np.full((hoehe * a, breite * a), np.nan, dtype="float32")
    x0, y1 = xmin, ymax
    x1, y0 = xmin + breite * a, ymax - hoehe * a
    # 1-km-Teilkacheln, die das Fenster berühren
    km_e = range(int(np.floor(x0 / 1000)), int(np.ceil(x1 / 1000)))
    km_n = range(int(np.floor(y0 / 1000)), int(np.ceil(y1 / 1000)))
    # 2-km-ZIPs: Ostwert ungerade, Nordwert gerade (so liefert das LGL-Portal die Kacheln)
    zips = sorted({(e if e % 2 else e - 1, n if n % 2 == 0 else n - 1) for e in km_e for n in km_n})
    for k, (e2, n2) in enumerate(zips):
        # erste Kachel ohne zweiten Versuch: ohne Netz scheitert die Zelle schnell
        z = zipfile.ZipFile(io.BytesIO(_lade_zip(url.format(e=e2, n=n2), 1 if k == 0 else 2)))
        for name in z.namelist():
            if not name.endswith(".xyz"):
                continue
            teile = os.path.basename(name).split("_")  # dgm1_32_<e>_<n>_1_bw_<jahr>.xyz
            e, n = int(teile[2]), int(teile[3])
            if e not in km_e or n not in km_n:
                continue
            xyz = np.loadtxt(io.BytesIO(z.read(name)), dtype="float64")
            spalte = np.floor(xyz[:, 0] - x0).astype(np.int64)
            zeile = np.floor(y1 - xyz[:, 1]).astype(np.int64)
            ok = (spalte >= 0) & (spalte < mosaik.shape[1]) & (zeile >= 0) & (zeile < mosaik.shape[0])
            mosaik[zeile[ok], spalte[ok]] = xyz[ok, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # leere Blöcke -> NaN
        grob = np.nanmean(mosaik.reshape(hoehe, a, breite, a), axis=(1, 3))
    return np.where(np.isfinite(grob), grob, NODATA), xmin, ymax


def synthetisches_gelaende(bbox=BBOX, rand_m=RAND_M, aufloesung_m=AUFLOESUNG_M, seed=7):
    """Glattes Tal mit Hängen und Kuppen auf dem Zielgitter, Höhen ca. 180 bis 550 m.

    Ein ebener Talboden (ca. 600 m breit) läuft von Nord nach Süd durch die Mitte,
    die Hänge steigen nach beiden Seiten an, Gauß-Kuppen machen sie unregelmäßig.
    """
    xmin, ymax, breite, hoehe = _ziel_gitter(bbox, rand_m, aufloesung_m)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:hoehe, 0:breite].astype("float64") * aufloesung_m  # Meter ab Nordwestecke
    achse = 0.5 * breite * aufloesung_m + 300.0 * np.sin(yy / 1500.0)      # geschwungene Talachse
    abstand = np.maximum(np.abs(xx - achse) - 300.0, 0.0)                 # 0 auf dem Talboden
    z = 180.0 + 5.0 * yy / 1000.0 + 280.0 * (1.0 - np.exp(-(abstand / 900.0) ** 2))
    for _ in range(15):
        cx = rng.uniform(0, breite * aufloesung_m)
        cy = rng.uniform(0, hoehe * aufloesung_m)
        r = rng.uniform(250.0, 700.0)
        kuppe = rng.uniform(20.0, 90.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        z += kuppe * (1.0 - np.exp(-(abstand / 300.0) ** 2))  # Talboden bleibt eben
    return z, xmin, ymax


def hole_dgm(pfad="data/dem_clip.tif", bbox=BBOX, rand_m=RAND_M, aufloesung_m=AUFLOESUNG_M,
             download=True, synthetisch_pfad="data/dem_synthetisch.tif", url=LGL_URL):
    """Pfad zu einem DGM-GeoTIFF (EPSG:25832, float32, nodata=-9999) für bbox plus Rand."""
    if os.path.exists(pfad):
        return pfad
    if download:
        try:
            z, xmin, ymax = _dgm1_lgl(bbox, rand_m, aufloesung_m, url)
            _schreibe_tif(pfad, z, xmin, ymax, aufloesung_m, DEM_QUELLE)
            return pfad
        except Exception as fehler:
            warnings.warn(f"DGM-Download fehlgeschlagen ({fehler!r}); nehme synthetisches Gelände.")
    if not os.path.exists(synthetisch_pfad):
        z, xmin, ymax = synthetisches_gelaende(bbox, rand_m, aufloesung_m)
        _schreibe_tif(synthetisch_pfad, z, xmin, ymax, aufloesung_m, DEM_QUELLE_SYNTHETISCH)
    print("ACHTUNG: synthetisches Gelände, keine echten Höhen:", synthetisch_pfad)
    return synthetisch_pfad

# Datei holen: erst der Cache, sonst Download, sonst synthetisches Gelände
DEM = hole_dgm()
print("DGM-Datei:", Path(DEM).resolve())
print(DEM_QUELLE)'''),

    # ------------------------------------------------------------------ 4
    ("md", r"""## Das Gelände ansehen

Wir lesen das DGM ein und rechnen die Neigung selbst aus (`numpy.gradient`, in Prozent). Das Schummerungsbild (*hillshade*) beleuchtet das Gelände schräg von Nordwesten. Es zeigt Hänge besser als Farben allein. Der Kreis ist `SOURCE`, das Quadrat `TARGET`, der cyanfarbene Rahmen das Untersuchungsgebiet.
Die Zahlen unten gelten für das Untersuchungsgebiet `BBOX`; das DGM reicht rundherum 300 m darüber hinaus.

Danach vergleichen wir die Neigung mit dem Kostenraster A: Wo es steil ist, liegt oft Wald."""),

    # ------------------------------------------------------------------ 5
    ("code", r'''# DGM einlesen, Neigung in Prozent, Schummerung
with rasterio.open(DEM) as src:
    dem = src.read(1).astype("float64")
    T = src.transform                      # Transform des DGM (5 m, EPSG:25832)
    dem[dem == src.nodata] = np.nan
    quelle = src.tags().get("QUELLE", "unbekannt")
ZELLE = abs(T.a)
dy, dx = np.gradient(dem, ZELLE)
slope = np.hypot(dx, dy) * 100             # Neigung in Prozent
EXT = (T.c, T.c + T.a * dem.shape[1], T.f + T.e * dem.shape[0], T.f)   # links, rechts, unten, oben
hs = LightSource(azdeg=315, altdeg=45).hillshade(np.nan_to_num(dem, nan=np.nanmean(dem)),
                                                 vert_exag=1, dx=ZELLE, dy=ZELLE)

def zelle_von_xy(x, y):
    # Spalte und Zeile (ganzzahlig) im DGM-Raster
    return int((x - T.c) / T.a), int((y - T.f) / T.e)

c0, r1 = zelle_von_xy(BBOX[0], BBOX[1])
c1, r0 = zelle_von_xy(BBOX[2], BBOX[3])
im_gebiet = np.s_[r0:r1, c0:c1]            # Ausschnitt von BBOX im DGM

def achsen(ax):
    # Achsen in Metern, wenige Beschriftungen
    ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Rechtswert [m]", fontsize=8); ax.set_ylabel("Hochwert [m]", fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
axes[0].imshow(hs, cmap="gray", extent=EXT)
bild = axes[0].imshow(dem, cmap="terrain", alpha=0.45, extent=EXT)
fig.colorbar(bild, ax=axes[0], shrink=0.75, label="Höhe [m]")
axes[0].set_title("Gelände (Höhe und Schummerung)", fontsize=10)
bild = axes[1].imshow(slope, cmap="magma", vmin=0, vmax=60, extent=EXT)
fig.colorbar(bild, ax=axes[1], shrink=0.75, label="Hangneigung [%] (Skala bis 60)")
axes[1].set_title("Neigung", fontsize=10)
for ax in axes:
    ax.add_patch(Rectangle(BBOX[:2], BBOX[2] - BBOX[0], BBOX[3] - BBOX[1], fill=False, ec="cyan", lw=1))
    ax.plot(*SOURCE, "o", mfc="white", mec="black", ms=6, label="SOURCE")
    ax.plot(*TARGET, "s", mfc="white", mec="black", ms=6, label="TARGET")
    achsen(ax)
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
plt.show()

hoehe = dem[im_gebiet]; neigung = slope[im_gebiet]
print("Quelle der Höhen:", quelle)
print(f"Höhe im Untersuchungsgebiet: {de(np.nanmin(hoehe), 1)} bis {de(np.nanmax(hoehe), 1)} m")
print(f"Neigung in %: Median (50 %) = {de(np.nanpercentile(neigung, 50), 1)}, 90 %-Quantil = {de(np.nanpercentile(neigung, 90), 1)}, 99 %-Quantil = {de(np.nanpercentile(neigung, 99), 1)}")
print(f"Anteil der Zellen steiler als 15 %: {de(100 * np.mean(neigung > 15), 1)} %, flacher als 3 %: {de(100 * np.mean(neigung < 3), 1)} %")'''),

    # ------------------------------------------------------------------ 6
    ("code", r'''# Neigung gegen Kostenraster A: Anteil der Kostenklassen je Neigungsklasse
with rasterio.open(RASTER_A) as src:
    kosten = src.read(1)
    Tk = src.transform
# Kostenraster liegt im DGM-Raster (gleiche Zellgröße, gleiche Ausrichtung); wir schneiden aus der Neigung aus
ck, rk = zelle_von_xy(Tk.c, Tk.f)
assert abs(Tk.a) == ZELLE and ck >= 0 and rk >= 0
neigung_k = slope[rk:rk + kosten.shape[0], ck:ck + kosten.shape[1]]
assert neigung_k.shape == kosten.shape

neigungsklasse = pd.cut(neigung_k.ravel(), [0, 10, 20, 40, np.inf], right=False,
                        labels=["0-10 %", "10-20 %", "20-40 %", "über 40 %"])
kostenklasse = pd.cut(kosten.ravel().astype("float64"), [0, 150, 300, 65535, 70000], right=False,
                      labels=["günstig (< 150)", "mittel (150-299)", "teuer (300 und mehr, vor allem Wald)", "gesperrt (65535)"])
tab = (pd.crosstab(neigungsklasse, kostenklasse, normalize="index") * 100).rename_axis(index=None, columns=None)
tab.insert(0, "Zellen", pd.Series(neigungsklasse).value_counts().sort_index())
print("Anteil der Kostenklassen [%] je Neigungsklasse (Zeilen ergeben 100 %):")
print(tab_de(tab).to_string())

ax = tab.drop(columns="Zellen").plot.barh(stacked=True, figsize=(7, 2.8), colormap="viridis")
ax.set_xlabel("Anteil der Zellen [%]", fontsize=8); ax.set_ylabel("Hangneigung", fontsize=8); ax.tick_params(labelsize=8); ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()'''),

    # ------------------------------------------------------------------ 7
    ("md", r"""## Eine 3D-Route berechnen

Ein DGM wird mit `dem=` an `PathFinder` gegeben. Wie stark die Neigung zählt, legt ein `Objective` fest:

```python
Objective(weights={"cost": 1.0, "gradient": 5.0})
```

Die Suche minimiert dann `Kosten + Gewicht × Neigung`, gerechnet über jedes Stück der Route. Schon ohne Gewicht gilt nach dem pyorps-Quelltext: Mit DGM ist jedes Stück am Hang länger (3D-Streckung `sqrt(1 + Neigung²)`, Neigung als Verhältnis, 0,2 = 20 %; nicht einstellbar).

Wir testen auf einem **Höhenrücken-Paar**: Start auf dem Talboden der Kinzig, Ziel im Harmersbachtal, dazwischen ein bewaldeter Höhenrücken. Die Suche bekommt 1500 m Puffer um die Luftlinie.
Die Tabelle misst jede Route gleich: 2D-Länge, 3D-Länge, steilste Kante, höchster Punkt und Gesamtanstieg rechnen wir selbst aus den Höhen entlang der Route. `Kosten auf Raster A [EUR]` bewertet jede Route mit Raster A (Zellwert × 2D-Länge, `route_metrics` wie in Teil 3). Diese Spalte ist für 2D- und 3D-Routen dieselbe Messung. Das `total_cost` einer 3D-Route lassen wir weg: pyorps nennt es Einheiten der Zielfunktion, nicht EUR, und garantiert nicht, dass es mit dem 2D-Wert vergleichbar ist."""),

    # ------------------------------------------------------------------ 8
    ("code", r'''RIDGE = ((428200, 5356500), (431200, 5356500))   # Talboden Kinzig -> Höhenrücken -> Harmersbachtal
ST_2D = ("tab:blue", 4.0, "-")                    # Linienstil: Farbe, Breite, Strich
ST_STRICH = (0, (4, 2))
PUFFER = 1500                                     # Suchfenster: Puffer um die Luftlinie in Metern
OBJ5 = Objective(weights={"cost": 1.0, "gradient": 5.0})
NUR_ERSTE_MELDUNG = "Path.total_cost is a recompute"

def route(paar, zeige_meldung=False, **kw):
    # eine Route mit Dijkstra; die bekannte total_cost-Meldung nur auf Wunsch zeigen
    with warnings.catch_warnings(record=True) as gesammelt:
        warnings.simplefilter("always")
        finder = PathFinder(RASTER_A, paar[0], paar[1], search_space_buffer_m=PUFFER,
                            neighborhood_str="r2", **kw)
        pfad = finder.find_route()
    for w in gesammelt:
        if NUR_ERSTE_MELDUNG in str(w.message):
            if zeige_meldung:
                print("pyorps sagt hier auf Englisch: total_cost ist eine Nachrechnung aus Zellwert × 2D-Länge. Sie berücksichtigt weder die 3D-Streckung noch die Neigung,")
                print("und die Zahl ist in Einheiten der Zielfunktion, nicht in EUR. Der Originaltext:")
                print(w.message)
        else:
            warnings.warn(w.message, w.category)
    return pfad

def hoehe_an(xy):
    # Höhe des DGM an Punkten (xy: n x 2), bilinear
    spalte = (xy[:, 0] - T.c) / T.a - 0.5
    zeile = (xy[:, 1] - T.f) / T.e - 0.5
    return map_coordinates(np.nan_to_num(dem, nan=np.nanmean(dem)), [zeile, spalte], order=1, mode="nearest")

def profil(pfad):
    # Koordinaten, zurückgelegte 2D-Strecke und Geländehöhe entlang der Route
    xy = np.asarray(pfad.path_coords)
    schritt = np.hypot(*np.diff(xy, axis=0).T)
    return xy, np.r_[0, np.cumsum(schritt)], hoehe_an(xy)

def kosten_auf_a(pfad):
    # Route auf Raster A bewerten: Zellen der Route im ganzen Raster, dann route_metrics (Zellwert x 2D-Länge)
    xy = np.asarray(pfad.path_coords)
    spalte = np.floor((xy[:, 0] - Tk.c) / Tk.a).astype(np.int64)
    zeile = np.floor((xy[:, 1] - Tk.f) / Tk.e).astype(np.int64)
    zellen = zeile * kosten.shape[1] + spalte
    zellen = zellen[np.r_[True, np.diff(zellen) != 0]]
    return route_metrics({0: zellen}, kosten, ZELLE)[0]["construction_cost"]

def kennzahlen(pfad):
    # gleiche Messung für 2D- und 3D-Routen (Höhen aus dem DGM entlang der Route)
    xy, lauf, h = profil(pfad)
    schritt, dh = np.diff(lauf), np.diff(h)
    return {"2D-Länge [m]": lauf[-1],
            "3D-Länge [m]": np.hypot(schritt, dh).sum(),
            "Kosten auf Raster A [EUR]": kosten_auf_a(pfad),
            "steilste Kante [%]": (np.abs(dh) / np.maximum(schritt, 1e-9) * 100).max(),
            "höchster Punkt [m]": h.max(),
            "Anstieg gesamt [m]": np.clip(dh, 0, None).sum()}

def tabelle(routen):
    return tab_de(pd.DataFrame({name: kennzahlen(p) for name, p in routen.items()}).T)

def karte(ax, routen, stile, rand=250, titel=""):
    # Schummerung im Fenster um alle Routen, Routen darüber
    alle = np.vstack([np.asarray(p.path_coords) for p in routen.values()])
    x0, y0 = alle.min(axis=0) - rand
    x1, y1 = alle.max(axis=0) + rand
    c0, r1 = zelle_von_xy(max(x0, EXT[0]), max(y0, EXT[2]))
    c1, r0 = zelle_von_xy(min(x1, EXT[1]), min(y1, EXT[3]))
    ext = (T.c + c0 * ZELLE, T.c + c1 * ZELLE, T.f - r1 * ZELLE, T.f - r0 * ZELLE)
    ax.imshow(hs[r0:r1, c0:c1], cmap="gray", extent=ext)
    ax.imshow(dem[r0:r1, c0:c1], cmap="terrain", alpha=0.35, extent=ext)
    for (name, p), (farbe, breite, strich) in zip(routen.items(), stile):
        xy = np.asarray(p.path_coords)
        ax.plot(xy[:, 0], xy[:, 1], color=farbe, lw=breite, ls=strich, label=name)
    ax.set_title(titel, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    achsen(ax)

pfad_2d = route(RIDGE)                                                    # ohne DGM
pfad_3d = route(RIDGE, zeige_meldung=True, dem=DEM, objective=OBJ5)       # mit DGM, Gewicht 5
routen = {"2D (ohne DGM)": pfad_2d, "3D (Gewicht 5)": pfad_3d}

print(f"\npyorps meldet für die 3D-Route: total_length = {de(pfad_3d.total_length, 1)} m (3D), "
      f"total_length_2d = {de(pfad_3d.total_length_2d, 1)} m | ohne DGM ist total_length_3d = {pfad_2d.total_length_3d}")
assert np.isclose(kennzahlen(pfad_3d)["3D-Länge [m]"], pfad_3d.total_length_3d, rtol=1e-3)   # eigene Messung stimmt
eur_2d = kennzahlen(pfad_2d)["Kosten auf Raster A [EUR]"]
assert np.isclose(eur_2d, pfad_2d.total_cost, rtol=1e-3)
print(f"Prüfung: route_metrics bewertet die 2D-Route mit {de(eur_2d)} EUR, PathFinder meldet {de(pfad_2d.total_cost)} EUR.")
print(f"Zum Vergleich bei der 3D-Route: total_cost = {de(pfad_3d.total_cost)}, auf Raster A bewertet {de(kennzahlen(pfad_3d)['Kosten auf Raster A [EUR]'])} EUR.\n")
print(tabelle(routen).to_string())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [2.2, 1]})
STILE_PAAR = [ST_2D, ("tab:red", 2.5, "-")]
karte(ax1, routen, STILE_PAAR, titel="Höhenrücken-Paar: Route ohne und mit DGM")
for p in RIDGE:
    ax1.plot(*p, "o", mfc="white", mec="black", ms=6)
for (name, p), (farbe, breite, strich) in zip(routen.items(), STILE_PAAR):
    _, lauf, h = profil(p)
    ax2.plot(lauf, h, color=farbe, lw=breite * 0.7, ls=strich, label=name)
ax2.set_xlabel("Strecke entlang der Route (2D) [m]", fontsize=8); ax2.set_ylabel("Höhe [m]", fontsize=8)
ax2.tick_params(labelsize=8)
ax2.set_title("Höhenprofil", fontsize=10); ax2.legend(fontsize=8)
fig.tight_layout()
plt.show()'''),

    # ------------------------------------------------------------------ 9
    ("md", r"""## Das Gewicht wählen

Das Gewicht legt fest, wie teuer ein Prozent Neigung ist. Merkregel: **Gewicht × Neigung in % sollte in der Größenordnung der Kosten pro Meter liegen**, sonst sieht die Suche die Neigung nicht (Gewicht zu klein) oder nur noch die Neigung (Gewicht zu groß).
Die Zelle rechnet diese Zahlen für Raster A aus und lässt dann die Gewichte 0, 2, 5 und 20 auf dem Höhenrücken-Paar laufen. In der Karte liegen Routen teils übereinander; deshalb sind einige Linien dick und andere gestrichelt darüber gezeichnet.

Danach machen wir dasselbe für das Hauptpaar `SOURCE` -> `TARGET` aus Teil 3. Ob sich die Route ändert, entscheiden die Zahlen, nicht unsere Erwartung."""),

    # ------------------------------------------------------------------ 10
    ("code", r'''# Kosten pro Meter der 2D-Route gegen Gewicht × Neigung
GEWICHTE = (0, 2, 5, 20)
STILE_GEWICHT = [ST_2D, ("black", 1.6, ST_STRICH), ("tab:orange", 3.5, "-"), ("tab:red", 1.8, ST_STRICH), ("magenta", 2.4, "-")]

kosten_je_m = pfad_2d.total_cost / pfad_2d.total_length
print(f"Mittlere Kosten der 2D-Route: {de(kosten_je_m)} EUR pro Meter")
for w in GEWICHTE:
    print(f"Gewicht {w:>2}: bei 20 % Neigung kommen {de(w * 20):>3} pro Meter dazu ({de(100 * w * 20 / kosten_je_m)} % der Kosten)")

def gewichte_testen(paar, gewichte=GEWICHTE):
    # 2D-Route und je eine 3D-Route pro Gewicht
    routen = {"2D (ohne DGM)": route(paar)}
    for w in gewichte:
        routen[f"3D, Gewicht {w}"] = route(paar, dem=DEM,
                                           objective=Objective(weights={"cost": 1.0, "gradient": float(w)}))
    return routen

def abstand_m(pfad_a, pfad_b):
    # größter Abstand eines Punktes von Route b zur Route a (in Metern)
    punkte_a = np.asarray(pfad_a.path_coords)
    return cKDTree(punkte_a).query(np.asarray(pfad_b.path_coords))[0].max()

routen_ridge = gewichte_testen(RIDGE)
print()
print(tabelle(routen_ridge).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
karte(ax, routen_ridge, STILE_GEWICHT, titel="Höhenrücken-Paar: Gewicht der Neigung 0, 2, 5, 20")
for p in RIDGE:
    ax.plot(*p, "o", mfc="white", mec="black", ms=6)
fig.tight_layout()
plt.show()'''),

    # ------------------------------------------------------------------ 11
    ("code", r'''# Hauptpaar aus Teil 3: gleiche Auswertung
routen_main = gewichte_testen((SOURCE, TARGET))
print(tabelle(routen_main).to_string())

namen = list(routen_main)
for name in namen[1:]:
    print(f"größter Abstand 2D-Route -> {name}: {de(abstand_m(routen_main[namen[0]], routen_main[name]))} m")
unterschied = max(abstand_m(routen_main[namen[0]], routen_main[n]) for n in namen[1:])
if unterschied > 100:
    print("\nErgebnis: Das Gelände ändert die Route für dieses Paar.")
else:
    print("\nErgebnis: Die Routen sind praktisch gleich. Das ist ein gültiges Ergebnis: das Gelände ändert nicht immer die Route.")
if abstand_m(routen_main[namen[0]], routen_main[namen[1]]) > 100:
    print("Schon bei Gewicht 0 ändert sich die Route: Die 3D-Streckung allein macht den steilen Hang teurer.")
w_aus = max(namen[1:], key=lambda n: abstand_m(routen_main[namen[1]], routen_main[n]))
print(f"Unterschied zwischen den 3D-Routen (Gewicht 0 gegen {w_aus}): {de(abstand_m(routen_main[namen[1]], routen_main[w_aus]))} m")

fig, ax = plt.subplots(figsize=(9, 5))
karte(ax, routen_main, STILE_GEWICHT, titel="Hauptpaar SOURCE -> TARGET: ohne und mit DGM")
ax.plot(*SOURCE, "o", mfc="white", mec="black", ms=6); ax.plot(*TARGET, "s", mfc="white", mec="black", ms=6)
fig.tight_layout()
plt.show()'''),

    # ------------------------------------------------------------------ 12
    ("md", r"""## Harte Grenze und Delta-Stepping

**Harte Neigungsgrenze.** `GradientOptions(max_gradient_pct=...)` verbietet jeden Schritt, der steiler ist als der Grenzwert. Das ist kein Preis, sondern ein Verbot. Wird die Grenze zu klein, gibt es keine Route mehr, und pyorps wirft `NoPathFoundError`. Diesen Fehler fangen wir in der nächsten Zelle ab.
Gemessen wird die Neigung zwischen zwei benachbarten Zellen auf der Route (Kante), nicht die Neigung der Rasterzelle aus der Karte oben. Darum kann die Karte an einer Stelle mehr zeigen als der Grenzwert.

**Delta-Stepping mit DGM.** Delta-Stepping kann Neigungsterme nicht auswerten. Geben wir ein DGM, läuft deshalb Dijkstra, und pyorps sagt das mit einer `AlgorithmFallbackWarning`. Das ist kein Fehler, nur ein Hinweis, dass der gewünschte Algorithmus nicht läuft. Wir fangen die Meldung absichtlich ab, um sie zu lesen."""),

    # ------------------------------------------------------------------ 13
    ("code", r'''# Harte Neigungsgrenze auf dem Höhenrücken-Paar (nur Kosten, dazu das Limit)
def route_mit_grenze(paar, grenze):
    return route(paar, dem=DEM,
                 objective=Objective(weights={"cost": 1.0},
                                     gradient_options=GradientOptions(max_gradient_pct=float(grenze))))

routen_grenze = {"2D (ohne DGM)": pfad_2d, "3D, ohne Grenze": route(RIDGE, dem=DEM, objective=Objective(weights={"cost": 1.0}))}
for grenze in (40, 30, 20, 10):
    routen_grenze[f"Grenze {grenze} %"] = route_mit_grenze(RIDGE, grenze)
tab = tabelle(routen_grenze)
print(tab.to_string())

# eine zu kleine Grenze: keine Route
try:
    route_mit_grenze(RIDGE, 5)
except NoPathFoundError as fehler:
    print("\nGrenze 5 %: Es gibt keine Route mehr. pyorps meldet das als NoPathFoundError, der Text ist englisch")
    print("(die beiden Zahlen darin sind interne Nummern, keine Koordinaten):")
    print(str(fehler)[:150], "...")
else:
    print("\nGrenze 5 %: unerwartet doch eine Route")

fig, ax = plt.subplots(figsize=(9, 5))
STILE_GRENZE = [ST_2D, ("black", 1.6, ST_STRICH), ("tab:orange", 3.5, "-"), ("tab:red", 1.8, ST_STRICH), ("magenta", 2.4, "-"), ("darkgreen", 2.4, "-")]
karte(ax, routen_grenze, STILE_GRENZE, titel="Höhenrücken-Paar: harte Neigungsgrenze")
for p in RIDGE:
    ax.plot(*p, "o", mfc="white", mec="black", ms=6)
fig.tight_layout()
plt.show()'''),

    # ------------------------------------------------------------------ 14
    ("code", r'''# Delta-Stepping mit DGM: pyorps fällt auf Dijkstra zurück und warnt
with warnings.catch_warnings(record=True) as gesammelt:
    warnings.simplefilter("always")
    finder = PathFinder(RASTER_A, RIDGE[0], RIDGE[1], search_space_buffer_m=PUFFER,
                        neighborhood_str="r2", dem=DEM, objective=OBJ5)
    pfad_ds = finder.find_route(algorithm="delta-stepping")

print("pyorps warnt auf Englisch: Delta-Stepping kann Neigungsterme nicht auswerten, deshalb läuft Dijkstra. Der Originaltext:")
for w in gesammelt:
    if issubclass(w.category, AlgorithmFallbackWarning):
        print(w.category.__name__, "->", w.message)
print("path.algorithm mit DGM:", pfad_ds.algorithm, "(gewünscht war delta-stepping)")
print("gleiche Länge wie die Dijkstra-Route von oben:", np.isclose(pfad_ds.total_length, pfad_3d.total_length))

# zum Vergleich ohne DGM: Delta-Stepping läuft wirklich
pfad_ds2 = PathFinder(RASTER_A, RIDGE[0], RIDGE[1], search_space_buffer_m=PUFFER,
                      neighborhood_str="r2").find_route(algorithm="delta-stepping")
print("path.algorithm ohne DGM:", pfad_ds2.algorithm)'''),

    # ------------------------------------------------------------------ 15
    ("code", r'''# 3D-Bild: Gelände im Fenster der Routen, Routen als Linien auf Geländehöhe
STUFE = 4                                   # jedes 4. Rasterelement (20 m) für ein schnelles Bild
alle = np.vstack([np.asarray(p.path_coords) for p in (pfad_2d, pfad_3d)])
x0, y0 = alle.min(axis=0) - 250
x1, y1 = alle.max(axis=0) + 250
c0, r1 = zelle_von_xy(max(x0, EXT[0]), max(y0, EXT[2]))
c1, r0 = zelle_von_xy(min(x1, EXT[1]), min(y1, EXT[3]))
Z = dem[r0:r1:STUFE, c0:c1:STUFE]
xs = T.c + (np.arange(c0, c1, STUFE) + 0.5) * ZELLE - x0          # Meter ab der Fensterecke
ys = T.f + (np.arange(r0, r1, STUFE) + 0.5) * T.e - y0
X, Y = np.meshgrid(xs, ys)
farben = LightSource(315, 45).shade(Z, plt.cm.terrain, vert_exag=1, blend_mode="soft", dx=STUFE * ZELLE, dy=STUFE * ZELLE)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(projection="3d", computed_zorder=False)       # feste Zeichenreihenfolge: erst Fläche, dann Linien
ax.plot_surface(X, Y, Z, facecolors=farben, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, zorder=1)
for p, farbe, name in ((pfad_2d, "tab:blue", "2D-Route"), (pfad_3d, "tab:red", "3D-Route (Gewicht 5)")):
    xy, _, h = profil(p)
    ax.plot(xy[:, 0] - x0, xy[:, 1] - y0, h + 8, color=farbe, lw=2.2, label=name, zorder=10)   # 8 m über dem Boden gezeichnet
ax.set_box_aspect((x1 - x0, y1 - y0, 2.0 * (np.nanmax(Z) - np.nanmin(Z))), zoom=1.15)   # Höhe 2-fach überhöht
ax.view_init(elev=35, azim=-80)
ax.set_xlabel("Ost [m ab Fensterecke]", fontsize=8, labelpad=6)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8, loc="upper left")
# Die Namen der y- und z-Achse stehen im Titel: Achsenbeschriftungen von 3D-Bildern werden im Notebook oft abgeschnitten
ax.set_title("Gelände (2-fach überhöht) mit 2D- und 3D-Route\n"
             "x: Ost, y: Nord (Meter ab Fensterecke), z: Höhe (m ü. NHN)", fontsize=9)
fig.subplots_adjust(left=0.02, right=0.86, bottom=0.04, top=0.94)
plt.show()'''),

    # ------------------------------------------------------------------ 16
    ("md", r"""## Was wir mitnehmen

**Länge und Kosten mit DGM**
- `total_length` ist mit DGM die **3D-Länge**. Dazu gibt es `total_length_2d`, `total_length_3d`, `mean_gradient_pct` und `max_gradient_pct`. Ohne DGM sind diese vier `None` und `total_length` ist die 2D-Länge. `path_geometry.length` ist immer die 2D-Länge der Linie.
- `total_cost` bleibt auch mit DGM die **2D-Nachrechnung** auf der Suchfläche (Zellwert × 2D-Länge) und ist bei 3D-Routen laut pyorps in Einheiten der Zielfunktion angegeben, **nicht in EUR garantiert** (siehe die Meldung oben; die Zahlen im Vergleich stehen in der Zelle oben). Das hat die Suche auch nicht minimiert: Gesucht wurde `feasibility` (Kosten plus Gewicht × Neigung über 3D-Längen); die Teile stehen in `Path.metrics`. pyorps warnt davor einmal pro `PathFinder` mit einer `UserWarning` und schreibt es in `Path.total_cost_basis`. Um Routen fair zu vergleichen, bewerten wir sie wie in diesem Teil mit `route_metrics` auf einem Raster.
- Die 3D-Streckung `sqrt(1 + Neigung²)` (Neigung als Verhältnis) gilt nach dem Quelltext immer, sobald ein DGM da ist.

**Gewicht und Grenze**
- Gewicht × Neigung in % soll zu den Kosten pro Meter passen (Zelle oben). Das Gewicht macht Neigung teurer, die Grenze verbietet Schritte. Eine Grenze kann jede Route unmöglich machen (`NoPathFoundError`).
- Ohne `objective=` rechnet pyorps nach dem Quelltext mit einer eingebauten exponentiellen Neigungsstrafe (hier nicht ausprobiert). Wir nehmen immer ein eigenes `Objective`, dann ist klar, was gerechnet wird.

**Abdeckung des DGM** (aus der pyorps-Dokumentation und dem Quelltext von Version 0.5.0; in diesem Notebook rechnen wir das nicht nach)
- Zellen, die das DGM nicht abdeckt (außerhalb, Nodata, Löcher), sind **gesperrt**, ohne Warnung. Nur wenn ein Start- oder Zielpunkt dort liegt und verschoben wird, meldet pyorps eine `CoverageWarning`.
- Das DGM muss im gleichen Koordinatensystem sein wie das Kostenraster (hier EPSG:25832). pyorps rechnet nicht um und bricht mit `InvalidInputError` ab. Ein DGM ohne Koordinatensystem gilt als im System des Rasters.
- Das DGM wird **bilinear** auf das Suchfenster des Kostenrasters umgerechnet. Am Fensterrand kann ein schmaler Streifen ohne Höhe bleiben; ein großzügiger Puffer hilft.
- Als Datei oder als Tupel `(feld, crs, transform)`. Ein nacktes Array geht bei einem Kostenraster aus einer Datei nicht.

**Auflösung.** Ein gröberes DGM (z. B. 25 m statt 5 m) ist schneller zu laden; schmale steile Teile werden beim Mitteln flacher. Ob die Route sich ändert, probiert ihr in Aufgabe 3 aus.

**Delta-Stepping** läuft mit DGM nicht (`AlgorithmFallbackWarning`, `path.algorithm == "dijkstra"`).

**GUI.** Die Karte „Objective & terrain" in der GUI (Teil 7) stellt dasselbe ein: DGM-Datei, Gewichte, Neigungsgrenze. Die GUI verweigert die Kombination DGM + Delta-Stepping und zeigt eine Meldung am Feld; die Bibliothek fällt dagegen mit Warnung auf Dijkstra zurück."""),

    # ------------------------------------------------------------------ 17
    ("md", r"""### Übung Teil 5

Arbeitet mit neuen Variablennamen; `RASTER_A`, `DEM`, `SOURCE` und `TARGET` bleiben unverändert.

1. **Zweites Paar.** Rechnet mit `gewichte_testen(((428663, 5356873), (431273, 5356048)))` und `tabelle(...)` das Paar `SOURCES[2]` -> `TARGETS[3]` aus Teil 3. Ändern sich Länge und steilste Kante mit dem Gewicht? Zeichnet die Routen mit `karte(ax, routen, STILE_GEWICHT)`.
2. **Kleinste Grenze.** Sucht für das Hauptpaar `SOURCE` -> `TARGET` mit `route_mit_grenze` die kleinste ganzzahlige Neigungsgrenze, bei der noch eine Route existiert (Tipp: fangt `NoPathFoundError` ab). Wie ändert sich die Länge der Route an dieser Grenze?
3. **Gröberes DGM.** Mittelt das DGM auf 25 m: `dem25 = dem.reshape(224, 5, 224, 5).mean(axis=(1, 3)).astype("float32")`, dazu `from affine import Affine` und `transform25 = Affine(25, 0, T.c, 0, -25, T.f)`. Übergebt `dem=(dem25, "EPSG:25832", transform25)` an `PathFinder` mit dem Gewicht 5 auf dem Höhenrücken-Paar. Vergleicht `total_length` und `max_gradient_pct` mit der 5-m-Route `pfad_3d` aus der Zelle oben."""),
]
