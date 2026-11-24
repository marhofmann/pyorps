# ruff: noqa: E501, B028  (long German prose lines in the cells)
"""Abschnitt 6: Freileitungen mit ConstrainedPathFinder (Route und Masten gemeinsam)."""

CELLS = [
    # ------------------------------------------------------------------ 1
    ("md", r"""---
# Teil 6: Freileitungen mit `ConstrainedPathFinder`

Ein Kabel liegt im Boden und folgt jeder Zelle seiner Route. Eine **Freileitung** hängt in der Luft: Der Leiter läuft gerade von Mast zu Mast. Daraus folgen neue Regeln:

- **Masten:** Sie kosten viel (Fundament und Stahl) und stehen nur auf erlaubten Flächen.
- **Spannfeld:** Das ist das Stück Leitung zwischen zwei Masten. Seine Länge heißt **Spannweite**; sie liegt zwischen einem kleinsten und einem größten Wert.
- **Winkel:** Die Leitung knickt nur an einem Mast. Je schärfer der Knick, desto schwerer und teurer der Mast.
- **Durchhang und Bodenabstand:** Der Leiter hängt zwischen den Masten durch. An keiner Stelle darf er dem Boden (oder Bäumen) näher kommen als ein Mindestabstand.

**Was `ConstrainedPathFinder` macht:** Er sucht die Route und die Maststandorte in *einer* Suche. Ein Zustand der Suche ist nicht nur die Zelle, sondern das Tripel (Zelle, Richtung, Länge seit dem letzten Mast). So kann die Suche Winkel- und Spannweitengrenzen schon während der Suche einhalten, statt erst eine Route zu suchen und dann Masten daraufzusetzen.

**Masttypen** (die Namen kommen aus dem Profil und stehen so im Ergebnis):

| im Ergebnis | auf Deutsch | wann |
|---|---|---|
| `terminal` | Endmast | am Start- und am Zielpunkt |
| `suspension` | Tragmast | Knick bis zur ersten Grenze des Profils (110 kV: 8°) |
| `light_angle` | leichter Winkelmast | größerer Knick (110 kV: bis 20°) |
| `heavy_angle` | schwerer Winkelmast | 110 kV: bis 30° |
| `strain` | Abspannmast | 110 kV: bis 40° |

**Grenzen der Suche** (aus dem Code von pyorps 0.5.0; die ersten beiden zeigt die kleine Testzelle in 6.2):
- Verbotene Zellen (Wert 65535) sperren auch den Leiter, nicht nur den Maststandort: Eine Wand aus einer einzigen verbotenen Zelle trennt Start und Ziel. Die Leitung kann also keine verbotene Fläche überspannen.
- Findet die Suche keine Leitung, gibt sie ein **leeres Ergebnis ohne Warnung** zurück (0 Masten). Unsere Hilfsfunktion `fl_suche` prüft das deshalb nach jeder Suche.
- An jeder Richtungsänderung steht ein Mast. Jedes Spannfeld ist deshalb eine gerade Linie im Raster.
- Geländemodell (DGM) und Oberflächenmodell (DOM) passt `ConstrainedPathFinder` nach der **Form** des Arrays an das Suchfenster an, nicht nach Koordinaten. Wir geben beide deshalb genau auf dem Gitter des Kostenrasters ein (Setup-Zelle)."""),

    # ------------------------------------------------------------------ 2
    ("code", r'''# Setup für Teil 6: Importe, Variablen aus Teil 3, DGM auf dem Gitter des Kostenrasters
import dataclasses
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource
from rasterio.features import rasterize

from pyorps import PathFinder, InvalidInputError, full_window_buffer_m
# Diese beiden Klassen exportiert pyorps 0.5.0 nicht auf oberster Ebene, deshalb die langen Importpfade:
from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.graph.constrained_path_finder import ConstrainedPathFinder


def de(x, nd=0):
    """Zahl im deutschen Format: Punkt für Tausender, Komma für Dezimalstellen."""
    return f"{x:,.{nd}f}".replace(",", "§").replace(".", ",").replace("§", ".")


# (falls Teil 3 nicht ausgeführt wurde): dieselben Werte wie dort
if "RASTER_A" not in globals():
    BBOX = (427000, 5354000, 432000, 5359000)
    CRS = "EPSG:25832"
    RASTER_A = "data/cost_raster_a.tiff"
    SOURCE = (427558, 5357403)                   # Kinzigtal-Talboden
    TARGET = (430288, 5355723)                   # westliches Ende des Harmersbachtals

FL_PROFILE = Path("data/profiles")              # Kopien der Profile aus dem pyorps-Repository
FL_DGM_DATEI = hole_dgm() if "hole_dgm" in globals() else "data/dem_clip.tif"   # hole_dgm() aus Teil 5

with rasterio.open(RASTER_A) as src:
    fl_kosten, FL_T, FL_CRS = src.read(1), src.transform, src.crs
with rasterio.open(FL_DGM_DATEI) as src:
    _dgm, _dt, _nodata = src.read(1).astype("float32"), src.transform, src.nodata

# Das DGM hat 300 m Rand um das Gebiet. Wir schneiden es auf das Gitter des Kostenrasters zu:
# gleiche Zellgröße, Versatz in ganzen Zellen, danach gleiche Form.
assert np.isclose(_dt.a, FL_T.a) and np.isclose(_dt.e, FL_T.e)
_spalte0, _zeile0 = (FL_T.c - _dt.c) / FL_T.a, (_dt.f - FL_T.f) / abs(FL_T.e)
assert float(_spalte0).is_integer() and float(_zeile0).is_integer()
_spalte0, _zeile0 = int(_spalte0), int(_zeile0)
fl_dgm = _dgm[_zeile0:_zeile0 + fl_kosten.shape[0], _spalte0:_spalte0 + fl_kosten.shape[1]]
assert fl_dgm.shape == fl_kosten.shape and not np.any(fl_dgm == _nodata)
FL_DGM = (fl_dgm, FL_CRS, FL_T)                 # so bekommt ConstrainedPathFinder das DGM: (Array, CRS, Transform)
FL_PUFFER = full_window_buffer_m(RASTER_A)      # Suchfenster = ganzes Raster (warum: siehe 6.2)
FL_TAL = ((427558, 5357403), (427558, 5356400))  # kurzes Punktepaar im Talboden (1 km nach Süden), für 6.6 und die Übung
FL_EXT = (FL_T.c, FL_T.c + FL_T.a * fl_kosten.shape[1], FL_T.f + FL_T.e * fl_kosten.shape[0], FL_T.f)
print(f"DGM: {FL_DGM_DATEI} | zugeschnitten ab Zeile {_zeile0}, Spalte {_spalte0} | Form {fl_dgm.shape} | "
      f"Höhe {de(fl_dgm.min())} bis {de(fl_dgm.max())} m")

FL_TYPEN = {"terminal": ("Endmast", "black"), "suspension": ("Tragmast", "tab:green"),
            "light_angle": ("leichter Winkelmast", "tab:orange"), "heavy_angle": ("schwerer Winkelmast", "tab:red"),
            "strain": ("Abspannmast", "tab:purple")}


def fl_lade_profil(kv, regel="remove_if_valid"):
    """Profil laden; die Endfeldregel setzen wir hier auf "remove_if_valid" (in den Dateien steht "remove", siehe 6.6)."""
    profil = InfrastructureProfile.load(str(FL_PROFILE / f"overhead_line_{kv}kv.yaml"))
    return dataclasses.replace(profil, terminal_span_rule=regel)


def fl_suche(profil, quelle=None, ziel=None, puffer=None, nachbarschaft="r2", **kw):
    """Eine Freileitungssuche; bricht bei leerem Ergebnis ab."""
    quelle, ziel = quelle or SOURCE, ziel or TARGET
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as meldungen:
        warnings.simplefilter("always")
        cpf = ConstrainedPathFinder(RASTER_A, quelle, ziel, profile=profil, neighborhood_str=nachbarschaft,
                                    search_space_buffer_m=puffer or FL_PUFFER, **kw)
        erg = cpf.find_route()
    for m in meldungen:
        if "CUDA" not in str(m.message):
            print(f"  Meldung von pyorps (englisch, {m.category.__name__}): {str(m.message)[:420]}")
    if erg.n_towers == 0:
        raise RuntimeError("Leeres Ergebnis: keine Freileitung im Suchfenster gefunden. "
                           "Größeres Fenster (search_space_buffer_m) oder anderes Punktepaar wählen.")
    print(f"  {profil.name}: {erg.n_towers} Masten, Suche {de(time.perf_counter() - t0, 1)} s")
    return erg, cpf


def fl_tabelle(ergebnisse):
    """Eine Zeile pro Leitung, kurze Spalten (ganze Zahlen, ohne Tausenderzeichen)."""
    zeilen = {}
    for name, erg in ergebnisse.items():
        maste = erg.towers_to_geodataframe(crs=FL_CRS)
        hoehen = maste.loc[maste["tower_type"] != "terminal", "height_m"].value_counts().sort_index()
        zeilen[name] = {"Masten": erg.n_towers, "Länge [m]": round(erg.total_length),
                        "Spannweite [m]": f"{erg.min_span_actual_m:.0f}-{erg.max_span_actual_m:.0f}",
                        "Höhe [m] × Anzahl": ", ".join(f"{h:g}×{n}" for h, n in hoehen.items()),
                        "Raster [EUR]": round(erg.total_terrain_cost), "Masten [EUR]": round(erg.total_tower_cost),
                        "Knicke [EUR]": round(erg.total_angle_penalty_cost), "Gesamt [EUR]": round(erg.total_cost)}
    return pd.DataFrame(zeilen).T


def fl_typen(ergebnisse):
    """Anzahl der Masten je Typ, eine Zeile pro Leitung."""
    return pd.DataFrame({name: {FL_TYPEN[t][0]: erg.tower_type_counts.get(t, 0) for t in FL_TYPEN}
                         for name, erg in ergebnisse.items()}).T


FL_SPALTEN = ("Spalten: Masten mit den 2 Endmasten; Höhe = Aufhängehöhe der Linienmasten; Raster = Rasterkosten der "
              "Route (Wert × Meter); Masten = Fundamente, Mast-Grundpreise und Höhenaufpreise; Knicke = Zuschläge für Knicke")


def fl_karte(ax, ergebnisse, farben, hintergrund="kosten", rand=300, maste=True):
    """Routen (und Masten nach Typ) auf Kostenraster oder Schummerung."""
    if hintergrund == "kosten":
        bild = np.ma.masked_equal(fl_kosten, 65535)
        cmap = plt.get_cmap("YlGnBu").copy(); cmap.set_bad("0.25")
        bild = ax.imshow(bild, cmap=cmap, extent=FL_EXT, vmin=50, vmax=700)
    else:
        bild = ax.imshow(LightSource(azdeg=315, altdeg=45).hillshade(fl_dgm, vert_exag=1, dx=5, dy=5),
                         cmap="gray", extent=FL_EXT)
    alle = []
    for (name, erg), farbe in zip(ergebnisse.items(), farben):
        xy = np.asarray(erg.path_coords); alle.append(xy)
        ax.plot(xy[:, 0], xy[:, 1], color=farbe, lw=1.8, label=name,
                path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])
        if maste and hasattr(erg, "towers"):
            for t in erg.towers:
                ax.plot(t.location.x, t.location.y, "o", ms=5, mfc=FL_TYPEN[t.tower_type][1], mec="white", mew=0.6)
    alle = np.vstack(alle)
    ax.set_xlim(alle[:, 0].min() - rand, alle[:, 0].max() + rand)
    ax.set_ylim(alle[:, 1].min() - rand, alle[:, 1].max() + rand)
    ax.set_aspect("equal"); ax.tick_params(labelsize=8)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("Rechtswert [m]", fontsize=8); ax.set_ylabel("Hochwert [m]", fontsize=8)
    return bild


def fl_legende(ax, typen=True, ncol=4):
    """Legende unter der Karte, damit sie keine Route verdeckt."""
    if typen:
        for typname, farbe in FL_TYPEN.values():
            ax.plot([], [], "o", mfc=farbe, mec="white", label=typname)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=ncol, frameon=False)'''),

    # ------------------------------------------------------------------ 3
    ("md", r"""## 6.1 Das Profil: alle Regeln einer Leitung in einer Datei

Ein `InfrastructureProfile` beschreibt eine Leitungsart. pyorps bringt im Repository drei Profile für 110, 220 und 380 kV mit (Ordner `profiles/`). Sie sind **nicht** im installierten Paket (pip/wheel) enthalten, deshalb liegen Kopien unter `data/profiles/`.

Die Felder, die wir in diesem Teil benutzen (Bedeutung nach dem Code von pyorps 0.5.0):

| Feld | Bedeutung |
|---|---|
| `hard_angle_limit_deg` | größte Richtungsänderung zwischen zwei Rasterschritten; schärfere Knicke sind verboten |
| `min_span_m`, `max_span_m` | kürzeste und längste Spannweite zwischen zwei Masten |
| `tower_cost_params["angle_types"]` | Masttyp nach Knickwinkel am Mast, mit Mast-Grundpreis |
| `tower_cost_params["terrain_cost_map"]` | Fundamentpreis in EUR pro m² je Rasterwert (linear interpoliert), mal `tower_ground_area_m2` |
| `angle_cost_function`, `angle_cost_params` | Zuschlag pro Knick nach Winkel (`piecewise`: linear zwischen den Stützstellen) |
| `max_gradient_percent`, `gradient_cost_params["scale"]` | nur mit DGM: steilster erlaubter Rasterschritt; Faktor `exp(scale × s²)` pro Schritt (s = Steigung als Verhältnis) |
| `tower_heights_m` | Aufhängehöhen des Leiters am Mast (nicht die Mastspitze), eine pro Höhenklasse |
| `conductor_weight_per_m`, `conductor_tension_n`, `min_clearance_m` | Leitergewicht (N/m), Zugkraft (N), Mindestabstand zu Boden und Hindernis (m) |
| `terminal_span_rule` | was mit einem Mast passiert, der näher als `min_span_m` an einem Endpunkt steht (6.6) |

Weitere Felder, kurz: `soft_angle_limit_deg` muss kleiner gleich der harten Grenze sein, die Suche liest es sonst nicht. `span_bin_size_m` ist Pflicht, die Suche fasst Spannweiten aber in Klassen der Breite `min_span_m` zusammen. `tower_height_cost_per_increment` ist der Aufpreis je 3 m über der niedrigsten Höhe. `terminal_tower_height_m` und `terminal_tower_cost` (Endmast) stehen nur im Bericht. `tower_cost_function`, `gradient_cost_function` und `recommended_buffer_m` liest die Suche nicht.

Die Bodenabstandsprüfung läuft nur, wenn ein DGM da ist **und** das Profil Masthöhe, Leitergewicht, Zugkraft und Mindestabstand hat."""),

    # ------------------------------------------------------------------ 4
    ("code", r'''# Profil laden und die Felder zeigen, die wir benutzen
fl_p110 = fl_lade_profil(110)
fl_datei_110 = InfrastructureProfile.load(str(FL_PROFILE / "overhead_line_110kv.yaml"))   # unverändert
fl_werte = {f: getattr(fl_p110, f) for f in ("hard_angle_limit_deg", "min_span_m", "max_span_m", "max_gradient_percent",
                                             "tower_heights_m", "conductor_weight_per_m", "conductor_tension_n",
                                             "min_clearance_m")}
print(pd.Series({k: (", ".join(f"{h:g}" for h in v) if isinstance(v, list) else f"{v:g}") for k, v in fl_werte.items()},
                name=fl_p110.name).to_string())
print(f"terminal_span_rule        {fl_p110.terminal_span_rule} (hier überschrieben, "
      f"in der Datei steht {fl_datei_110.terminal_span_rule})")

print("\nMasttypen (Knick bis ... Grad, Mast-Grundpreis):")
print(pd.DataFrame({FL_TYPEN[k][0]: {"Knick bis [°]": int(v["max_angle_deg"]), "Mast-Grundpreis [EUR]": int(v["base_cost"])}
                    for k, v in fl_p110.tower_cost_params["angle_types"].items()}).T.to_string())

# Fundamentpreis je Mast für einige Werte von Kostenraster A (Tabelle × Fundamentfläche) und Höhenaufpreise
fl_fundament = fl_p110.precompute_tower_terrain_costs()
print("\nFundament pro Mast [EUR] nach Rasterwert (365 = Wald):")
print(pd.Series({w: round(float(fl_fundament[w])) for w in (74, 200, 250, 365, 712)}).to_string())
fl_aufpreis = dict(zip(fl_p110.effective_tower_heights_m, fl_p110.precompute_height_premium()))
print("Höhenaufpreis je Mast [EUR]:", ", ".join(f"{h:g} m: {de(float(fl_aufpreis[h]))}" for h in sorted(fl_aufpreis)))

# Ein eigenes Profil im Code (Pflicht: Name, Beschreibung, beide Winkelgrenzen, Winkelkostenform)
fl_eigen = InfrastructureProfile(
    name="mein_110kv_test", description="kleines Beispiel",
    soft_angle_limit_deg=5.0, hard_angle_limit_deg=30.0,
    angle_cost_function="linear", angle_cost_params={"scale": 1000.0},   # 1000 EUR pro Grad Knick
    min_span_m=60, max_span_m=250, span_bin_size_m=10,
    tower_cost_params={"terrain_cost_map": {"0": 300, "500": 1500},
                       "angle_types": {"suspension": {"max_angle_deg": 5, "base_cost": 30000},
                                       "strain": {"max_angle_deg": 30, "base_cost": 200000}}},
    tower_ground_area_m2=64.0)
fl_kurz = dataclasses.replace(fl_p110, name="110kv_kurze_felder", max_span_m=200)   # Kopie mit einer Änderung
print(f"\nEigenes Profil: {fl_eigen.name} | Spannweitengrenzen: {fl_eigen.has_span_constraints} | "
      f"Bodenabstand prüfbar: {fl_eigen.has_clearance_constraints}")
print(f"Kopie: {fl_kurz.name}, max_span_m = {fl_kurz.max_span_m} | Original: {fl_p110.max_span_m}")'''),

    # ------------------------------------------------------------------ 5
    ("md", r"""## 6.2 Die erste Freileitung (noch ohne Gelände)

Wir verbinden `SOURCE` und `TARGET` aus Teil 3 auf Kostenraster A mit dem 110-kV-Profil.

**Suchfenster:** Die kleine Testzelle unten zeigt: Mit einem Puffer von 200 m oder 400 m um die Luftlinie findet die Suche **keine** Leitung (0 Masten, ohne Warnung), obwohl `PathFinder` im selben Fenster ein Kabel findet. Die zweite Hälfte der Zelle zeigt den Grund an einem Spielraster: Eine Wand aus einer einzigen verbotenen Zelle sperrt die Leitung; mit einer Lücke geht die Route durch die Lücke. Im Kostenraster seht ihr in der Karte weiter unten schmale verbotene Streifen (grau), zum Beispiel entlang der Bahn. Die gefundene Leitung nimmt deshalb einen weiten Umweg nach Westen und Süden, der in den kleinen Fenstern nicht liegt.

Wir suchen also im ganzen Raster (`full_window_buffer_m`): 1 Million Zellen × 16 Richtungen × 6 Spannweitenklassen. In unserem Lauf belegte die Suche ohne Gelände etwa 1,5 GB Arbeitsspeicher.

**Nachbarschaft:** `r2` (16 Richtungen). Mit `r1` (8 Richtungen, 45° Abstand) wäre bei 40° harter Grenze jeder Knick verboten."""),

    # ------------------------------------------------------------------ 6
    ("code", r'''# Kleine Testläufe (je unter einer Sekunde): zu kleines Fenster und eine Sperrwand
with warnings.catch_warnings(record=True) as fl_meldungen:
    warnings.simplefilter("always")
    for fl_puffer in (200, 400):
        fl_klein = ConstrainedPathFinder(RASTER_A, SOURCE, TARGET, profile=fl_p110,
                                         search_space_buffer_m=fl_puffer).find_route()
        fl_kabel_klein = PathFinder(RASTER_A, SOURCE, TARGET, search_space_buffer_m=fl_puffer,
                                    neighborhood_str="r2").find_route()
        print(f"Puffer {fl_puffer} m: Freileitung {fl_klein.n_towers} Masten, "
              f"Kabel (PathFinder) {de(fl_kabel_klein.total_length)} m")
    # Spielraster 1 km × 1 km, überall Wert 100, in Spalte 100 eine Wand aus EINER verbotenen Zelle
    fl_wand = np.full((200, 200), 100, np.uint16)
    fl_wand[:, 100] = 65535
    fl_wand_t = rasterio.transform.from_origin(500000, 5600000, 5, 5)
    fl_a, fl_b = (500102.5, 5599497.5), (500902.5, 5599497.5)      # Spalte 20 und 180, Zeile 100
    for fl_titel in ("Wand ohne Lücke", "Wand mit Lücke im Süden"):
        if fl_titel.endswith("Süden"):
            fl_wand[180:, 100] = 100                                 # die untersten 20 Zellen frei
        fl_r = ConstrainedPathFinder(fl_wand, fl_a, fl_b, profile=fl_p110, transform=fl_wand_t, crs="EPSG:25832",
                                     search_space_buffer_m=2000).find_route()
        fl_text = (f", südlichster Punkt der Route {de(fl_b[1] - np.asarray(fl_r.path_coords)[:, 1].min())} m "
                   f"südlich der Luftlinie" if fl_r.n_towers else "")
        print(f"{fl_titel}: {fl_r.n_towers} Masten{fl_text}")
fl_andere = [str(m.message)[:100] for m in fl_meldungen if "CUDA" not in str(m.message)]
print("Warnungen bei diesen Läufen:", fl_andere or "keine")'''),

    # ------------------------------------------------------------------ 7
    ("code", r'''# Freileitung 110 kV ohne Gelände im ganzen Raster, dazu das Kabel (PathFinder) im selben Fenster
fl_flach, fl_cpf = fl_suche(fl_p110)
fl_kabel = PathFinder(RASTER_A, SOURCE, TARGET, search_space_buffer_m=FL_PUFFER, neighborhood_str="r2").find_route()

fl_maste = fl_flach.towers_to_geodataframe(crs=FL_CRS)
assert np.isclose(fl_maste["total_cost"].sum(), fl_flach.total_tower_cost)   # Mastkosten = Summe der Masten
fl_maste_tab = pd.DataFrame({   # Zahlen mit Dezimalkomma, ohne Tausenderzeichen
    "Typ": fl_maste["tower_type"].map(lambda t: FL_TYPEN[t][0]),
    "Knick [°]": fl_maste["turn_angle_deg"].map(lambda v: de(v, 1)),
    "Spannweite davor [m]": fl_maste["span_to_previous_m"].map(lambda v: "-" if pd.isna(v) else de(v, 1)),
    "Höhe [m]": fl_maste["height_m"].map(lambda v: f"{v:g}"),
    "Fundament [EUR]": fl_maste["terrain_cost"].round().astype(int),
    "Mast-Grundpreis [EUR]": fl_maste["angle_cost"].round().astype(int),
    "Mast gesamt [EUR]": fl_maste["total_cost"].round().astype(int)})
print(fl_maste_tab.head(8).to_string())
print(f"... insgesamt {len(fl_maste)} Masten")

fl_vergleich = pd.DataFrame({
    "Kabel (PathFinder)": {"Länge [m]": round(fl_kabel.total_length), "Raster [EUR]": round(fl_kabel.total_cost),
                           "Masten [EUR]": 0, "Gesamt [EUR]": round(fl_kabel.total_cost)},
    "Freileitung 110 kV": {"Länge [m]": round(fl_flach.total_length), "Raster [EUR]": round(fl_flach.total_terrain_cost),
                           "Masten [EUR]": round(fl_flach.total_tower_cost), "Gesamt [EUR]": round(fl_flach.total_cost)}}).T
print(); print(fl_vergleich.to_string())
print(f"\nAnteil der Masten an den Gesamtkosten der Freileitung: {de(100 * fl_flach.total_tower_cost / fl_flach.total_cost)} %")

# Welche Knicke kann das r2-Raster überhaupt? (Richtungen der 16 Schritte)
fl_schritte = np.asarray(fl_cpf.steps, dtype=float)
fl_richtung = np.degrees(np.arctan2(fl_schritte[:, 0], fl_schritte[:, 1]))
fl_knicke = sorted({round(abs((a - b + 180) % 360 - 180), 1) for a in fl_richtung for b in fl_richtung} - {0.0})
print("kleinste mögliche Knicke in r2:", ", ".join(f"{de(k, 1)}°" for k in fl_knicke[:3]))

fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
fl_bild = fl_karte(ax, {"Freileitung 110 kV": fl_flach, "Kabel (PathFinder)": fl_kabel}, ["crimson", "black"])
fig.colorbar(fl_bild, ax=ax, shrink=0.7, label="Kosten [EUR/m]")
fl_legende(ax)
ax.set_title("Freileitung und Kabel auf Kostenraster A (grau = verboten)", fontsize=9)
plt.show()'''),

    # ------------------------------------------------------------------ 8
    ("md", r"""**Was wir sehen:**
- Die Freileitung ist länger als das Kabel. Sie kann sich nicht durch die engen Lücken im Talboden schlängeln, denn jeder Knick kostet einen Mast und zwischen zwei Masten liegen mindestens 50 m gerade Strecke.
- Die Masten machen den größten Teil der Kosten aus; die Rasterkosten (Rasterwert × Meter, wie beim Kabel) sind der kleinere Teil.
- Im Raster `r2` ist der kleinste Knick 18,4°. Ein Tragmast (bis 8°) steht deshalb nur auf geraden Stücken; an jedem Knick steht ein Winkel- oder Abspannmast.
- Die Kosten des Kabels und der Freileitung sind hier nur Rasterpreise. Ein echter Vergleich Kabel gegen Freileitung braucht auch die Kabelkosten pro Meter.

## 6.3 Drei Spannungsebenen

Dasselbe Punktepaar mit den Profilen für 110, 220 und 380 kV. Die Profile unterscheiden sich vor allem in der längsten Spannweite (300, 350, 400 m), in der Fundamentfläche (100, 144, 256 m²), in den Mastpreisen und in der harten Winkelgrenze (40°, 35°, 30°)."""),

    # ------------------------------------------------------------------ 9
    ("code", r'''# Drei Spannungsebenen, ohne Gelände
fl_ebenen = {"110 kV": fl_flach}
for fl_kv in (220, 380):
    fl_ebenen[f"{fl_kv} kV"], _ = fl_suche(fl_lade_profil(fl_kv))
print(fl_tabelle(fl_ebenen).to_string())
print(FL_SPALTEN)
print("\nMasten je Typ:"); print(fl_typen(fl_ebenen).to_string())

fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
fl_karte(ax, fl_ebenen, ["crimson", "darkorange", "navy"])
fl_legende(ax)
ax.set_title("Drei Spannungsebenen (Punkte = Masten nach Typ)", fontsize=9)
plt.show()'''),

    # ------------------------------------------------------------------ 10
    ("md", r"""**Was die Zahlen zeigen (ohne Gelände):** Mit höherer Spannung werden die Spannfelder länger (die größte Spannweite folgt `max_span_m`), und die Leitung braucht weniger Masten. Jeder Mast ist aber teurer (größeres Fundament, höherer Grundpreis), deshalb steigen die Gesamtkosten trotzdem stark an. Die Aufhängehöhe der Linie steigt mit dem Profil (25, 35, 40 m); ohne DGM prüft die Suche keinen Bodenabstand und meldet für alle Linienmasten die niedrigste Höhe des Profils.

## 6.4 Mit Gelände (DGM)

Jetzt geben wir das DGM mit `dem=` (als `(Array, CRS, Transform)` auf dem Gitter des Kostenrasters). Damit schalten sich drei Dinge zugleich ein: die Neigungsgrenze `max_gradient_percent` (110 kV: 40 %), der Neigungsfaktor auf Route und Fundament, und die Bodenabstandsprüfung mit Durchhang. Welcher Teil wie viel bewirkt, trennen wir hier nicht.

Im Längsprofil zeichnen wir den Leiter als Parabel, so wie pyorps ihn prüft: `z(x) = zA + (zB - zA) · x / L - w · x · (L - x) / (2 T)` mit den Aufhängepunkten zA, zB (Gelände am Mast plus Aufhängehöhe), Spannweite L, Leitergewicht w und Zugkraft T."""),

    # ------------------------------------------------------------------ 11
    ("code", r'''# 110 kV mit DGM
fl_dgm_erg, _ = fl_suche(fl_p110, dem=FL_DGM)
print("Hinweis von pyorps (englisch), sinngemäß: Die Suche hat nur mit der niedrigsten Masthöhe gesucht und diese "
      "Leitung genommen, ohne sie mit Leitungen mit höheren Masten zu vergleichen:")
print("  search_notes:", fl_dgm_erg.search_notes)
print(fl_tabelle({"ohne DGM": fl_flach, "mit DGM": fl_dgm_erg}).to_string())
print(FL_SPALTEN)

fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
fl_karte(ax, {"ohne DGM": fl_flach, "mit DGM": fl_dgm_erg}, ["crimson", "deepskyblue"], hintergrund="schummerung")
fl_legende(ax)
ax.set_title("110 kV ohne und mit DGM (Schummerung)", fontsize=9)
plt.show()


# Längsprofil: Gelände, Masten und Leiter entlang der Route
def fl_laengsprofil(erg, profil, ax, oberflaeche=None, titel="", legende=True):
    """Zeichnet das Längsprofil; gibt den kleinsten Leiterabstand je mittlerem Spannfeld zurück [m].

    Leiter: Parabel aus pyorps (span_clearance.py): z(x) = zA + (zB - zA) x / L - w x (L - x) / (2 T).
    """
    xy = np.asarray(erg.path_coords)
    lauf = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    zeile, spalte = rasterio.transform.rowcol(FL_T, xy[:, 0], xy[:, 1])
    zeile, spalte = np.asarray(zeile), np.asarray(spalte)
    boden = fl_dgm[zeile, spalte]
    oben = boden if oberflaeche is None else oberflaeche[zeile, spalte]
    ax.fill_between(lauf, boden.min() - 20, boden, color="0.8", label="Gelände (DGM)")
    if oberflaeche is not None:
        ax.plot(lauf, oben, color="darkgreen", lw=0.8, label="angenommene Baumkronen (DOM)")
    # Mastpositionen entlang der Route (jeder Mast steht auf einer Zelle der Route)
    pos, start = [], 0
    for t in erg.towers:
        d = np.hypot(xy[start:, 0] - t.location.x, xy[start:, 1] - t.location.y)
        start = start + int(np.argmax(d < 0.5)); pos.append(start)
    w, T = profil.conductor_weight_per_m, profil.conductor_tension_n
    abstand = []
    for i in range(len(pos) - 1):
        a, b = pos[i], pos[i + 1]
        za, zb = boden[a] + erg.towers[i].height_m, boden[b] + erg.towers[i + 1].height_m
        L = lauf[b] - lauf[a]
        x = lauf[a:b + 1] - lauf[a]
        z = za + (zb - za) * x / L - w * x * (L - x) / (2 * T)
        endfeld = i in (0, len(pos) - 2)
        zu_tief = False
        if not endfeld and b - a > 1:
            abstand.append(float(np.min(z[1:-1] - oben[a + 1:b])))
            zu_tief = abstand[-1] < profil.min_clearance_m
        if zu_tief:   # Spannfeld unter dem Mindestabstand: dick und rot
            ax.plot(lauf[a:b + 1], z, color="red", lw=2.5)
        else:
            ax.plot(lauf[a:b + 1], z, color="tab:blue", lw=1, ls="--" if endfeld else "-")
    for k, t in zip(pos, erg.towers):
        ax.plot([lauf[k], lauf[k]], [boden[k], boden[k] + t.height_m], color=FL_TYPEN[t.tower_type][1], lw=2)
    ax.set_xlabel("Entfernung entlang der Route [m]", fontsize=8); ax.set_ylabel("Höhe [m ü. NHN]", fontsize=8)
    ax.set_title(titel, fontsize=9); ax.tick_params(labelsize=8)
    if legende:   # Platzhalter-Linien nur für die Legende
        ax.plot([], [], color="tab:blue", lw=1, label="Leiter")
        ax.plot([], [], color="tab:blue", lw=1, ls="--", label="Leiter im Endfeld (Höhe des Endmasts nur Bericht)")
        ax.plot([], [], color="red", lw=2.5, label="Leiter zu tief (unter min_clearance_m)")
        ax.plot([], [], color="tab:orange", lw=2, label="Mast bis zur Aufhängehöhe (Farbe = Typ)")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    return np.array(abstand)


fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
fl_abstand_dgm = fl_laengsprofil(fl_dgm_erg, fl_p110, ax, titel="110 kV mit DGM: Längsprofil")
plt.show()
print(f"kleinster Abstand Leiter-Boden in den mittleren Spannfeldern: {de(fl_abstand_dgm.min(), 1)} m "
      f"(Vorgabe min_clearance_m = {de(fl_p110.min_clearance_m)} m; nachgerechnet an den Zellen der Route)")'''),

    # ------------------------------------------------------------------ 12
    ("md", r"""## 6.5 Vertiefung (optional): Bodenabstand über Wald mit einem angenommenen Oberflächenmodell (DOM)

Für den Bodenabstand zählt nicht nur der Boden, sondern alles, was darauf steht: Bäume und Gebäude. Das beschreibt ein **digitales Oberflächenmodell** (DOM, englisch DSM, Argument `dsm=`). pyorps rechnet die Hindernishöhe als `max(DOM - DGM, 0)` und verlangt den Mindestabstand über dem Hindernis.

**Wichtig:** Für unser Gebiet haben wir kein offenes DOM. Wir bauen deshalb ein **angenommenes DOM nur für den Unterricht**: Gelände plus 25 m Baumhöhe auf allen ALKIS-Flächen der Klassen Laubholz, Nadelholz, Laub- und Nadelholz und Gehölz, sonst 0 m. Das ist grob (keine Lichtungen, keine Gebäude, gleiche Baumhöhe überall). Für echte Planung nehmt ihr ein gemessenes Oberflächenmodell aus Laserscanning, zum Beispiel das DOM1 der Landesvermessung. Das DOM liegt auf demselben Gitter wie DGM und Kostenraster, aus dem gleichen Grund wie beim DGM.

**Diese Suche ist die langsamste des Notebooks:** In unserem Lauf dauerte sie etwa 20 bis 30 Sekunden auf einem schnellen Laptop und belegte gut 3 GB Arbeitsspeicher. Mit nur 25-m-Masten findet pyorps hier keine Leitung, deshalb sucht es über alle Höhenklassen (25, 34, 42 m)."""),

    # ------------------------------------------------------------------ 13
    ("code", r'''# Angenommenes DOM: Gelände + 25 m auf Waldflächen (nur zum Lernen!)
fl_alkis = gpd.read_file(f"data/layers/alkis_{BBOX[0]}_{BBOX[1]}_{BBOX[2]}_{BBOX[3]}.gpkg")
FL_WALD = ("Laubholz", "Nadelholz", "Laub- und Nadelholz", "Gehölz")
FL_BAUMHOEHE_M = 25.0
fl_baeume = rasterize(((g, FL_BAUMHOEHE_M) for g in fl_alkis.loc[fl_alkis["unterart_name"].isin(FL_WALD), "geometry"]),
                      out_shape=fl_kosten.shape, transform=FL_T, fill=0.0, dtype="float32")
fl_dom = fl_dgm + fl_baeume
print(f"Waldanteil im Raster: {de(100 * (fl_baeume > 0).mean(), 1)} %, "
      f"größte Hindernishöhe DOM - DGM: {de(float((fl_dom - fl_dgm).max()), 1)} m")

fl_dom_erg, _ = fl_suche(fl_p110, dem=FL_DGM, dsm=(fl_dom, FL_CRS, FL_T))
print("search_notes:", fl_dom_erg.search_notes or "leer (die Suche lief über alle Höhenklassen)")
print(fl_tabelle({"mit DGM": fl_dgm_erg, "mit DGM + DOM": fl_dom_erg}).to_string())
print(FL_SPALTEN)
print(f"Höhenaufpreise der Leitung mit DOM: {de(fl_dom_erg.cost_breakdown.get('height_premiums', 0.0))} EUR")

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharey=True, constrained_layout=True)
fl_abstand_alt = fl_laengsprofil(fl_dgm_erg, fl_p110, axes[0], oberflaeche=fl_dom, legende=False,
                                 titel="Leitung nur mit DGM gesucht, gemessen am angenommenen DOM")
fl_abstand_neu = fl_laengsprofil(fl_dom_erg, fl_p110, axes[1], oberflaeche=fl_dom,
                                 titel="Leitung mit DGM + DOM gesucht")
plt.show()
for fl_name, fl_e, fl_a in (("nur DGM", fl_dgm_erg, fl_abstand_alt), ("DGM + DOM", fl_dom_erg, fl_abstand_neu)):
    fl_xy = np.asarray(fl_e.path_coords)
    fl_im_wald = (fl_baeume[rasterio.transform.rowcol(FL_T, fl_xy[:, 0], fl_xy[:, 1])] > 0).mean()
    print(f"{fl_name}: {de(100 * fl_im_wald)} % der Routenzellen im angenommenen Wald; kleinster Abstand "
          f"Leiter-Baumkrone in den mittleren Spannfeldern {de(fl_a.min(), 1)} m; Spannfelder unter "
          f"{de(fl_p110.min_clearance_m)} m: {(fl_a < fl_p110.min_clearance_m).sum()} von {len(fl_a)}")'''),

    # ------------------------------------------------------------------ 14
    ("md", r"""**Was wir sehen:** Die Leitung, die nur mit dem DGM gesucht wurde, hängt über dem angenommenen Wald an mehreren Stellen zu tief. Mit DOM setzt die Suche dort höhere Masten (34 m und 42 m statt 25 m) und verschiebt einzelne Masten; die Gesamtkosten steigen um die Höhenaufpreise und die geänderten Standorte. Der Effekt ist hier moderat: Nur ein kleiner Teil der Routenzellen liegt im angenommenen Wald (siehe Ausgabe).

Zwei Grenzen der Suche (stehen auch in `search_notes` und in der pyorps-Doku):
- Mit mehreren Masthöhen sucht pyorps zuerst nur mit der niedrigsten Höhe und nimmt diese Leitung, wenn es eine gibt. Ob ein einzelner höherer Mast billiger wäre, prüft es dann nicht. Mit DGM allein ist das hier passiert (alle Linienmasten 25 m).
- Die Suche prüft das letzte Spannfeld nicht und rechnet das erste mit einer Höhe der Linie. Die Höhe des Endmasts im Ergebnis (15 m) ist ein Berichtswert; deshalb sind die Endfelder im Längsprofil gestrichelt.

## 6.6 Vertiefung (optional): Die Leitungsenden, Endfeldregel und Anschlusshöhen

Die Suche hält das erste Spannfeld mindestens `min_span_m` lang, nimmt das Ziel aber in jeder Entfernung an. So kann ein Mast sehr nah am Ziel stehen. Was mit ihm passiert, sagt `terminal_span_rule` im Profil:
- `"remove"` (so steht es in den Profildateien): Der Mast wird nach der Suche gestrichen. Seine Kosten fehlen dann in `total_cost`, und es kann ein Knick ohne Mast bleiben (pyorps warnt mit `SpanRuleWarning`).
- `"remove_if_valid"`: nur streichen, wenn die Leitung gültig bleibt; sonst behalten und in `span_notes` vermerken. **Das haben wir in diesem Teil überall benutzt** (`fl_lade_profil` überschreibt den Wert der Datei).
- `"keep"`: jeden Mast behalten, kurze Endfelder nur vermerken.
- `"enforce"`: die Suche selbst hält jedes Endfeld mindestens `min_span_m` lang. Dafür braucht sie exakte Spannweitenklassen (eine Klasse pro Rasterschritt) und viel mehr Speicher.

**Anschlusshöhen** (`source_connection_height_m`, `target_connection_height_m`, Meter über Gelände): die Höhe, in der der Leiter am Umspannwerk oder an einem vorhandenen Mast hängt. Sie wirken nur mit DGM und Bodenabstandsprüfung; dann prüft die Suche auch das erste und das letzte Spannfeld, ebenfalls mit exakten Spannweitenklassen."""),

    # ------------------------------------------------------------------ 15
    ("code", r'''# a) Dieselbe Suche mit DGM, aber mit der Regel "remove" aus der Profildatei
print("Gleich kommt eine englische SpanRuleWarning. Sie sagt: Der Mast 35 m vor dem Ziel wurde gestrichen, "
      "und dort bleibt ein Knick ohne Mast.")
fl_remove, _ = fl_suche(fl_lade_profil(110, "remove"), dem=FL_DGM)
fl_regeln = pd.DataFrame({
    name: {"Masten": e.n_towers, "Gesamt [EUR]": round(e.total_cost), "gestrichene Masten": len(e.removed_towers),
           "deren Kosten [EUR]": round(e.cost_breakdown.get("removed_towers", 0))}
    for name, e in (("remove", fl_remove), ("remove_if_valid", fl_dgm_erg))}).T
print(fl_regeln.to_string())
print("span_notes bei remove_if_valid (englisch, sinngemäß: kurzes Endfeld behalten, sonst bliebe ein Knick ohne Mast):")
print("  ", fl_dgm_erg.span_notes)

# b) "enforce" im ganzen Raster: exakte Klassen brauchen zu viel Speicher, pyorps lehnt vor der Suche ab
try:
    fl_suche(fl_lade_profil(110, "enforce"), dem=FL_DGM)
except InvalidInputError as fl_fehler:
    print("\nenforce im ganzen Raster abgelehnt, weil die exakte Suche zu viele Zustände bräuchte "
          "(Meldung englisch, die Zahl ist die Anzahl der Zustände):")
    print("  ", str(fl_fehler)[:230], "...")

# c) Kleines Fenster im Talboden (FL_TAL, 1 km nach Süden): ohne und mit Anschlusshöhen 20 m / 12 m
fl_tal = {"ohne Anschlusshöhen": fl_suche(fl_p110, *FL_TAL, puffer=250, dem=FL_DGM)[0],
          "Quelle 20 m, Ziel 12 m": fl_suche(fl_p110, *FL_TAL, puffer=250, dem=FL_DGM,
                                             source_connection_height_m=20.0, target_connection_height_m=12.0)[0]}
print(pd.DataFrame({
    name: {"Masten": e.n_towers, "Gesamt [EUR]": round(e.total_cost),
           "Endmast Quelle / Ziel [m]": f"{e.towers[0].height_m:g} / {e.towers[-1].height_m:g}",
           "erste / letzte Spannweite [m]": f"{e.spans[0]:.0f} / {e.spans[-1]:.0f}",
           "Spannweitenklasse [m]": de(e.span_bin_size_m, 2)}
    for name, e in fl_tal.items()}).T.to_string())
print("search_notes (englisch, sinngemäß: exakte Spannweitenklassen von knapp 5 m statt 50 m):")
print("  ", fl_tal["Quelle 20 m, Ziel 12 m"].search_notes)'''),

    # ------------------------------------------------------------------ 16
    ("md", r"""**Was wir sehen:** Mit `"remove"` sieht die Leitung billiger aus, aber nur weil der gestrichene Mast nicht mehr in `total_cost` zählt; dafür bleibt kurz vor dem Ziel ein Knick ohne Mast. `"remove_if_valid"` behält den Mast und sagt es in `span_notes`. Im Talboden ändern die Anschlusshöhen die Kosten nicht; die exakte Suche setzt aber den letzten Linienmast an eine andere Stelle, und die Endmasten tragen die angegebenen Höhen. Im ganzen Raster sind exakte Klassen zu groß: Dafür brauchen wir ein kleineres Fenster, eine kürzere `max_span_m` oder ein gröberes Raster."""),

    # ------------------------------------------------------------------ 16b
    ("md", r"""## 6.7 Ergebnisse speichern

Route und Masten gehen als GeoPackage (zwei Layer) und die Masten zusätzlich als GeoJSON nach `data/ergebnisse/`. Wir speichern die Leitung mit DOM, wenn ihr die Vertiefung 6.5 gerechnet habt, sonst die Leitung mit DGM aus 6.4; die Zelle sagt, welche. Die Masten sind Quadrate in der Größe der Fundamentfläche (10 m × 10 m), gedreht in die Winkelhalbierende der Leitung; in QGIS könnt ihr sie direkt anzeigen."""),

    # ------------------------------------------------------------------ 17
    ("code", r'''# Export: die Leitung mit DOM (6.5), sonst mit DGM (6.4), sonst ohne Gelände (6.2)
for fl_export_name, fl_export_var in (("mit DGM + DOM (6.5)", "fl_dom_erg"), ("mit DGM (6.4)", "fl_dgm_erg"),
                                      ("ohne Gelände (6.2)", "fl_flach")):
    if fl_export_var in globals():
        fl_export = globals()[fl_export_var]
        break
else:
    raise NameError("Noch keine Leitung gerechnet: bitte zuerst die Zellen ab 6.2 ausführen.")
print("Wir speichern die 110-kV-Leitung", fl_export_name)
FL_ERGEBNISSE = Path("data/ergebnisse")
FL_ERGEBNISSE.mkdir(parents=True, exist_ok=True)
fl_gpkg = FL_ERGEBNISSE / "freileitung_110kv.gpkg"
fl_gpkg.unlink(missing_ok=True)                 # bei erneutem Lauf neu schreiben

fl_route = gpd.GeoDataFrame(
    {"profil": [fl_export.profile_name], "variante": [fl_export_name], "masten": [fl_export.n_towers],
     "laenge_m": [fl_export.total_length], "gesamt_eur": [fl_export.total_cost],
     "endfeldregel": [fl_export.terminal_span_rule]},
    geometry=[fl_export.path_geometry], crs=FL_CRS)
fl_mastflaechen = fl_export.towers_to_geodataframe(crs=FL_CRS)
fl_route.to_file(fl_gpkg, layer="route", driver="GPKG")
fl_mastflaechen.to_file(fl_gpkg, layer="masten", driver="GPKG")
fl_mastflaechen.to_file(FL_ERGEBNISSE / "masten_110kv.geojson", driver="GeoJSON")

print(gpd.list_layers(fl_gpkg).to_string(index=False))
print(f"Masten zurückgelesen: {len(gpd.read_file(fl_gpkg, layer='masten'))} | Spalten: {list(fl_mastflaechen.columns)}")'''),

    # ------------------------------------------------------------------ 18
    ("md", r"""## Was wir mitnehmen

- `ConstrainedPathFinder` sucht Route **und** Masten zusammen; das Profil (`InfrastructureProfile`, YAML-Datei) enthält alle Regeln.
- Die Masten dominieren die Kosten. Höhere Spannung heißt hier: längere Spannfelder, weniger, aber teurere Masten.
- Verbotene Zellen sperren auch den Leiter. Enge Lücken im Kostenraster zwingen eine Freileitung zu Umwegen; ein zu kleines Suchfenster gibt ein **leeres Ergebnis ohne Warnung**.
- DGM und DOM für `ConstrainedPathFinder` immer auf dem Gitter des Kostenrasters übergeben.
- Mit DGM werden Neigung und Bodenabstand geprüft; mit DOM auch Bäume. Das DOM in diesem Notebook ist eine Annahme zum Lernen.
- `search_notes`, `span_notes` und die Endfeldregel lesen: Sie sagen, wo die Suche abkürzt.

**In der GUI** (sie ist englisch beschriftet): Der Schalter **"Constrained routing (overhead line / towers)"** im Routing-Bereich blendet die Profiloptionen ein; "Run routing" rechnet dann eine Freileitung auf denselben Punkten (jeder Zwischenpunkt erzwingt einen Mast). Das Profil erscheint als Formular mit 24 Feldern und drei Tabellen:

| GUI-Feld | Profilfeld |
|---|---|
| Soft turn limit / Hard turn limit (forbidden above) | `soft_angle_limit_deg` / `hard_angle_limit_deg` |
| Turn cost shape, Turn cost factor, Tabelle der Knickkosten | `angle_cost_function`, `angle_cost_params` |
| Shortest span / Longest span / Span step | `min_span_m` / `max_span_m` / `span_bin_size_m` |
| Tabelle der Fundamentkosten, Between table values | `terrain_cost_map` (EUR pro m²), `terrain_interpolation` |
| Tabelle der Masttypen | `angle_types` |
| Steepest allowed slope / Slope penalty factor | `max_gradient_percent` / `gradient_cost_params["scale"]` |
| Base attachment height / Height options / Extra cost per 3 m step | `tower_height_m` / `tower_heights_m` / `tower_height_cost_per_increment` |
| Height at the end points / Cost of each end tower | `terminal_tower_height_m` / `terminal_tower_cost` |
| Conductor weight / Conductor tension / Ground clearance | `conductor_weight_per_m` / `conductor_tension_n` / `min_clearance_m` |
| Foundation area / Foundation pricing | `tower_ground_area_m2` / `tower_area_cost_mode` |
| End-span rule | `terminal_span_rule` |

Dazu kommen Name, Beschreibung und "Suggested corridor width (info only)". Die Anschlusshöhen stehen im Block "Line ends"."""),

    # ------------------------------------------------------------------ 19
    ("md", r"""### Übung Teil 6

1. **Feinere Richtungen:** Rechnet die Leitung ohne Gelände mit 32 Richtungen: `fl_suche(fl_p110, nachbarschaft="r3")`. Wie ändern sich Zahl der Masten, Gesamtkosten und Rechenzeit? **Achtung:** `r3` im ganzen Raster braucht mehr Arbeitsspeicher als `r2` und kann einen Laptop mit 8 GB an die Grenze bringen. Mit wenig Speicher nehmt ihr das kleine Talpaar: `fl_suche(fl_p110, *FL_TAL, puffer=250, nachbarschaft="r3")` und vergleicht mit `nachbarschaft="r2"`.
2. **Spannungsebenen mit Gelände:** Sucht die 380-kV-Leitung mit `dem=FL_DGM` und vergleicht sie mit `fl_dgm_erg` (110 kV). Gilt "höhere Spannung, weniger Masten" auch im Gelände? Schaut auf `max_gradient_percent` und `hard_angle_limit_deg` der beiden Profile.
3. **Kürzere Spannfelder:** Sucht mit dem Profil `fl_kurz` (`max_span_m = 200`) ohne Gelände. Wie viele Masten kommen dazu, und was kostet das?"""),
]
