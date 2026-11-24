# ruff: noqa: E501, B028  (long German prose lines in the cells)
"""Abschnitt 2: Kostenraster aus mehreren überlappenden Layern (Workshop-Notebook, deutsch).

Definiert CELLS = [("md", text) | ("code", quelltext), ...].
Die Code-Zellen laufen von oben nach unten mit dem Arbeitsordner workshop/.
Nach dem ersten Lauf liegen data/layers/*.gpkg, data/cost_raster_a.tiff und data/cost_raster_b.tiff im Ordner;
danach läuft alles ohne Netz. Der Layer-Cache trägt das Fenster im Dateinamen: ein anderes BBOX lädt neu.
"""

CELLS = [

    # ------------------------------------------------------------------ 0: Idee
    ("md", """---
# Teil 3: Kostenraster aus mehreren überlappenden Layern

**Warum mehrere Layer?** Planerinnen und Planer suchen nicht die kürzeste, sondern die günstigste Route, und dabei reden viele Datenquellen mit:
Nutzung des Bodens, Schutzgebiete, vorhandene Straßen und eigene Planungsvorgaben. Legen wir jede Quelle als eigenen Layer auf ein Raster,
wird jede Annahme einzeln sichtbar und lässt sich einzeln ändern, ohne alles neu zu zeichnen.

Ein Kostenraster ist ein Gitter. Jede Zelle sagt, was ein Meter Kabel in dieser Zelle kostet (EUR/m).
Die Werte sind ganze Zahlen vom Typ `uint16`; der Wert **65535 heißt verboten**.
Wir bauen das Raster in zwei Schritten: Ein **Basis-Layer** deckt die ganze Fläche ab und gibt jeder Zelle einen Grundwert.
Darüber legen wir **modifizierende Layer**, die sich überlappen dürfen. Jeder Layer ändert nur die Zellen, die er bedeckt:
er **ersetzt** den Wert oder **multipliziert** ihn mit einem Faktor, auch mit einem Faktor unter 1.
Die Layer werden nacheinander angewendet. Gesperrte Zellen bleiben dabei gesperrt, und bei zwei Faktoren auf derselben Zelle multiplizieren sich beide.

Unser Untersuchungsgebiet ist ein Fenster von 5 × 5 km im Kinzigtal bei Biberach, dort mündet das Harmersbachtal ein: ein Talboden, Hänge mit Wald und der Rand des Schwarzwalds.

| Layer | Quelle | Wirkung |
|---|---|---|
| Tatsächliche Nutzung (ALKIS) | LGL-BW, WFS | Basis: Grundwert je Nutzungsart |
| Naturschutzgebiete | LUBW, WFS | ersetzen durch 65535 |
| Biotope (Biotopkartierung) | LUBW, WFS | ersetzen durch 65535 (unsere Annahme) |
| FFH-Gebiete (Natura 2000) | LUBW, WFS | Faktor 1,75 |
| Wasserschutzgebiete | LUBW, WFS | Faktor 1,5 |
| Überschwemmungsgebiete | LUBW, WFS | Faktor 1,3 |
| Straßenbündelung | Straßenflächen aus ALKIS, Puffer 10 m | Faktor 0,8 |
| Planungssperrfläche | eigenes Polygon | ersetzen durch 65535 |

*© LGL-BW (ALKIS) und LUBW (Schutzgebiete, Biotope, Überschwemmungsgebiete), Datenlizenz Deutschland – Namensnennung – Version 2.0 (www.govdata.de/dl-de/by-2-0); die Daten wurden zugeschnitten, gerastert und umklassifiziert.*
"""),

    # ------------------------------------------------------------------ 1: Setup
    ("code", '''# Bausteine für diesen Abschnitt
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from shapely.geometry import box, LineString, Point
import rasterio
from rasterio.transform import array_bounds, rowcol

from pyorps import (PathFinder, GeoRasterizer, CostAssumptions,
                    initialize_geo_dataset, WFSResponseParsingError,
                    full_window_buffer_m, route_metrics)


def de(x, nd=0):
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    return f"{x:,.{nd}f}".replace(",", "§").replace(".", ",").replace("§", ".")


# Untersuchungsgebiet: 5 × 5 km im Kinzigtal bei Biberach (EPSG:25832, Koordinaten in Metern)
BBOX = (427000, 5354000, 432000, 5359000)   # (min Rechtswert, min Hochwert, max Rechtswert, max Hochwert)
CRS = "EPSG:25832"
AUFLOESUNG = 5                              # Zellgröße in Metern
VERBOTEN = 65535                            # uint16-Maximum: die Zelle ist gesperrt

# Layer-Cache und Kostenraster liegen unter data/
LAYER_DIR = Path("data/layers")
LAYER_DIR.mkdir(parents=True, exist_ok=True)
RASTER_A = "data/cost_raster_a.tiff"
RASTER_B = "data/cost_raster_b.tiff"'''),

    # ------------------------------------------------------------------ 2: Loader
    ("md", """### Layer laden: erst der Cache, dann das Netz

Die Layer kommen von WFS-Servern (Web Feature Service) der Landesbehörden.
Die Funktion `lade_layer` schaut zuerst in den Cache `data/layers/`. Nur wenn die Datei fehlt, lädt sie über pyorps
(`initialize_geo_dataset` und `load_data`), schneidet auf das Fenster zu und speichert das Ergebnis.
Wir behalten nur die Spalte, die das Kostenmodell braucht; so bleiben die Dateien klein.

**Kein Netz nötig:** Die Cache-Dateien liegen schon im Repository. Deshalb läuft dieses Notebook ohne Internet, und die Download-Zweige von `lade_layer` laufen im Workshop nicht.
Wollt ihr einen Layer neu laden, löscht seine Datei in `data/layers/` und führt die Zelle noch einmal aus.
Das Fenster steckt im Dateinamen: Wer `BBOX` ändert, bekommt neue Dateien und lädt neu; alte Dateien eines anderen Fensters werden nie benutzt."""),

    ("code", '''def layer_datei(name, bbox=BBOX):
    """Cache-Datei eines Layers; das Fenster steht im Namen, damit nie ein Layer des falschen Fensters gelesen wird."""
    return LAYER_DIR / f"{name}_{'_'.join(str(int(v)) for v in bbox)}.gpkg"


def lade_layer(name, wfs, bbox, spalte=None):
    """Layer aus dem Cache lesen; fehlt er, per WFS laden, auf bbox zuschneiden und speichern."""
    datei = layer_datei(name, bbox)
    if datei.exists():
        return gpd.read_file(datei), "Cache"
    try:
        ds = initialize_geo_dataset(wfs, bbox=bbox)            # WFS-Anfrage vorbereiten
        ds.load_data()                                         # herunterladen
        gdf = ds.data.clip(box(*bbox), keep_geom_type=True)    # der Server liefert auch Objekte, die über den Rand ragen
        gdf = gdf[[spalte, "geometry"] if spalte else ["geometry"]]
    except WFSResponseParsingError:                            # kein Objekt im Fenster
        gdf = gpd.GeoDataFrame({spalte: []} if spalte else None, geometry=[], crs=CRS)
    gdf.to_file(datei, driver="GPKG")
    return gdf, "WFS"


lubw = "https://rips-gdi.lubw.baden-wuerttemberg.de/arcgis/services/wfs"
alkis_wfs = {"url": "https://owsproxy.lgl-bw.de/owsproxy/wfs/WFS_LGL-BW_ALKIS?version=2.0.0",
             "layer": "Tatsaechliche Nutzung"}


def lubw_wfs(layer):
    return {"url": f"{lubw}/{layer}/MapServer/WFSServer", "layer": layer}


QUELLEN = {   # Name im Cache: (WFS-Anfrage, Spalte, Behörde)
    "alkis":           (alkis_wfs, "unterart_name", "LGL-BW"),
    "naturschutz":     (lubw_wfs("Naturschutzgebiet"), None, "LUBW"),
    "biotop":          (lubw_wfs("Biotopkartierung"), None, "LUBW"),
    "ffh":             (lubw_wfs("FFH_Gebiet"), None, "LUBW"),
    "wasserschutz":    (lubw_wfs("Wasserschutzgebiet"), None, "LUBW"),
    "ueberschwemmung": (lubw_wfs("Ueberschwemmungsgebiet"), None, "LUBW"),
}

layer, zeilen = {}, []
for name, (wfs, spalte, behoerde) in QUELLEN.items():
    layer[name], herkunft = lade_layer(name, wfs, BBOX, spalte)
    anteil = layer[name].area.sum() / box(*BBOX).area * 100
    zeilen.append((name, len(layer[name]), round(anteil, 1), behoerde, herkunft))

print(pd.DataFrame(zeilen, columns=["Layer", "Objekte", "Fläche [%]", "Quelle", "geladen aus"]).to_string(index=False))
print("Layer ohne Objekt im Fenster:", [n for n, g in layer.items() if len(g) == 0] or "keine")'''),

    # ------------------------------------------------------------------ 4: Basis
    ("md", """### Basis-Layer: Kosten je Nutzungsart

**Leere Layer.** Die Tabelle oben zeigt, was im Fenster wirklich liegt. Ein Layer ohne Objekt ist leer:
der Server meldet das als Fehler (`WFSResponseParsingError`), `lade_layer` fängt ihn ab und speichert einen leeren Layer.
Die Naturschutzgebiete laden wir trotzdem mit, damit die Regel "Naturschutz = gesperrt" im Modell steht, wenn ihr das Fenster wechselt.
In unseren Läufen hatten die Vogelschutz- und Landschaftsschutzgebiete aus dem OSMSES-Tutorial in diesem Fenster keine Objekte; wir laden sie deshalb nicht.
Als zweiten Sperr-Layer nehmen wir die kartierten **Biotope**: eine Annahme für den Workshop, keine Rechtsauskunft.

Der ALKIS-Layer hat in der Spalte `unterart_name` die Nutzungsart jeder Fläche.
Wir ordnen jeder Art einen Wert in EUR/m zu. Die Kostenwerte stammen aus dem Tutorial zum OSMSES-Workshop 2026
(Ordner `tutorial/` im Repository, https://github.com/marhofmann/pyorps); Arten, die es dort nicht gibt, haben wir ergänzt (Kommentar `# neu`).
Eine Art ohne Eintrag behandelt pyorps (mit einer Warnung) als verboten, also als 65535. Deshalb prüft die Zelle unten mit `assert`,
dass jede Art im Fenster einen Wert hat.

**Auflösung:** 5 m pro Zelle, also 1000 × 1000 = 1 Million Zellen. Das reicht für Wege und Gräben,
und die Suchen der späteren Abschnitte laufen darauf in kurzer Zeit (bei 2,5 m wären es 4 Millionen Zellen).
Eine Zelle bekommt den Wert einer Fläche nur, wenn ihr Mittelpunkt in der Fläche liegt. Schmale verbotene Flächen können deshalb Löcher haben
oder ganz verschwinden. pyorps warnt davor lang und ausführlich (`on_thin_features`); wir schalten die Warnung hier ab.
Für eine echte Genehmigungsplanung würden wir feiner rastern."""),

    ("code", '''KOSTEN_A = {   # Variante A "wirtschaftlich": EUR/m je Nutzungsart (Spalte unterart_name)
    # --- verboten: Bebauung, Flugplatz, Abbau ---
    "Wohnbaufläche": VERBOTEN,
    "Gebäude- und Freifläche Industrie und Gewerbe": VERBOTEN,
    "Gebäude- und Freifläche Land- und Forstwirtschaft": VERBOTEN,
    "Handel und Dienstleistung": VERBOTEN,
    "Friedhof": VERBOTEN,
    "Tagebau/Grube/Steinbruch": VERBOTEN,                                  # neu (wie Bergbau)
    "Flugverkehr": VERBOTEN,
    "Fläche gemischter Nutzung": VERBOTEN,                                 # neu
    "Gebäude- und Freifläche Versorgungsanlage": VERBOTEN,                 # neu
    "Gebäude- und Freifläche Entsorgungsanlage": VERBOTEN,                 # neu
    "Gebäude- und Freifläche Sport, Freizeit und Erholung": VERBOTEN,      # neu
    "Landwirtschaftliche Betriebsfläche": VERBOTEN,                        # neu
    "Lagerfläche": VERBOTEN,                                               # neu
    "Öffentliche Zwecke": VERBOTEN,                                        # neu
    "Historische Anlage": VERBOTEN,                                        # neu
    "Medien und Kommunikation": VERBOTEN,                                  # neu
    # --- begehbar, nach Kosten sortiert ---
    "Unland/Vegetationslose Fläche": 92,
    "Weg": 97,
    "Sportanlage": 107,                                                    # neu (wie Sport-, Freizeit- und Erholungsfläche)
    "Graben": 120,
    "Platz": 152,
    "Parkplatz": 152,                                                      # neu
    "Parken": 152,                                                         # neu
    "Bach": 155,                                                           # neu (wie Fließgewässer)
    "Fluss": 155,                                                          # neu (wie Fließgewässer)
    "Kanal": 155,                                                          # neu (wie Fließgewässer)
    "See": 155,                                                            # neu (wie Stehendes Gewässer)
    "Teich": 155,                                                          # neu (wie Stehendes Gewässer)
    "Straßenverkehr": 178,
    "Grünland": 200,
    "Grünanlage": 200,                                                     # neu
    "Brachland": 200,                                                      # neu
    "Ackerland": 250,
    "Garten": 300,
    "Rebfläche": 300,                                                      # neu (wie Garten; im Fenster nicht vorhanden)
    "Laubholz": 365,
    "Nadelholz": 365,
    "Laub- und Nadelholz": 365,
    "Gehölz": 365,                                                         # neu
    "Bahnverkehr": 415,
}

# Beweis: jede Nutzungsart im Fenster hat einen Kostenwert
arten = set(layer["alkis"]["unterart_name"])
ohne_kosten = sorted(arten - set(KOSTEN_A))
print("Nutzungsarten im Fenster:", len(arten), "| ohne Kostenwert:", ohne_kosten or "keine")
assert not ohne_kosten, "Kostenwert fehlt für: " + ", ".join(ohne_kosten)


def basis_raster(kosten, aufloesung=AUFLOESUNG):
    """Rastert den ALKIS-Layer mit einer Kostentabelle und gibt den GeoRasterizer zurück (Raster in .raster)."""
    gr = GeoRasterizer(layer_datei("alkis"), CostAssumptions({"unterart_name": kosten}))
    gr.rasterize(resolution_in_m=aufloesung,            # Zellgröße in Metern
                 bounding_box=box(*BBOX),               # Rasterrand = Fenster, nicht die Ausdehnung der Daten
                 on_thin_features="ignore")             # keine lange Warnung (siehe Text)
    return gr


def zaehle(raster):
    """Anzahl der verbotenen Zellen."""
    return int((raster == VERBOTEN).sum())


gr = basis_raster(KOSTEN_A)
basis = gr.raster.copy()          # Kopie: die Layer unten ändern gr.raster an Ort und Stelle
assert array_bounds(*basis.shape, gr.transform) == BBOX
print(f"Raster: {basis.shape[0]} × {basis.shape[1]} Zellen, {basis.dtype}, Zellgröße {gr.transform.a:g} m")
print(f"verbotene Zellen: {de(zaehle(basis))} ({de(zaehle(basis) / basis.size * 100, 1)} %)")

# Darstellung: Farbe = EUR/m, schwarz = verboten
EXTENT = (BBOX[0], BBOX[2], BBOX[1], BBOX[3])
KARTE = plt.get_cmap("YlOrRd").copy()
KARTE.set_bad("black")                                  # maskierte Zellen erscheinen schwarz


def beschrifte(ax):
    """Achsen einer Karte: Koordinaten in Metern, Schrift mindestens 8 pt (für den Beamer)."""
    ax.set_xticks(range(BBOX[0], BBOX[2], 2000))
    ax.set_yticks(range(BBOX[1], BBOX[3], 2000))
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Rechtswert [m]", fontsize=9)
    ax.set_ylabel("Hochwert [m]", fontsize=9)


def zeige(ax, raster, titel, vmax=800, alpha=1.0):
    """Kostenraster zeichnen; Werte über vmax bekommen die dunkelste Farbe."""
    bild = ax.imshow(np.ma.masked_equal(raster, VERBOTEN), cmap=KARTE, vmin=0, vmax=vmax,
                     extent=EXTENT, origin="upper", interpolation="nearest", alpha=alpha)
    ax.set_title(titel, fontsize=10)
    beschrifte(ax)
    return bild


def farbleiste(fig, bild, ax):
    """Farbleiste mit Beschriftung."""
    leiste = fig.colorbar(bild, ax=ax, label="Kosten [EUR/m]", extend="max", shrink=0.8)
    leiste.ax.tick_params(labelsize=8)
    leiste.set_label("Kosten [EUR/m]", fontsize=9)
    return leiste


fig, ax = plt.subplots(figsize=(7, 6))
bild = zeige(ax, basis, "Basis-Kostenraster (Variante A, nur ALKIS)")
farbleiste(fig, bild, ax)
ax.legend(handles=[Patch(color="black", label="verboten (65535)")], loc="lower left", fontsize=8)
plt.show()'''),

    # ------------------------------------------------------------------ 6: Schutzgebiete
    ("md", """### Modifizierende Layer: Schutzgebiete und Überschwemmungsgebiete

Jetzt kommen die Layer von der LUBW obendrauf. Wir beschreiben jeden Layer mit zwei Angaben:
dem Wert und `multiply`. Mit `multiply=False` ersetzt der Wert die Zellen, mit `multiply=True` wird multipliziert.
`modify_raster_from_dataset` liest den Layer aus unserer GPKG-Datei, rastert ihn auf dasselbe Gitter und ändert `gr.raster` an Ort und Stelle.
Zellen mit dem Wert 65535 lässt die Methode standardmäßig unverändert (`ignore_value=65535`).
Leere Layer überspringen wir."""),

    ("code", '''SCHUTZ_A = {   # Layer im Cache: (Wert, multiply)
    "naturschutz":     (VERBOTEN, False),   # ersetzen: gesperrt
    "biotop":          (VERBOTEN, False),   # ersetzen: gesperrt (unsere Annahme)
    "ffh":             (1.75, True),        # Faktor
    "wasserschutz":    (1.5, True),         # Faktor
    "ueberschwemmung": (1.3, True),         # Faktor
}


def schutz_anwenden(gr, wirkung, ausgabe=True):
    """Wendet die Layer der LUBW nacheinander an und zählt die verbotenen Zellen davor und danach."""
    for name, (wert, multiplizieren) in wirkung.items():
        if len(layer[name]) == 0:
            if ausgabe:
                print(f"  {name:16s} übersprungen (kein Objekt im Fenster)")
            continue
        vorher = zaehle(gr.raster)
        gr.modify_raster_from_dataset(layer_datei(name), cost_assumptions=wert, multiply=multiplizieren)
        if ausgabe:
            art = f"Faktor {str(wert).replace('.', ',')}" if multiplizieren else f"ersetzen durch {wert}"
            print(f"  {name:16s} {art:22s} verbotene Zellen: {de(vorher):>8} -> {de(zaehle(gr.raster)):>8}")


print(f"nach dem Basis-Layer: {de(zaehle(gr.raster))} verbotene Zellen")
schutz_anwenden(gr, SCHUTZ_A)
mit_schutz = gr.raster.copy()'''),

    # ------------------------------------------------------------------ 8: eigene Layer
    ("md", """### Layer anderer Art: Puffer und eigene Polygone

Nicht jeder Layer kommt von einem WFS. `modify_raster_from_dataset` nimmt auch ein `GeoDataFrame` entgegen.
Wir zeigen zwei Arten:

1. **Straßenbündelung:** Wir wählen die Straßenflächen aus dem ALKIS-Layer, legen einen **Puffer von 10 m** darum (`geometry_buffer_m=10`)
   und multiplizieren mit **0,8**. Ein Faktor unter 1 macht die Zellen billiger: Wer neben einer Straße baut, spart Aufwand.
2. **Planungssperrfläche:** Ein Polygon, das wir selbst im Notebook zeichnen: ein geplantes Gewerbegebiet von 350 m × 300 m
   auf dem Talboden, das frei bleiben soll. Es ersetzt die Zellen durch den Wert 65535.

Unten sehen wir vier Bilder: Basis, mit den Layern der LUBW, mit Straßenbündelung und Sperrfläche und eine Karte der Änderungen."""),

    ("code", '''strassen = layer["alkis"][layer["alkis"]["unterart_name"] == "Straßenverkehr"]

# eigenes Polygon: ein GeoDataFrame mit einer Fläche; (x, y) ist die untere linke Ecke, 350 m breit, 300 m hoch
SPERR_X, SPERR_Y = 428100, 5357150
sperrflaeche = gpd.GeoDataFrame({"name": ["geplantes Gewerbegebiet"]},
                                geometry=[box(SPERR_X, SPERR_Y, SPERR_X + 350, SPERR_Y + 300)], crs=CRS)


def eigene_layer_anwenden(gr, buendelung):
    """Straßenbündelung (Puffer + Faktor) und Planungssperrfläche (ersetzen)."""
    gr.modify_raster_from_dataset(strassen, cost_assumptions=buendelung, multiply=True,
                                  geometry_buffer_m=10)          # Puffer in Metern, vor dem Rastern
    gr.modify_raster_from_dataset(sperrflaeche, cost_assumptions=VERBOTEN)   # multiply=False: ersetzen


vorher = zaehle(gr.raster)
eigene_layer_anwenden(gr, 0.8)
endraster_a = gr.raster.copy()
print(f"Straßenbündelung und Sperrfläche: verbotene Zellen {de(vorher)} -> {de(zaehle(endraster_a))}")
print(f"Basis -> Ende: {de(zaehle(basis))} -> {de(zaehle(endraster_a))} verbotene Zellen")

# Karte der Änderungen: Basis gegen Endraster
aenderung = np.zeros(basis.shape, dtype=np.uint8)
aenderung[(endraster_a > basis) & (endraster_a != VERBOTEN)] = 1      # teurer
aenderung[endraster_a < basis] = 2                                    # billiger
aenderung[(endraster_a == VERBOTEN) & (basis != VERBOTEN)] = 3        # neu gesperrt

fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
for ax, raster_i, titel in zip(axes.ravel()[:3], (basis, mit_schutz, endraster_a),
                               ("1. Basis (ALKIS)", "2. mit Layern der LUBW", "3. mit Straßenbündelung\\nund Sperrfläche")):
    bild = zeige(ax, raster_i, titel)
leiste = fig.colorbar(bild, ax=axes.ravel().tolist(), extend="max", shrink=0.6)
leiste.set_label("Kosten [EUR/m]", fontsize=9)
leiste.ax.tick_params(labelsize=8)
farben = ["#d9d9d9", "#f4a582", "#4393c3", "black"]
axes[1, 1].imshow(aenderung, cmap=ListedColormap(farben), vmin=0, vmax=3, extent=EXTENT,
                  origin="upper", interpolation="nearest")
axes[1, 1].set_title("4. Änderung gegenüber der Basis", fontsize=10)
beschrifte(axes[1, 1])
axes[1, 1].legend(handles=[Patch(color=f, label=t) for f, t in
                           zip(farben, ("unverändert", "teurer", "billiger", "neu gesperrt"))],
                  loc="lower left", fontsize=8)
plt.show()'''),

    # ------------------------------------------------------------------ 10: Varianten
    ("md", """### Zwei Kostenvarianten auf denselben Layern

Dieselben Layer, andere Zahlen: Variante A ist die **wirtschaftliche** (die Werte von oben).
Variante B ist **naturverträglich**: Wald und Schutzgebiete sind viel teurer, Gewässer sind teuer zu queren, Felder sind billiger.

| Parameter | A "wirtschaftlich" | B "naturverträglich" |
|---|---|---|
| Wald (Laubholz, Nadelholz, Gehölz) | 365 EUR/m | 657 EUR/m (Faktor 1,8) |
| Ackerland / Grünland | 250 / 200 EUR/m | 150 / 130 EUR/m |
| Bach, Fluss, Kanal, See, Teich | 155 EUR/m | 600 EUR/m |
| FFH- / Wasserschutz- / Überschwemmungsgebiet | Faktor 1,75 / 1,5 / 1,3 | Faktor 4 / 3 / 2 |
| Straßenbündelung | Faktor 0,8 | Faktor 0,6 |
| verboten (Bebauung, Biotope, Sperrfläche) | gleich | gleich |

Alle Schritte stecken in der Funktion `baue_raster`. Sie speichert das Ergebnis mit `save_raster` als GeoTIFF.
Variante A muss dasselbe Raster liefern wie die Einzelschritte oben; die Zelle prüft das mit `assert` und meldet es."""),

    ("code", '''KOSTEN_B = dict(KOSTEN_A)                                # Kopie, dann ändern
for wald in ["Laubholz", "Nadelholz", "Laub- und Nadelholz", "Gehölz"]:
    KOSTEN_B[wald] = round(KOSTEN_A[wald] * 1.8)         # Wald × 1,8
KOSTEN_B.update({"Ackerland": 150, "Grünland": 130,
                 "Bach": 600, "Fluss": 600, "Kanal": 600, "See": 600, "Teich": 600})
SCHUTZ_B = {**SCHUTZ_A, "ffh": (4.0, True), "wasserschutz": (3.0, True),
            "ueberschwemmung": (2.0, True)}


def baue_raster(kosten, schutz, buendelung, pfad):
    """Basis, Layer der LUBW, Straßenbündelung, Sperrfläche; speichert das Raster als GeoTIFF."""
    gr = basis_raster(kosten)
    schutz_anwenden(gr, schutz, ausgabe=False)
    eigene_layer_anwenden(gr, buendelung)
    gr.save_raster(pfad)
    return gr


VARIANTEN = {"A": dict(kosten=KOSTEN_A, schutz=SCHUTZ_A, buendelung=0.8, pfad=RASTER_A),
             "B": dict(kosten=KOSTEN_B, schutz=SCHUTZ_B, buendelung=0.6, pfad=RASTER_B)}
raster = {}
for v, parameter in VARIANTEN.items():
    raster[v] = baue_raster(**parameter).raster.copy()
    groesse = Path(parameter["pfad"]).stat().st_size / 1e6
    print(f"Variante {v}: {parameter['pfad']} ({de(groesse, 1)} MB), verbotene Zellen {de(zaehle(raster[v]))} "
          f"({de(zaehle(raster[v]) / raster[v].size * 100, 1)} %), mittlerer Wert der freien Zellen "
          f"{de(raster[v][raster[v] != VERBOTEN].mean())} EUR/m")

assert np.array_equal(raster["A"], endraster_a), "Funktion und Einzelschritte liefern verschiedene Raster"
with rasterio.open(RASTER_A) as src:                     # Gegenprobe: die Datei enthält dasselbe Raster
    assert np.array_equal(src.read(1), raster["A"])
    print(f"Datei: {src.width} × {src.height} Zellen, {src.dtypes[0]}, {src.crs}, Zellgröße {src.res[0]:g} m")
print("Funktion und Einzelschritte liefern dasselbe Raster A; die Datei enthält dasselbe Raster.")'''),

    # ------------------------------------------------------------------ 12: Routen
    ("md", """### Dieselbe Strecke, zwei Kostenwelten

Jetzt rechnen wir mit `PathFinder` die günstigste Route von `SOURCE` nach `TARGET` (3 bis 4 km Luftlinie), einmal pro Raster.
`SOURCE` liegt auf dem Talboden, `TARGET` auf der anderen Seite eines bewaldeten Hangs.
Wir nutzen dabei das ganze Raster als Suchfenster (`search_space_buffer_m=full_window_buffer_m(...)`) und die 16er-Nachbarschaft `"r2"`.
Alle Punkte müssen auf freien Zellen liegen; sonst verschiebt pyorps sie mit einer Warnung auf eine andere Zelle. Das prüfen wir mit `assert`.
Die erste Tabelle zeigt auch, wie viele Meter jede Route durch Wald, durch Schutzgebiete (FFH und Wasserschutz) und durch Überschwemmungsgebiet führt.

**Vorsicht beim Vergleichen:** Die Kosten der Routen stammen aus zwei verschiedenen Kostentabellen, sie sind nicht direkt vergleichbar.
Fair ist die Gegenprobe: Wir bewerten jede Route auf beiden Rastern (`route_metrics` rechnet wie `PathFinder`;
die Diagonale der zweiten Tabelle stimmt mit den Kosten der ersten Tabelle überein).

Die Punktlisten `SOURCES` (3 Punkte) und `TARGETS` (4 Punkte) brauchen wir in den nächsten Abschnitten für Mehrfach-Suchen."""),

    ("code", '''SOURCE = (427558, 5357403)        # Rechtswert, Hochwert (EPSG:25832): auf dem Talboden der Kinzig
TARGET = (430288, 5355723)        # am Rand des Harmersbachtals, hinter einem bewaldeten Hang
SOURCES = [SOURCE, (428238, 5358313), (428663, 5356873)]
TARGETS = [TARGET, (431613, 5355083), (429853, 5355148), (431273, 5356048)]

# Alle Punkte müssen in beiden Rastern auf freien Zellen liegen
with rasterio.open(RASTER_A) as src:
    transform = src.transform
for v in raster:
    for punkt in [SOURCE, TARGET, *SOURCES, *TARGETS]:
        zeile, spalte = rowcol(transform, *punkt)
        assert raster[v][zeile, spalte] != VERBOTEN, f"{punkt} ist in Variante {v} gesperrt"

luftlinie = np.hypot(TARGET[0] - SOURCE[0], TARGET[1] - SOURCE[1])
assert 3000 <= luftlinie <= 4000, "SOURCE und TARGET sollen 3 bis 4 km auseinander liegen"

routen = {}
for v, pfad in (("A", RASTER_A), ("B", RASTER_B)):
    finder = PathFinder(dataset_source=pfad, source_coords=SOURCE, target_coords=TARGET,
                        search_space_buffer_m=full_window_buffer_m(pfad),   # ganzes Raster
                        neighborhood_str="r2")
    routen[v] = finder.find_route()                      # Dijkstra, eine Route

wald = layer["alkis"][layer["alkis"]["unterart_name"].isin(
    ["Laubholz", "Nadelholz", "Laub- und Nadelholz", "Gehölz"])].union_all()
schutzflaeche = pd.concat([layer["ffh"], layer["wasserschutz"]]).union_all()
ueberschwemmt = layer["ueberschwemmung"].union_all()

zeilen = []
for v, p in routen.items():
    g = p.path_geometry
    zeilen.append((v, round(p.total_length), round(p.total_cost), round(p.total_length / luftlinie, 2),
                   round(g.intersection(wald).length), round(g.intersection(schutzflaeche).length),
                   round(g.intersection(ueberschwemmt).length)))
print(pd.DataFrame(zeilen, columns=["Variante", "Länge [m]", "Kosten [EUR]", "Umweg-Faktor", "davon Wald [m]",
                                    "davon Schutzgebiet [m]", "davon Überschwemmungsgebiet [m]"]).to_string(index=False))
abstand = routen["A"].path_geometry.hausdorff_distance(routen["B"].path_geometry)
print(f"Luftlinie {de(luftlinie)} m | größter Abstand zwischen den Routen A und B: {de(abstand)} m")
im_wasserschutz = layer["wasserschutz"].union_all().contains(Point(SOURCES[1]))
print("SOURCES[1] =", SOURCES[1], "liegt im Wasserschutzgebiet:", "ja" if im_wasserschutz else "nein")

# Gegenprobe: jede Route auf beiden Rastern bewerten (die Diagonale ist die Zahl von PathFinder).
# route_metrics erwartet das Suchfenster als Raster; hier ist das Fenster das ganze Raster.
kreuz = {}
for rv in raster:
    bewertung = route_metrics(routen, raster[rv], AUFLOESUNG)   # Route -> Länge und Kosten auf diesem Raster
    kreuz[f"auf Raster {rv} [EUR]"] = [round(bewertung[k]["construction_cost"]) for k in routen]
print(pd.DataFrame(kreuz, index=[f"Route {k}" for k in routen]).to_string())

fig, ax = plt.subplots(figsize=(8, 6))
bild = zeige(ax, raster["A"], "Zwei Routen, gleiche Punkte (Hintergrund: Raster A)", alpha=0.5)
farbleiste(fig, bild, ax)
for v, farbe in (("A", "tab:blue"), ("B", "tab:green")):
    p = routen[v]
    x, y = p.path_geometry.xy
    ax.plot(x, y, color=farbe, lw=2.2,
            label=f"Variante {v}: {de(p.total_length)} m, {de(p.total_cost / 1000)} kEUR")
ax.plot(*SOURCE, "o", ms=9, mfc="white", mec="black", label="SOURCE")
ax.plot(*TARGET, "s", ms=9, mfc="white", mec="black", label="TARGET")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)   # unter der Karte, verdeckt keine Route
plt.show()'''),

    # ------------------------------------------------------------------ 13: Auswertung
    ("md", """### Was die Tabellen zeigen

- Route B ist deutlich länger als Route A, hat aber fast keine Meter im Wald (Spalte "davon Wald"). Route A läuft ein Stück über den bewaldeten Hang und ist kürzer.
  B nimmt den Umweg über den Talboden, weil Wald in Variante B viel teurer ist.
- Die Kosten in der ersten Tabelle stammen aus zwei verschiedenen Kostentabellen. Dass Route B dort billiger aussieht, sagt nichts über die Routen selbst.
- Die zweite Tabelle vergleicht fair: Auf Raster A ist Route A billiger als Route B, auf Raster B ist Route B billiger als Route A.
  Jede Route ist in ihrer eigenen Kostenwelt die beste.
- Die Annahmen in der Kostentabelle entscheiden also mit über die Route, nicht nur die Geländedaten: Eine Kostentabelle ist eine planerische Entscheidung, und wir sollten sie offenlegen."""),

    # ------------------------------------------------------------------ 14: Kurzform
    ("md", """### Kurzform: alles in einem `PathFinder`-Aufruf

Wer das Raster nicht speichern will, gibt `PathFinder` die Kostentabelle und die Layer direkt mit:
`cost_assumptions` für den Basis-Layer und `datasets_to_modify` für die modifizierenden Layer (dieselben Angaben wie bei `modify_raster_from_dataset`).
Der Puffer `search_space_buffer_m` ist der Abstand um die Luftlinie, in dem gesucht wird. Die Diagonale unseres 5-km-Fensters ist 5 km × √2, also etwa 7,07 km;
mit 7100 m ist das Suchfenster daher das ganze Fenster, die Ausgabe unten zeigt die Rastergröße.
Wir nehmen die Dateien aus dem Cache, also ohne Netz, und vergleichen mit Variante A."""),

    ("code", '''layer_a = [{"input_data": layer_datei(name), "cost_assumptions": wert, "multiply": mult}
           for name, (wert, mult) in SCHUTZ_A.items() if len(layer[name]) > 0]
layer_a += [{"input_data": strassen, "cost_assumptions": 0.8, "multiply": True, "geometry_buffer_m": 10},
            {"input_data": sperrflaeche, "cost_assumptions": VERBOTEN}]

kurz = PathFinder(dataset_source=layer_datei("alkis"),           # Basis-Layer aus dem Cache
                  source_coords=SOURCE, target_coords=TARGET,
                  cost_assumptions={"unterart_name": KOSTEN_A},
                  datasets_to_modify=layer_a,
                  resolution_in_m=AUFLOESUNG,                    # wie oben
                  on_thin_features="ignore",                     # wie oben: keine lange Warnung
                  search_space_buffer_m=7100,                    # etwas mehr als die Diagonale des Fensters
                  neighborhood_str="r2")
pfad_kurz = kurz.find_route()
zeilen_kurz, spalten_kurz = kurz.geo_rasterizer.raster.shape
print(f"Raster der Kurzform: {zeilen_kurz} × {spalten_kurz} Zellen")
print(f"Kurzform: {de(pfad_kurz.total_length)} m, {de(pfad_kurz.total_cost)} EUR")
print(f"Variante A oben: {de(routen['A'].total_length)} m, {de(routen['A'].total_cost)} EUR")'''),

    # ------------------------------------------------------------------ 16: Übung
    ("md", """### Übung Teil 3

Die nächsten Abschnitte brauchen `RASTER_A`, `RASTER_B`, `SOURCE` und `TARGET` unverändert. Arbeitet in der Übung deshalb mit neuen Namen und neuen Dateien.

1. **Preise ändern:** Kopiert die Tabelle (`eigene_kosten = dict(KOSTEN_A)`) und setzt alle Wald-Arten (`"Laubholz"`, `"Nadelholz"`, `"Laub- und Nadelholz"`, `"Gehölz"`)
   auf 500 statt 365. Baut mit `baue_raster(eigene_kosten, SCHUTZ_A, 0.8, "data/uebung_1.tiff")` ein neues Raster und rechnet die Route von `SOURCE` nach `TARGET`
   noch einmal (neuer `PathFinder` auf `data/uebung_1.tiff`). Wohin wandert die Route, und wie viele Meter laufen jetzt durch den Wald?
2. **Eigene Layer:** Baut ein Raster mit `basis_raster(KOSTEN_A)` und wendet `schutz_anwenden` und `eigene_layer_anwenden` an.
   Legt dann eine vorhandene Leerrohrtrasse darüber, zum Beispiel `LineString([(428500, 5357000), (429500, 5356300), (430250, 5355800)])`:
   ein `GeoDataFrame` mit `crs=CRS`, `geometry_buffer_m=4`, `cost_assumptions=20` und `ignore_value=None`, damit die Leerrohrtrasse auch gesperrte Zellen quert.
   Speichert unter einem neuen Namen (`save_raster`) und rechnet die Route neu: Wie stark zieht die Leerrohrtrasse die Route an? Probiert danach eine zweite Sperrfläche
   quer über Route A, zum Beispiel `box(429300, 5355500, 429700, 5355800)`.
3. **Schutzgebiete treffen:** Route A läuft ein Stück durch Schutzgebiet (siehe erste Tabelle). Nehmt `ziel_nw = SOURCES[1]`, einen Punkt im Wasserschutzgebiet im Nordwesten
   (die Ausgabe oben bestätigt, dass er dort liegt), und rechnet mit je einem neuen `PathFinder` auf `RASTER_A` und `RASTER_B` von `SOURCE` nach `ziel_nw`.
   Messt die Meter im Schutzgebiet wie in der Zelle oben (`intersection`): Wo laufen A und B jetzt?"""),
]
