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
]
