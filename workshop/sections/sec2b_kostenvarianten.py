# ruff: noqa: E501, B028  (long German prose lines in the cells)
"""Abschnitt 2, zweiter Teil: Layer anderer Art, zwei Kostenvarianten, Vergleich und Übung (Workshop-Notebook, deutsch).

Setzt den Namensraum von sec2_kostenraster fort (Code-Zellen laufen von oben nach unten mit dem Arbeitsordner workshop/).
"""

CELLS = [

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
