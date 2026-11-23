# Data provenance of the free-siting case study

Where every data file came from, when it was fetched, how, what was done with it, and what it is used
for. Written 2026-10-05. Three records go together:

| record | what it holds | who writes it |
|---|---|---|
| `data_manifest.yaml` | one entry per layer: source, licence, capture date and time, SHA-256, how fetched, what it is used for | the fetch scripts and `DataManifest.annotate` |
| `download_log.jsonl` | one line per download event, never rewritten: service, request, counts, file, hash, tool versions, script hash | the fetch scripts (`scripts/download_log.py`) |
| this file | the readable account, with the reasoning and the limits | written by hand, kept in step with the two above |

Times are local time in Germany, CEST (UTC+2) on 2026-10-05. UTC is the local time minus two hours.
Hashes are SHA-256; the first 16 hex digits are shown here, the full value is in the manifest.

## 1. What was downloaded on 2026-10-05

All through the pyorps WFS and WCS access, the same services and calls as the CIRED 2026 scripts. The
last three rows were written to the log by the fetch script itself at the moment of the download; the
first seven were logged afterwards (section 8). The service and layer names of the last three were found by
listing the state's service catalogue on 2026-10-05; they are not in the pyorps presets.
The owner gave permission on 2026-10-05 ("you may download the data yourself"), approved the land-use
download after its size was stated, and approved the 1 m DEM in tiles.

| layer | content | service | finished (CEST) | counts | file and size | sha256 |
|---|---|---|---|---|---|---|
| `water_zones` | drinking-water protection zones | HLNUG WFS `.../inspire/bewirtschaftungsgebiete/MapServer/WFSServer`, layer `TWS_HQS_TK25` | 14:41:52 | 99 in the mask box, 98 kept | `layers/water_zones.gpkg`, 0.29 MB | 02d25cde9061 |
| `soil_units` | soil units 1:500 000 | HLNUG WFS `.../inspire/boden/MapServer/WFSServer`, layer `Bodeneinheiten_Bodenuebersicht_500000` | 14:41:53 | 31, 31 | `layers/soil_units.gpkg`, 0.14 MB | e10a6e296646 |
| `nature_reserves` | nature reserves (NSG) | HMLU WFS `.../inspire/schutzgebiete/MapServer/WFSServer`, layer `Naturschutzgebiete` | 14:41:54 | 13, 13 | `layers/nature_reserves.gpkg`, 0.13 MB | a2e11a8aa039 |
| `landscape_protection` | landscape protection areas (LSG) | same service, layer `Landschaftsschutzgebiete` | 14:41:55 | 1, 1 | `layers/landscape_protection.gpkg`, 0.16 MB | b3cd81f7193f |
| `natura2000_habitats` | Natura 2000 habitats directive areas (FFH) | same service, layer `Fauna-Flora-Habitate` | 14:41:56 | 9, 9 | `layers/natura2000_habitats.gpkg`, 0.22 MB | 25271d383ec3 |
| `bird_protection_areas` | bird protection areas (SPA) | same service, layer `Vogelschutzgebiete` | 15:02:58 | 2, 2 | `layers/bird_protection_areas.gpkg`, 0.10 MB | 9ecc2c6d2646 |
| `biotopes_protected` | legally protected biotopes (section 30 BNatSchG) | HLNUG WFS `.../inspire/lebensraeume_biotope/MapServer/WFSServer`, layer `GESCH_BIOTOPE` | 15:03:01 | 1 112 in the mask box, 1 103 kept | `layers/biotopes_protected.gpkg`, 1.38 MB | b96a306d5bc5 |
| `biotope_complexes_protected` | legally protected biotope complexes | same service, layer `GESCH_KOMPLEXE` | 15:03:02 | 42, 42 | `layers/biotope_complexes_protected.gpkg`, 0.37 MB | 159b9e90b3ab |
| `alkis_land_use` | cadastre actual land use (incl. roads and paths) | Hessen ALKIS WFS `https://www.gds.hessen.de/wfs2/aaa-suite/cgi-bin/alkis/vereinf/wfs`, layer `ave:Nutzung` | 14:45:09 (started about 14:45:02) | 32 199 in the mask box, 31 938 kept | `layers/alkis_land_use.gpkg`, 21.04 MB | 95d5e973f4ae |
| `dem` | terrain model, 1 m | Hessen WCS 2.0.1 `https://inspire-hessen.de/raster/dgm1/ows`, coverage `he_dgm1` | 14:54:54 (started 14:47:22) | 72 tiles of 2 km, 262 329 324 cells | `dem/dgm1.tif`, 531.7 MB | 782149e5ccc2 |

**Study area.** The mask `case_studies/cired2026/data/input/raster_mask.shp` (23.5 x 11.2 km, EPSG:25832,
x 431 374 to 454 874, y 5 582 217 to 5 593 379). Its shapefile hash (`.shp`) is 22609a553a07. The DEM
covers the mask's bounding box snapped outward to whole metres (x 431 373 to 454 875).

## 2. How each file was fetched

**Vector layers (six).** `scripts/fetch_layers.py` counts the features in the mask's bounding box with a
WFS 2.0.0 `resultType=hits` request, then loads the layer with `initialize_geo_dataset({"url", "layer"},
mask=mask)` and `load_data()` (the pyorps WFS loader, which clips to the mask polygon), writes a GeoPackage and
registers it in the manifest. A layer above `--max-features` (default 5 000) is refused unless the owner has
approved it and `--allow-large` is given; the land-use layer was run that way. Licence text is copied from
the service's own capabilities document (`Fees` and `AccessConstraints`).

**DEM.** `scripts/fetch_dem.py` asks the WCS for 2 000 m tiles (`GetCoverage`, GeoTIFF, `E` and `N` subsets),
one at a time with a 3 s pause, holds each in memory and writes it once as a window of one internally tiled,
deflate-compressed float32 GeoTIFF (`dgm1.tif.partial`, renamed when complete). One file, one process,
no raw intermediate copies (the university ransomware guard).

**Tools** (this machine): Python 3.13.5, geopandas 1.0.1, pyogrio 0.11.0, rasterio 1.4.3, requests 2.32.3.
pyorps is an editable install of the working tree whose package metadata still says 0.3.2 (the repository
says 0.4.0), so the version string is not a reliable identifier; the script hashes below are.

| script | sha256 as used |
|---|---|
| `fetch_layers.py` | b1d9944a20b45bbd |
| `fetch_dem.py` | 6433cc6a3dd735d3 |

(Both scripts were extended afterwards to write the log themselves; the hashes above are of the versions
that ran.)

## 3. What was done with the files after fetching

Only checks, nothing that changes the data:

- All nine vector files: read back through `DataManifest.read_vector`, hash verified, CRS EPSG:25832, every
  geometry valid (counts 98, 31, 13, 1, 9, 31 938, 2, 1 103 and 42). Biotopes: 1 112 counted, 1 103 kept, the
  same box-corner explanation, not individually checked.
- Land use: 31 938 features, no duplicate `gml_id`, none outside the mask polygon. The service counted 32 199
  features in the mask's bounding box; the 261 not kept are the box corners outside the mask polygon (the
  polygon fills 99.2 % of its box, and 261 of 32 199 is 0.8 %). Water zones: 99 counted, 98 kept, the same
  explanation, not individually checked.
- Land use feature update dates run from 2009-09-17 to 2026-10-01, so the cadastre was current to about a
  week before the download.
- DEM: re-read in full: 23 502 x 11 162 = 262 329 324 cells, no missing cell, elevation 105.8 to 398.3 m,
  grid origin on whole metres, hash verified.

## 3b. Derived rasters built from the layers (2026-10-05)

`scripts/build_cost_raster.py` (recipe in `scripts/cost_recipe.py`) built three rasters on the DEM's 1 m grid
(23 502 x 11 162 cells, origin x 431 373, y 5 593 379, whole-metre aligned, EPSG:25832). It verified the
manifest hash of every input first, ran as one process (about 22 s for all three), and registered the outputs with
the input hashes in the log.

| layer | content | file | sha256 |
|---|---|---|---|
| `cost_base_1m` | EUR per metre of trench by land-use class, uint16, 65535 = forbidden; main roads buffered by name prefix (A 10 m, B 4 m, L 2 m) | `rasters/cost_base_1m.tif`, 524.7 MB (uncompressed, written by the pyorps rasterizer) | 9c8e1c777848 |
| `protection_flags_1m` | bits 0-5: nature reserve, landscape protection, Natura 2000 habitat, bird area, protected biotope, protected biotope complex; bits 8-11: water-protection zone code (1 = zone I most restrictive ... 5 = IIIB) | `rasters/protection_flags_1m.tif`, 2.2 MB | c74fb0ae8706 |
| `soil_unit_ids_1m` | uint8 id of the soil unit, 0 = none; id table `rasters/soil_unit_ids_1m.json` | `rasters/soil_unit_ids_1m.tif`, 1.4 MB | 6ad825f5d6d6 |

**Protected areas are priced** (owner, 2026-10-05). `cost_priced_1m` (7.5 MB, sha256 d33dc2f2729c) is the closed cost
map multiplied inside protected areas by `consent_classes_2026.yaml` (HDD price 1.5 from the catalogue for nature
reserves, Natura 2000, bird areas and protected biotopes, 1.0 for landscape protection, water zones I 100, II 2,
III 1.5, IIIA 1.5, IIIB 1.2): 38.6 % of the cells are priced, forbidden cells are unchanged, no cell is cheaper,
cells outside every protected area are unchanged (all checked over the whole raster before registering). The one-off
permit costs and procedure times are not in the raster; they belong to the consent lattice. All 20 values of the
consent table are LOW-confidence stand-ins from the plan's table (the middle of its ranges).

Inputs (first 12 hex of their hashes): land use 95d5e973f4ae, soil units e10a6e296646, the six protection
layers and the water zones as in section 1, the DEM 782149e5ccc2 (grid only). The class costs are the CIRED 2026
**stand-ins** (the owner kept all stand-ins, 2026-10-05), not sourced prices of this study.

**Design choice (owner, 2026-10-05).** Protection is stored as separate flag layers and is **not** burned into
the cost, so the legal choice (hard or priced, plan Q12) can be made later without rebuilding the 0.5 GB map.

**Result.** 262 329 324 cells; 12.2 % forbidden (land-use classes marked forbidden in the recipe, e.g.
residential, industrial); 24 distinct cost values; the most common are 437 (arable and grassland, 46 % of cells),
425 (mixed forest, 21 %) and forbidden (12 %). Flag cells: nature reserves 2.7 M, landscape protection 10.6 M,
Natura 2000 26.8 M, bird areas 1.4 M, protected biotopes 3.4 M, biotope complexes 2.2 M, water zones 87.0 M.

**Checks.**
- Grid: the rasterizer's output grid equals the DEM grid exactly (the script refuses to register otherwise).
- A 2 km test box was built first in a scratch folder.
- **Comparison with the CIRED 2026 routing surface** (`mod2_raster_wp.tiff`, whose grid is offset by 0.38-0.77 m):
  200 000 to 300 000 random points sampled from both. Outside every protection flag, the old value equals the
  new base value or `floor(base x 1.15)` at 96.8 % of passable points. The 1.15 follows three soil units
  (Braunerden; Braunerden with Pseudogleye and Podsol-Braunerden; Rendzinen: 98 % each), and the other units
  are unchanged. Two small units (about 1 400 and 600 points) match neither and stay unresolved. So the
  land-use recipe is reproduced, and the CIRED 2026 surface additionally carried soil and protection
  multipliers that this map does not.

**Problem found with the first burn (pyorps' own audit, severity CRITICAL), and its fix.** In the first plain
burn (`cost_base_1m`, built 2026-10-05) of the 9 157 forbidden land-use features 15 burned no cell, 24 burned with holes, 99
touch only at corners and 220 are narrower than 1.41 m; routes could cross them (60 of 320 test routes did). On
2026-10-05 the owner required that routes never traverse forbidden features, so the cost map was **closed**: the
rasterizer's `all_touched` repair burns every cell a forbidden polygon touches, and `build_cost_raster.py`
refuses to register unless an independent check (its own `all_touched` burn of the forbidden polygons) finds no
touched cell that is not forbidden. Result (`cost_base_1m_closed`, 7.1 MB, compressed): 0 misses over 30.8 M touched
cells; forbidden share 12.55 % (the plain burn 12.2 %); 915 194 free cells in 12 enclosed pockets are sealed,
the main free region keeps 99.9 % of the free cells and contains all seven turbines (the three connection points
lie on forbidden substation cells in both maps; the endpoint rule handles them). **The price:** forbidden areas
grow by up to one cell, which also closes real sub-cell openings; that is the intended, conservative direction.
`scripts/verify_forbidden_closure.py` traces random least-cost routes with the widest step set (r3) and tests each
step's exact chord against the forbidden cells and each route polyline against the forbidden polygons: closed map 0
violations in 320 routes (about 100 000 steps), the plain burn 60 routes through polygons. The first burn
`cost_base_1m` is kept and marked superseded, for the comparison.

## 3c. Site raster, farm window (2026-10-05, 15:31 to 15:33 CEST)

`scripts/build_site_raster.py --window farm --check-candidates` built the first **site** raster: EUR per m2 of switchyard, 65535 =
cannot build, on the whole-metre window of the farm plus 1 km (4 374 x 2 947 m, x 441 701 to 446 075, y 5 584 764 to 5 587 711).
The constraint model (statutes, setbacks, premiums, the base price per land-use class) is the TopoMILP site-raster script's, imported
unchanged (script sha256 b07b6242920d; the folder is called `examples/cired2027`, which is only a name, no paper is planned for that
conference). `site_cost_farm_eur_m2.tif` (0.3 MB, sha256 18e0c4b55007) and `site_cost_farm_sidecar.json` (constraint model, ledger, input
hashes, warnings, the candidate screen). **Closed burn:** the base forbidden classes and every exclusion burn all touched cells, and an
independent check refused the build unless all of them were forbidden (passed). Treatments for protected areas follow the owner's answer of
2026-10-05: nature reserves, Natura 2000, bird areas, water zones I and II, protected biotopes and biotope complexes are excluded;
landscape protection and water zone III are premiums (x1.8; x1.25 and x1.15).

**Result.** 12 890 178 cells, 63.58 % buildable, 20 distinct prices from 250 to 2 242 EUR/m2; the largest single exclusions are the
300 m residential setbacks (8.6 and 7.1 points) and the Natura 2000 areas with their 300 m buffer (6.0 points). **Candidate screen of the six
hand-placed CIRED 2026 sites (60 x 80 m, 36 rotations): every site fails** (one lies outside the window); the sites that come closest still have 351 to 446
forbidden footprint cells (about 7-9 %). This repeats what the script's own header predicts and is the reason the plan frees the site.
An earlier run of the unmodified script (plain burn, 64.38 % buildable, log `site_cost_farm_run_2026-10-05.log`, file
`site_cost_farm_eur_m2.tif`, replaced by the closed build) differs by 0.8 points: the closed burn and the padded land-use layer (a railway 158
features within the 300 m padding that the unpadded fetch missed).

**New layers fetched for this raster** (seven, saved under `layers/site_constraints/`, each with a log line and a manifest entry; the log time is the
time they were saved and registered after the raster was built, a few minutes after the fetch itself): buildings (1 534), water zones on the
cadastre base (7), biotopes (141), a natural-monument layer (1), flood areas HQ100 and flow areas (1 and 1), low-probability flooding (0). The licence
text of these seven entries still reads "to be confirmed at download": the services' capabilities were not read for them. **[open]**

**Limits.** Warnings in the run: four road polygons with no name carry no building-ban zone (39 917 m2); the filters for Autobahn names match nothing in the window
(none there). The base price table (300, 1 200, 1 500 EUR/m2 by class) and the premium values are the legacy stand-ins that plan C6 replaces. Only the farm window
exists; the envelope and the whole study mask are not built.

### Site raster, envelope window (2026-10-05, 15:58 to 16:07 CEST; 472 s)

The same builder for the **envelope** window (turbines, the three connection points and 2 km, clipped to the DEM grid): x 431 748 to 453 908, y 5 582 217 to 5 592 961,
22 160 x 10 744 = 238 087 040 cells. `site_cost_envelope_eur_m2.tif` (3.4 MB, sha256 4cababdd1447) and its sidecar JSON; closure check passed. Seven constraint layers fetched
for the window and registered (log time 16:06:41 to 16:06:43, the time they were saved after the raster was built): buildings 77 516 features (28.5 MB), biotopes 3 208 (3.6 MB),
flood areas, natural monuments and water zones on the cadastre base (each below 1 MB). **Result: 34.0 % buildable** (the land-use table alone leaves 77.2 %), 61 distinct prices
from 250 to 2 837 EUR/m2. The largest single exclusions: the 300 m setback around residential parcels (22.4 points) and the Natura 2000 areas with their 300 m buffer
(11.2 points), then the 300 m setback around residential buildings (5.2 points). Those 300 m setbacks are engineering judgement in the legacy model, not statute, and
decide most of the result: they must be a sensitivity in the paper. The window touches the south edge of the study mask, where layers we hold are clipped to the mask, so
setbacks from features outside it are missing there (permissive at that edge). All six CIRED 2026 hand-placed sites fail the footprint screen (accepted by the owner, they were a
proof of concept).

## 3d. Exact site-cost field, farm window (2026-10-05, about 16:20 CEST, 103 s)

`scripts/compute_site_field.py --window farm` ran the plan's integer-exact FFT field (`pyorps.siting.site_field`, base-256 limb layers) over
`site_cost_farm` for a 60 x 80 m pad at 36 rotations (5 degree steps over 180), prices in integer cents. Outputs on the site raster's grid:
`site_field_farm_cents.tif` (uint32, 0.8 MB, sha256 c36560b00055; 4294967295 = infeasible) and `site_field_farm_rotation.tif` (uint8, 0.3 MB, sha256
13e54e2d18bd; 255 = none). **Independent check before registering:** 400 random feasible anchors re-priced by direct pixel sums (0 cost mismatches, no
cheaper rotation) and 100 infeasible anchors (0 wrongly infeasible). **Result:** 4 070 468 of 12 890 178 anchors are feasible (31.6 %); the cheapest pad
costs 1 199 250 EUR (anchor 444 208.5 E, 5 585 195.5 N, rotation 170 degrees); the median is 8.27 M EUR and the 95th percentile 8.28 M EUR, almost
equal, because most feasible ground is forest of one price (1 500 x 1.15 EUR/m2 over about 4 800 pixels). The legacy forest price (1 500 EUR/m2) and the
other base prices are the stand-ins that plan C6 replaces; until then the site term is not a result. The pad size 60 x 80 m is the legacy template, still to be sourced.

### Exact site-cost field, envelope window (2026-10-05, finished about 17:00 CEST)

The same script over `site_cost_envelope` (238 087 040 cells, 66.0 % forbidden). `site_field_envelope_cents.tif` (10.0 MB, sha256 e7595c138df7) and
`site_field_envelope_rotation.tif` (3.6 MB, sha256 4d75b960039a). **Independent check:** 400 feasible and 100 infeasible anchors, 0 disagreements, no cheaper
rotation. **Result:** 37 425 764 feasible anchors (15.7 %); cheapest pad 1 199 250 EUR (the same place as in the farm window, 444 210 E, 5 585 213 N, rotation 170
degrees); median 7.20 M EUR, 95th percentile 10.34 M EUR, maximum 13.28 M EUR. The run took 2 192 s on a machine that also ran other users' processes, so that time is
not a performance figure. The prices are the legacy stand-ins (plan C6 replaces them).

## 4. How the data is used, and what is not used yet

**Nothing downloaded or built on 2026-10-05 has entered any result yet.** The three rasters of section 3b exist,
but no cost field, incumbent or certificate has been computed from them. The uses below are planned and each waits
for what is named.

| layer | planned use | waits for |
|---|---|---|
| `alkis_land_use` | base trench-cost raster (built, `cost_base_1m`), site-cost classes, road and path classes for the access term and the road crossings | the site raster and the access raster (plan B1, B3) |
| `water_zones` | trench-cost multiplier or hard exclusion by zone (CIRED 2026 used 100, 2, 1.5, 1.2 for zones 1 to 4) | the owner's legal choice, hard or priced (plan Q12) |
| `soil_units` | trench-cost multiplier by soil unit | the rebuild |
| `nature_reserves`, `landscape_protection`, `natura2000_habitats`, `bird_protection_areas` | consent classes in the cost raster and in relax-and-verify (the CIRED 2026 script had the nature-reserve layer switched off) | plan Q12 |
| `biotopes_protected`, `biotope_complexes_protected` | consent class for legally protected biotopes | plan Q12 |
| `lines`, `towers_110`, `terminals`, `turbines` | HV connection and tap points, the grid connection points, the seven turbine nodes | the data loader into the window format and the incumbent designs (plan B2, E0) |
| `dem` | slope limit for access roads (`g_max`) and terrain for the tower fields | access raster (plan B3) and the tower-field build |

Still to fetch, with no service chosen yet: the regional plan (2010 version in force, 2025 draft as a
flagged second set) and the archaeology register. Still to build: the site raster (EUR per m2 of switchyard,
integer cents), the access raster and the roads layer (from the land-use classes). The turbine register (`turbine_evidence`) is a CSV and stays
`pending`. All of these are `pending` in the manifest.

## 5. Data that already existed (not downloaded on 2026-10-05)

From the CIRED 2026 study, in `TopoMILP/data/cired2026/`. These files were **recovered from a pyorps git
stash on 2026-09-04** (its README says so), so their file times are the recovery time, **not** the time the
data was first captured; the original capture dates are not recorded in this repository.

| file | content | size | file time | sha256 | planned use |
|---|---|---|---|---|---|
| `input/power_lines_110.geojson` | 110 kV network (OpenStreetMap extract) | 1.4 MB | 2026-09-04 23:15 | 61b4495c5ee3 | HV connection (`lines`) |
| `input/towers_110.geojson` | 4 977 110 kV towers (OpenStreetMap) | 9.0 MB | 2026-09-04 23:15 | 2210e54f1a72 | overhead tap points |
| `input/PCCs.shp` (+ sidecar files) | the three grid connection points, ids 1, 2, 0 | 184 B | 2026-09-04 23:34 | 39641f4eddbd | `terminals` (key by `id`, not by row) |
| `input/wea.geojson` | turbine positions of the case-study farm | 4.9 KB | 2026-09-04 23:15 | a20a675466d5 | turbine nodes |
| `input/HLNUG_Windenergieanlagen_epsg_31467.csv` | state turbine register | 0.44 MB | 2026-09-04 23:15 | 3e31182c6bf2 | `turbine_evidence` (cited, not committed) |
| `input/produkt_ff_stunde_20101001_20241231_06344.txt` | DWD hourly wind, station 06344, 2010-10-01 to 2024-12-31 | 5.4 MB | 2026-09-04 23:15 | 9bd252b3a56f | input of the loss table |
| `input/substation_sites_points.geojson` | six candidate sites of CIRED 2026 | 1.9 KB | 2026-09-04 23:15 | 28a724002fcf | comparison only |
| `raster/mod2_raster_wp.tiff` | the CIRED 2026 routing cost surface, 43 938 x 23 792 cells at 1 m, uint16 | 2.09 GB | 2026-09-05 18:17 | not computed (2 GB) | the surface the published CLCP costs were priced on; replaced by the rebuilt raster |

Derived and already used: `data/loss_categories.npz` (1.05 MB, file time 2026-09-24 12:35, sha256
c62b4266d1ad) was built from the DWD wind record by `scripts/build_loss_categories.py`; it feeds the loss weights of the
collector model (implementation log, C11).

**Frozen copies (2026-10-05, 15:03:37 to 15:03:38 CEST).** `scripts/register_existing_inputs_2026-10-05.py` copied
`power_lines_110.geojson`, `towers_110.geojson` and `wea.geojson` unchanged into `data/layers/`, checked each copy has the same
SHA-256 as its source, and registered them in the manifest as `lines`, `towers_110` and `turbines`. The three
connection points were converted from the shapefile to one GeoPackage (`data/layers/pccs.gpkg`, registered as
`terminals`) with the ids and the row order checked equal; the hashes of the source shapefile parts are in the
log. For all four the manifest's `captured` is the recovery date 2026-09-04 and `notes` says the original
capture date is unknown. The turbine file has no recorded licence or coordinate source: the owner states it.
A first log line gave it an OpenStreetMap licence by a default of the script; a correction line in the log
(15:04:01) withdraws that. The register CSV, the DWD wind file, the candidate-site file and the 2 GB CIRED 2026 raster
are not copied or registered.

## 6. Licences, and what they mean for the paper

- **Land use (ALKIS)**: the service states that any use is allowed without restriction or condition,
  commercial and non-commercial, including copying, editing, passing on to third parties, merging into new
  data sets and building into products; automated retrieval is free of charge (section 24 HVGG). If a source
  credit is given, it must say that the data were changed or processed (section 18 HVGG).
- **Drinking-water zones and soil units (HLNUG)**: free for private and commercial use, licence
  Creative Commons BY 4.0, with the source credit "Hessisches Landesamt fuer Naturschutz, Umwelt und Geologie
  (HLNUG)". For the water zones, showing them on an unassigned base map (ALK or TK25) is not allowed; the
  soil-unit text has no such sentence. Both say the service is not a legally binding source of information.
- **Nature reserves, landscape protection, Natura 2000 (HMLU)**: free for private use, for commercial use
  as long as the data are **not passed on, wholly or partly, for a fee**, and for planning purposes; making an
  own work from them is allowed. Source credit: the Hessian ministry (HMLU). The service says it is not a legally
  binding source of information.
- **DEM**: the pyorps service preset states dl-de/zero-2-0 (no restriction). Read from the preset, not
  re-read from the service.
- **Consequence:** by default the source layers are kept out of the repository and the paper (as the outline
  already says), because of the HMLU terms on passing data on for a fee; the ALKIS and HLNUG terms would allow
  sharing with the stated credit. The manifest (hashes, sources, times) and the cost rasters derived from the
  layers are published; the owner decides about the source files.

The full service texts are stored in the manifest (`licence`) and in the log (`licence_text`).

## 7. How to verify, and what to do on a re-fetch

- Verify every present file against its hash: `DataManifest.load(".../data_manifest.yaml").verify()`.
  The shipped test (`tests/test_siting/test_data_manifest.py`) does this when the files are on disk.
- A re-fetch returns whatever the service holds that day. The cadastre changes, so the hash will differ.
  Keep the old file, fetch to a new file name, add a new log line, and say in the manifest which one the
  results use.
- Never change a registered file in place: the hash check refuses it.

## 8. Limits of this record

- The seven events written first (five WFS layers, land use, DEM) were logged **after the fact**, from
  the file's modification time (the end of the download), its hash, the manifest, and the counts the scripts
  printed. Their log lines say `"reconstructed": true`. The DEM start time is the creation time of the
  script's output log. Events from the next fetch on are written by the scripts themselves.
- The exact request parameters of the pyorps WFS loader (paging, field names) are not logged; the layer, the
  mask and the bounding box are.
- The acquisition epoch (flight date) of the DEM and the reference dates of the HLNUG layers were not read.
- The terms of the three services were read from their capabilities documents on 2026-10-05, not from a
  separate licence page.

## 9. Use ledger (append a row whenever a layer enters a result)

| date and time | layers (sha256 prefix) | what was done | script and its hash | output file and its hash | by |
|---|---|---|---|---|---|
| | | | | | |
