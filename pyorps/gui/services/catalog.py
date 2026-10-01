"""
PYORPS GUI: OGC service catalog — GetCapabilities discovery + WMS/DEM loading.

Germany's geobasis data is open since June 2024 but split across 16 state
servers plus the nationwide BKG services (:data:`pyorps.gui.presets.MAP_SERVICES`),
and the land-use feature-type name differs per state. Rather than hard-code a
guess, the GUI can hit a WFS/WMS ``GetCapabilities`` and list the actual layers
(:func:`wfs_feature_types` / :func:`wms_layers`) so any server is usable.

``owslib`` isn't a dependency, so we parse the capabilities XML with the stdlib
(namespace-agnostic local-name matching — WFS 1.1/2.0 and WMS 1.1/1.3 all differ
in namespaces). WMS layers load as a live tile overlay; the BKG DGM digital
terrain model loads as a real elevation raster (WCS GetCoverage) usable for
slope/least-cost routing.
"""
from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET  # nosec B405  # nosemgrep - OGC service replies, entity expansion is limited by expat
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import urlopen

USER_AGENT = "pyorps-gui"


def _local(tag: str) -> str:
    """Strip the XML namespace from a tag: '{ns}Name' -> 'Name'."""
    return tag.rsplit("}", 1)[-1]


def _with_query(url: str, params: dict) -> str:
    """Append query params to a base URL, preserving any it already carries."""
    parts = urlparse(url)
    existing = parts.query
    query = urlencode(params)
    query = f"{existing}&{query}" if existing else query
    return urlunparse(parts._replace(query=query))


def _fetch(url: str, timeout: float) -> bytes:
    import ssl
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310  # nosec B310  # nosemgrep - http(s) URLs of configured services
            return response.read()
    except urllib.error.URLError as exc:
        # several German gov servers present a self-signed cert in the chain;
        # retry once without verification (last resort — these are public
        # open-data endpoints, not credentialed).
        if isinstance(getattr(exc, "reason", None), ssl.SSLError):
            insecure = ssl.create_default_context()
            insecure.check_hostname = False
            insecure.verify_mode = ssl.CERT_NONE
            with urlopen(request, timeout=timeout,  # noqa: S310  # nosec B310  # nosemgrep - http(s) URLs of configured services
                         context=insecure) as response:
                return response.read()
        raise


def _service_exception(root: ET.Element) -> str | None:
    """Return an OGC ServiceException message if the response is an error."""
    for elem in root.iter():
        if _local(elem.tag) in ("ServiceException", "ExceptionText",
                                "ExceptionReport"):
            text = (elem.text or "").strip()
            if text:
                return text
    return None


def wfs_feature_types(url: str, *, timeout: float = 30.0) -> list[dict]:
    """List a WFS's feature types: ``[{"name","title"}, ...]``.

    Tries WFS 2.0.0 GetCapabilities. Raises ValueError with a readable message
    when the server errors or returns no feature types.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    caps = _with_query(url, {"service": "WFS", "request": "GetCapabilities",
                             "version": "2.0.0"})
    try:
        raw = _fetch(caps, timeout)
    except Exception as exc:  # network / TLS / 404
        raise ValueError(
            f"Could not reach the WFS GetCapabilities of '{url}': {exc}. Check "
            "the URL, or that the server is up and reachable.") from exc
    try:
        root = ET.fromstring(raw)  # nosec B314  # nosemgrep - OGC service reply
    except ET.ParseError as exc:
        raise ValueError(
            f"The WFS at '{url}' did not return valid capabilities XML "
            f"({exc}).") from exc
    problem = _service_exception(root)
    if problem and not any(_local(e.tag) == "FeatureType" for e in root.iter()):
        raise ValueError(f"WFS server error: {problem}")

    types: list[dict] = []
    for feature in root.iter():
        if _local(feature.tag) != "FeatureType":
            continue
        name = title = ""
        for child in feature:
            if _local(child.tag) == "Name":
                name = (child.text or "").strip()
            elif _local(child.tag) == "Title":
                title = (child.text or "").strip()
        if name:
            types.append({"name": name, "title": title or name})
    if not types:
        raise ValueError(
            f"The WFS at '{url}' advertises no feature types — it may be a "
            "WMS/WCS or require a different version.")
    return types


def wms_layers(url: str, *, timeout: float = 30.0) -> list[dict]:
    """List a WMS's named layers: ``[{"name","title"}, ...]``."""
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    caps = _with_query(url, {"service": "WMS", "request": "GetCapabilities",
                             "version": "1.3.0"})
    raw = _fetch(caps, timeout)
    root = ET.fromstring(raw)  # nosec B314  # nosemgrep - service reply
    layers: list[dict] = []
    for layer in root.iter():
        if _local(layer.tag) != "Layer":
            continue
        name = title = ""
        for child in layer:
            if _local(child.tag) == "Name":
                name = (child.text or "").strip()
            elif _local(child.tag) == "Title":
                title = (child.text or "").strip()
        if name:
            layers.append({"name": name, "title": title or name})
    return layers


def wms_tile_url(url: str, layers: str) -> dict:
    """A WMS overlay descriptor for dash-leaflet's WMSTileLayer."""
    return {"base_url": url, "layers": layers, "format": "image/png",
            "transparent": True, "version": "1.3.0"}


def load_dem_raster(url: str, coverage_id: str, bounds, *,
                    axis_labels=("E", "N"), work_dir: str | Path = ".",
                    timeout: float = 120.0) -> str:
    """Fetch a DEM as GeoTIFF for ``bounds`` via WCS 2.0.1 GetCoverage.

    ``bounds`` is ``(minx, miny, maxx, maxy)`` in the coverage's NATIVE CRS
    (e.g. EPSG:25832 for BKG DGM200) with axis labels ``axis_labels`` (E/N).
    Returns the saved .tif path. Errors raise ValueError with guidance.
    """
    minx, miny, maxx, maxy = bounds
    ax_e, ax_n = axis_labels
    # Build the query by hand: many WCS servers reject percent-encoded SUBSET
    # parentheses/commas, so those must stay literal.
    query = (f"SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
             f"&COVERAGEID={coverage_id}&FORMAT=image/tiff"
             f"&SUBSET={ax_e}({minx},{maxx})&SUBSET={ax_n}({miny},{maxy})")
    parts = urlparse(url)
    full = urlunparse(parts._replace(
        query=(parts.query + "&" + query) if parts.query else query))
    try:
        raw = _fetch(full, timeout)
    except Exception as exc:
        raise ValueError(
            f"DEM download failed from '{url}': {exc}. Draw a study area first "
            "and check the DEM service is reachable.") from exc
    if raw[:2] not in (b"II", b"MM"):        # not a GeoTIFF (probably XML error)
        try:
            msg = _service_exception(ET.fromstring(raw)) or "unknown error"  # nosec B314  # nosemgrep - OGC service reply
        except ET.ParseError:
            msg = "the response was not a GeoTIFF"
        raise ValueError(f"DEM service did not return a raster: {msg}")
    out = Path(work_dir) / f"dem_{uuid.uuid4().hex[:8]}.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    return str(out)
