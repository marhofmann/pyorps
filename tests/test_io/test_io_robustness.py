"""Regression tests for the audited defects of ``pyorps.io`` (audit T4, findings
#6, #10 (io half), #13, #23, #27 and #28).

Nothing here touches the network or a file of the repository: HTTP is mocked
the same way ``test_vector_loader.py`` does it, rasters are a few cells written
to ``tmp_path``, and the file-handle tests use stand-in objects that record
``close()``.
"""
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling
from shapely.geometry import Point, Polygon, box

from pyorps.core.exceptions import (
    WFSConnectionError,
    WFSError,
    WFSResponseParsingError,
    WFSServerError,
)
from pyorps.io import raster_loader
from pyorps.io.geo_dataset import (
    InMemoryRasterDataset,
    LocalRasterDataset,
    LocalVectorDataset,
    WFSVectorDataset,
    initialize_geo_dataset,
)
from pyorps.io.vector_loader import (
    _fetch_capabilities_xml,
    _fetch_wfs_data,
    _get_bbox_from_mask,
    _load_data_in_parallel,
    _parse_geojson_response,
    _parse_xml_response,
    _try_direct_load,
    _validate_wfs_url,
    load_from_wfs,
)

URL = "https://example.com/wfs"


@pytest.fixture(autouse=True)
def _clear_capabilities_cache():
    """GetCapabilities is cached per URL; the URL is the same in every test."""
    _fetch_capabilities_xml.cache_clear()
    yield
    _fetch_capabilities_xml.cache_clear()


def _http_response(status, body, content_type="application/json"):
    response = requests.Response()
    response.status_code = status
    response._content = body
    response.headers["Content-Type"] = content_type
    response.url = URL
    return response


def _points(bbox, n):
    """``n`` distinct points around the centre of ``bbox`` (EPSG:25832)."""
    minx, miny, maxx, maxy = bbox
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    return gpd.GeoDataFrame({"id": list(range(n))},
                            geometry=[Point(cx + i * 1e-4, cy) for i in range(n)],
                            crs="EPSG:25832")


# ---------------------------------------------------------------------------
# #6: a download that fails part-way must not return partial data silently
# ---------------------------------------------------------------------------
class TestPartialWFSDownload:

    @patch("pyorps.io.vector_loader._fetch_wfs_data")
    def test_dead_server_raises_with_counts_and_failed_request(self, mock_fetch):
        calls = []

        def fetch(url, layer, bbox, filter_params=None, srs=None):
            calls.append(bbox)
            if len(calls) == 1:
                return _points(bbox, 2)
            raise WFSServerError("overloaded")

        mock_fetch.side_effect = fetch
        with pytest.raises(WFSServerError) as caught:
            _load_data_in_parallel(URL, "layer1", (0, 0, 1000, 1000), max_workers=1)

        message = str(caught.value)
        assert "17 chunk requests failed in a row" in message
        assert "2 features from 1 successful chunk requests" in message
        assert f"chunk {calls[-1]}" in message          # which request failed last
        assert isinstance(caught.value.__cause__, WFSServerError)
        assert "overloaded" in str(caught.value.__cause__)

    @patch("pyorps.io.vector_loader.requests.get")
    def test_http_failures_are_not_reported_as_empty_chunks(self, mock_get):
        """A chunk whose requests all fail used to count as an empty chunk."""
        mock_get.side_effect = requests.ConnectionError("network unreachable")
        with pytest.raises(WFSConnectionError) as caught:
            _load_data_in_parallel(URL, "layer1", (0, 0, 1000, 1000), max_workers=1)

        assert "0 features from 0 successful chunk requests" in str(caught.value)
        # chain: WFSConnectionError <- WFSConnectionError (one chunk) <- requests
        chunk_error = caught.value.__cause__
        assert isinstance(chunk_error, WFSConnectionError)
        assert isinstance(chunk_error.__cause__, requests.ConnectionError)

    @patch("pyorps.io.vector_loader.MAX_CHUNK_DEPTH", 1)
    @patch("pyorps.io.vector_loader._fetch_wfs_data")
    def test_truncated_chunk_at_max_depth_raises(self, mock_fetch):
        """The server's feature limit still cuts the finest chunk: data is missing."""
        mock_fetch.side_effect = (
            lambda url, layer, bbox, filter_params=None, srs=None: _points(bbox, 1000))
        with pytest.raises(WFSError) as caught:
            _load_data_in_parallel(URL, "layer1", (0, 0, 1000, 1000), max_workers=1)

        assert type(caught.value) is WFSError
        message = str(caught.value)
        assert "returned exactly 1000 features" in message
        assert "maximum subdivision depth 1" in message
        assert "5000 features from 5 successful chunk requests" in message

    @patch("pyorps.io.vector_loader.MAX_CHUNK_DEPTH", 1)
    @patch("pyorps.io.vector_loader._fetch_wfs_data")
    def test_allow_partial_returns_data_and_warns_with_counts(self, mock_fetch):
        mock_fetch.side_effect = (
            lambda url, layer, bbox, filter_params=None, srs=None: _points(bbox, 1000))
        with pytest.warns(UserWarning, match=r"PARTIAL.*allow_partial=True") as record:
            result = _load_data_in_parallel(URL, "layer1", (0, 0, 1000, 1000),
                                            max_workers=1, allow_partial=True)

        assert isinstance(result, gpd.GeoDataFrame) and len(result) > 0
        message = next(str(w.message) for w in record if "PARTIAL" in str(w.message))
        assert "20000 features from 20 successful chunk requests" in message
        assert "16 request(s) left a hole" in message

    @patch("pyorps.io.vector_loader._fetch_wfs_data")
    def test_allow_partial_after_dead_server_keeps_what_was_received(self, mock_fetch):
        calls = []

        def fetch(url, layer, bbox, filter_params=None, srs=None):
            calls.append(bbox)
            if len(calls) == 1:
                return _points(bbox, 2)
            raise WFSServerError("overloaded")

        mock_fetch.side_effect = fetch
        with pytest.warns(UserWarning, match=r"2 features from 1 successful"):
            result = _load_data_in_parallel(URL, "layer1", (0, 0, 1000, 1000),
                                            max_workers=1, allow_partial=True)
        assert len(result) == 2

    @patch("pyorps.io.vector_loader._fetch_wfs_data")
    def test_complete_download_is_silent_and_unchanged(self, mock_fetch):
        """Failures that subdividing recovers are not holes: no raise, no warning."""
        def fetch(url, layer, bbox, filter_params=None, srs=None):
            if bbox[2] - bbox[0] > 5:
                raise WFSServerError("overloaded")
            return _points(bbox, 1)

        mock_fetch.side_effect = fetch
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _load_data_in_parallel(URL, "layer1", (0, 0, 16, 16), max_workers=1)
        assert len(result) == 16

    @patch("pyorps.io.vector_loader._resolve_layer", return_value="layer1")
    @patch("pyorps.io.vector_loader._load_data_in_parallel")
    def test_load_from_wfs_forwards_allow_partial_only_when_set(self, mock_parallel, _):
        mock_parallel.return_value = None
        load_from_wfs(URL, "layer1", bbox=(0, 0, 1, 1))
        assert "allow_partial" not in mock_parallel.call_args.kwargs
        load_from_wfs(URL, "layer1", bbox=(0, 0, 1, 1), allow_partial=True)
        assert mock_parallel.call_args.kwargs["allow_partial"] is True


# ---------------------------------------------------------------------------
# #27: helpers must not turn errors into None or decide on message text
# ---------------------------------------------------------------------------
class TestWFSHelperErrors:

    def test_private_address_guard_does_not_match_on_message_text(self):
        # old code: ip_address() fails for a hostname, its message contains the
        # hostname, and a hostname with "private" in it was re-raised as a hit
        _validate_wfs_url("https://private-reserved.example.org/wfs")
        _validate_wfs_url("https://example.com/wfs")

    @pytest.mark.parametrize("host", ["127.0.0.1", "10.1.2.3", "192.168.0.7", "[::1]"])
    def test_private_address_guard_still_blocks_literal_ips(self, host):
        with pytest.raises(ValueError, match="private/reserved IP address"):
            _validate_wfs_url(f"http://{host}/wfs")

    @patch("pyorps.io.vector_loader.requests.get")
    def test_direct_load_failing_everywhere_raises_with_cause(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("network unreachable")
        with pytest.raises(WFSConnectionError) as caught:
            _try_direct_load(URL, "layer1")

        assert mock_get.call_count == 3                      # all three WFS versions tried
        assert "layer 'layer1'" in str(caught.value)
        assert isinstance(caught.value.__cause__, requests.ConnectionError)

    @patch("pyorps.io.vector_loader.requests.get")
    def test_direct_load_empty_answer_after_a_failed_version_is_not_an_error(self, mock_get):
        empty = _http_response(200, b'{"features": []}')
        mock_get.side_effect = [requests.ConnectionError("blip"), empty, empty]
        assert _try_direct_load(URL, "layer1") == (None, False)

    @patch("pyorps.io.vector_loader._get_extent_from_capabilities", return_value=(0, 0, 10, 10))
    @patch("pyorps.io.vector_loader._load_data_in_parallel")
    @patch("pyorps.io.vector_loader.requests.get")
    def test_load_from_wfs_falls_back_to_chunks_and_says_why(
            self, mock_get, mock_parallel, _extent, caplog):
        mock_get.side_effect = requests.ConnectionError("unbounded request refused")
        chunked = _points((0, 0, 10, 10), 3)
        mock_parallel.return_value = chunked
        with caplog.at_level("WARNING", logger="pyorps.io.vector_loader"):
            result = load_from_wfs(URL, "layer1", auto_match=False)

        assert result is chunked
        assert "falling back to chunked loading" in caplog.text
        assert "unbounded request refused" in caplog.text

    @patch("pyorps.io.vector_loader._get_extent_from_capabilities", return_value=None)
    @patch("pyorps.io.vector_loader.requests.get")
    def test_load_from_wfs_without_extent_chains_the_direct_failure(self, mock_get, _extent):
        mock_get.side_effect = requests.ConnectionError("network unreachable")
        with pytest.raises(WFSError, match="direct request failed first") as caught:
            load_from_wfs(URL, "layer1", auto_match=False)
        assert isinstance(caught.value.__cause__, WFSConnectionError)

    def test_non_json_body_under_json_content_type_is_a_parsing_error(self):
        response = _http_response(200, b"<html><body>Service Unavailable</body></html>")
        with pytest.raises(WFSResponseParsingError, match="Service Unavailable") as caught:
            _parse_geojson_response(response)
        assert isinstance(caught.value.__cause__, ValueError)

    def test_geojson_without_features_is_still_an_empty_answer(self):
        assert _parse_geojson_response(_http_response(200, b'{"features": []}')) is None
        assert _parse_geojson_response(_http_response(200, b'{"type": "FeatureCollection"}')) is None

    @patch("pyorps.io.vector_loader.gpd.read_file", side_effect=OSError("disk full"))
    def test_unreadable_temp_file_is_a_parsing_error(self, _read_file):
        with pytest.raises(WFSResponseParsingError, match="disk full") as caught:
            _parse_xml_response(_http_response(200, b"<x/>", "text/xml"))
        assert isinstance(caught.value.__cause__, OSError)

    def test_gml_without_features_is_still_an_empty_answer(self):
        """GDAL raises IndexError for a FeatureCollection with zero members."""
        body = (b'<?xml version="1.0"?><wfs:FeatureCollection '
                b'xmlns:wfs="http://www.opengis.net/wfs/2.0" numberMatched="0" '
                b'numberReturned="0"/>')
        assert _parse_xml_response(_http_response(200, body, "text/xml")) is None

    @patch("pyorps.io.vector_loader.requests.get")
    def test_chunk_fetch_failing_everywhere_is_not_none(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("network unreachable")
        with pytest.raises(WFSConnectionError) as caught:
            _fetch_wfs_data(URL, "layer1", (0, 0, 10, 10))
        assert isinstance(caught.value.__cause__, requests.ConnectionError)

    @patch("pyorps.io.vector_loader.requests.get")
    def test_chunk_fetch_with_html_error_page_is_a_parsing_error(self, mock_get):
        mock_get.return_value = _http_response(200, b"<html>gateway</html>")
        with pytest.raises(WFSResponseParsingError, match="gateway"):
            _fetch_wfs_data(URL, "layer1", (0, 0, 10, 10))


# ---------------------------------------------------------------------------
# #13: WFSVectorDataset request options, shapely masks, empty results
# ---------------------------------------------------------------------------
class TestWFSVectorDataset:
    SOURCE = {"url": URL, "layer": "layer1"}

    @staticmethod
    def _fake_loader(seen, result=None):
        def fake(**kwargs):
            seen.update(kwargs)
            return result
        return fake

    def test_shapely_mask_gives_the_request_bbox_and_clips(self):
        seen = {}
        features = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[Point(1, 1), Point(50, 50)],
                                    crs="EPSG:25832")
        with patch("pyorps.io.geo_dataset.load_from_wfs",
                   self._fake_loader(seen, features)):
            dataset = WFSVectorDataset(self.SOURCE, mask=box(0, 0, 10, 10))
            dataset.load_data()

        assert seen["bbox"] == (0.0, 0.0, 10.0, 10.0)
        assert list(dataset.data["id"]) == [1]               # clipped to the mask

    def test_defaults_reproduce_the_request_made_before(self):
        seen = {}
        features = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(1, 1)], crs="EPSG:25832")
        with patch("pyorps.io.geo_dataset.load_from_wfs",
                   self._fake_loader(seen, features)):
            WFSVectorDataset(self.SOURCE, bbox=(0, 0, 10, 10)).load_data()

        assert seen == {
            "url": URL, "layer": "layer1", "bbox": (0, 0, 10, 10), "filter_params": None,
            "auto_match": True, "crs": None, "max_workers": 4, "allow_partial": False,
        }

    def test_request_options_reach_load_from_wfs_and_load_data_overrides(self):
        seen = {}
        features = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(1, 1)], crs="EPSG:4326")
        with patch("pyorps.io.geo_dataset.load_from_wfs",
                   self._fake_loader(seen, features)):
            dataset = WFSVectorDataset(self.SOURCE, bbox=(0, 0, 10, 10),
                                       request_crs="EPSG:4326", max_workers=2,
                                       allow_partial=True)
            dataset.load_data()
            assert (seen["crs"], seen["max_workers"], seen["allow_partial"]) == (
                "EPSG:4326", 2, True)
            dataset.load_data(request_crs="EPSG:3857", max_workers=1, allow_partial=False)
            assert (seen["crs"], seen["max_workers"], seen["allow_partial"]) == (
                "EPSG:3857", 1, False)

    @patch("pyorps.io.vector_loader.requests.get")
    def test_request_crs_labels_the_http_request(self, mock_get):
        mock_get.return_value = _http_response(
            200, b'{"features": [{"type": "Feature", "properties": {"id": 1}, '
                 b'"geometry": {"type": "Point", "coordinates": [1, 1]}}]}')

        WFSVectorDataset(self.SOURCE, bbox=(0, 0, 10, 10), request_crs="EPSG:4326",
                         ).load_data(auto_match=False)
        for call in mock_get.call_args_list:
            assert call.kwargs["params"]["SRSNAME"] == "EPSG:4326"
            assert call.kwargs["params"]["BBOX"].endswith("EPSG:4326")

        mock_get.reset_mock()
        WFSVectorDataset(self.SOURCE, bbox=(0, 0, 10, 10)).load_data(auto_match=False)
        for call in mock_get.call_args_list:                  # today's request
            assert call.kwargs["params"]["SRSNAME"] == "EPSG:25832"

    def test_shapely_bbox_is_accepted_like_a_tuple(self):
        seen = {}
        features = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(1, 1)], crs="EPSG:25832")
        with patch("pyorps.io.geo_dataset.load_from_wfs",
                   self._fake_loader(seen, features)):
            WFSVectorDataset(self.SOURCE, bbox=Polygon([(0, 0), (4, 0), (4, 3)])).load_data()
        assert seen["bbox"] == (0.0, 0.0, 4.0, 3.0)

    def test_geodataframe_and_geoseries_masks_give_a_plain_tuple(self):
        mask = gpd.GeoDataFrame(geometry=[box(1, 2, 3, 4)], crs="EPSG:25832")
        assert _get_bbox_from_mask(mask) == (1.0, 2.0, 3.0, 4.0)
        assert _get_bbox_from_mask(mask.geometry) == (1.0, 2.0, 3.0, 4.0)

    def test_empty_result_raises_a_documented_error_not_attribute_error(self):
        seen = {}
        with patch("pyorps.io.geo_dataset.load_from_wfs", self._fake_loader(seen, None)):
            dataset = WFSVectorDataset(self.SOURCE, bbox=(0, 0, 10, 10))
            with pytest.raises(WFSResponseParsingError) as caught:
                dataset.load_data()

        assert isinstance(caught.value, WFSError)
        message = str(caught.value)
        assert "no data for layer 'layer1'" in message and URL in message
        assert "(0, 0, 10, 10)" in message
        assert dataset.data is None


# ---------------------------------------------------------------------------
# #23: dataset factory: os.PathLike, the tuple form, error messages
# ---------------------------------------------------------------------------
def _write_tif(path, crs="EPSG:32632", value=1):
    data = np.full((1, 3, 3), value, dtype="uint16")
    with rasterio.open(path, "w", driver="GTiff", height=3, width=3, count=1,
                       dtype="uint16", crs=crs,
                       transform=from_origin(500000, 5600000, 1, 1)) as dst:
        dst.write(data)
    return Path(path)


class TestDatasetFactory:
    TRANSFORM = Affine(1, 0, 500000, 0, -1, 5600000)

    def test_pathlib_paths_are_accepted(self, tmp_path):
        tif = _write_tif(tmp_path / "cells.tif")
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(1, 1)], crs="EPSG:25832")
        geojson = tmp_path / "points.geojson"
        gdf.to_file(geojson, driver="GeoJSON")

        raster = initialize_geo_dataset(tif)
        vector = initialize_geo_dataset(geojson)

        assert isinstance(raster, LocalRasterDataset)
        assert raster.file_source == str(tif)
        assert isinstance(vector, LocalVectorDataset)
        vector.load_data()
        assert len(vector.data) == 1

    def test_missing_pathlib_path_is_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            initialize_geo_dataset(tmp_path / "missing.tif")

    def test_documented_tuple_form(self):
        array = np.arange(9, dtype="uint16").reshape(3, 3)
        dataset = initialize_geo_dataset((array, "EPSG:32632", self.TRANSFORM))

        assert isinstance(dataset, InMemoryRasterDataset)
        assert dataset.crs == "EPSG:32632" and dataset.transform == self.TRANSFORM
        np.testing.assert_array_equal(dataset.data, array)

    def test_tuple_form_explicit_arguments_override_the_tuple(self):
        array = np.zeros((3, 3), dtype="uint16")
        other = Affine(2, 0, 0, 0, -2, 0)
        dataset = initialize_geo_dataset((array, "EPSG:32632", self.TRANSFORM),
                                         crs="EPSG:25832", transform=other)
        assert dataset.crs == "EPSG:25832" and dataset.transform == other

    def test_existing_dataset_is_returned_as_it_is(self):
        dataset = InMemoryRasterDataset(np.zeros((3, 3)), "EPSG:32632", self.TRANSFORM)
        assert initialize_geo_dataset(dataset) is dataset

    def test_error_names_the_type_and_lists_the_forms_without_dumping_data(self):
        class Payload:
            def __repr__(self):
                return "SECRET_PAYLOAD" * 50

        with pytest.raises(ValueError) as caught:
            initialize_geo_dataset(Payload())

        message = str(caught.value)
        assert "Payload" in message
        assert "SECRET_PAYLOAD" not in message
        for form in (".tif", ".geojson", "os.PathLike", "GeoDataFrame", "numpy array",
                     "(data_array, crs, transform) tuple", "'url' and 'layer'", "GeoDataset"):
            assert form in message

    def test_wrong_tuple_does_not_dump_the_array(self):
        data = np.full((3, 3), 123456.789)
        with pytest.raises(ValueError) as caught:
            initialize_geo_dataset((data, "EPSG:32632"))
        message = str(caught.value)
        assert "tuple of length 2" in message
        assert "123456" not in message

    def test_unknown_extension_lists_the_supported_ones(self, tmp_path):
        asc = tmp_path / "grid.asc"
        asc.write_text("ncols 1\n")
        with pytest.raises(ValueError) as caught:
            initialize_geo_dataset(asc)
        message = str(caught.value)
        assert "'.asc'" in message and ".tif" in message and ".gpkg" in message


# ---------------------------------------------------------------------------
# #10 (io half): options that cannot apply to a raster are not dropped silently
# ---------------------------------------------------------------------------
class TestRasterOptions:
    TRANSFORM = Affine(1, 0, 500000, 0, -1, 5600000)

    def test_bbox_and_mask_for_an_array_are_reported(self):
        array = np.zeros((3, 3), dtype="uint16")
        with pytest.warns(UserWarning, match="bbox and mask ignored for a raster"):
            initialize_geo_dataset(array, crs="EPSG:32632", bbox=(0, 0, 1, 1),
                                   mask=box(0, 0, 1, 1), transform=self.TRANSFORM)
        with pytest.warns(UserWarning, match="bbox ignored for a raster"):
            initialize_geo_dataset(array, bbox=(0, 0, 1, 1), transform=self.TRANSFORM)

    def test_no_warning_without_bbox_and_mask_or_for_vector_sources(self):
        array = np.zeros((3, 3), dtype="uint16")
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:25832")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            initialize_geo_dataset(array, transform=self.TRANSFORM)
            initialize_geo_dataset(gdf, bbox=(-1, -1, 1, 1), mask=box(-1, -1, 1, 1))

    def test_requested_crs_that_differs_from_the_file_is_reported_once(self, tmp_path):
        tif = _write_tif(tmp_path / "cells.tif", crs="EPSG:32632")
        dataset = initialize_geo_dataset(str(tif), crs="EPSG:25832")

        with pytest.warns(UserWarning, match="Raster files are not reprojected"):
            dataset.load_metadata()
        assert dataset.crs == CRS.from_epsg(32632)           # the file's CRS wins
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dataset.load_data()                               # already reported
        assert dataset.crs == CRS.from_epsg(32632)

    @pytest.mark.parametrize("requested", ["EPSG:32632", "epsg:32632", CRS.from_epsg(32632)])
    def test_requested_crs_equal_to_the_file_is_silent(self, tmp_path, requested):
        tif = _write_tif(tmp_path / "cells.tif", crs="EPSG:32632")
        dataset = initialize_geo_dataset(str(tif), crs=requested)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dataset.load_metadata()
            dataset.load_data()


# ---------------------------------------------------------------------------
# #28: raster_loader: handles are closed on failure, merge settings reachable
# ---------------------------------------------------------------------------
UTM32 = CRS.from_epsg(32632)


class _Handle:
    """Stands in for a rasterio dataset or a MemoryFile and records ``close()``."""

    def __init__(self, crs=UTM32, fail_open=False):
        self.crs = crs
        self.dtypes = ("uint16",)
        self.nodata = None
        self.profile = {"driver": "GTiff", "height": 2, "width": 2, "count": 1,
                        "dtype": "uint16", "crs": crs}
        self.close_calls = 0
        self.fail_open = fail_open

    def close(self):
        self.close_calls += 1

    @property
    def closed(self):
        return self.close_calls > 0

    def read(self):
        return np.zeros((1, 2, 2), dtype="uint16")

    # MemoryFile side of the stand-in
    def open(self, **profile):
        if self.fail_open:
            raise RuntimeError("cannot create the in-memory copy")
        return _Handle(profile["crs"])

    def write(self, data):
        pass


def _dummy_files(tmp_path, n):
    paths = []
    for i in range(n):
        path = tmp_path / f"tile_{i}.tif"
        path.write_bytes(b"")                    # only has to exist: rasterio.open is mocked
        paths.append(path)
    return paths


@pytest.fixture
def opened(monkeypatch):
    """Patch ``rasterio.open`` to hand out the given handles in order."""
    def install(*handles):
        queue = list(handles)
        monkeypatch.setattr(rasterio, "open", lambda *a, **k: queue.pop(0))
        return list(handles)
    return install


@pytest.fixture
def fake_merge(monkeypatch):
    merged = (np.zeros((1, 2, 2), dtype="uint16"), Affine.identity())
    mock = MagicMock(return_value=merged)
    monkeypatch.setattr(raster_loader, "merge", mock)
    return mock


class TestRasterLoaderHandles:

    def test_read_raster_files_closes_everything_when_a_later_file_fails(
            self, tmp_path, opened):
        first, second, no_crs = _Handle(), _Handle(), _Handle(crs=None)
        opened(first, second, no_crs)

        with pytest.raises(ValueError, match="no CRS defined"):
            raster_loader.read_raster_files(_dummy_files(tmp_path, 3))

        assert first.closed and second.closed and no_crs.closed

    def test_read_raster_files_closes_everything_when_the_in_memory_copy_fails(
            self, tmp_path, opened, monkeypatch):
        first, no_crs = _Handle(), _Handle(crs=None)
        memfile = _Handle(fail_open=True)
        opened(first, no_crs)
        monkeypatch.setattr(raster_loader, "MemoryFile", lambda: memfile)

        with pytest.raises(RuntimeError, match="in-memory copy"):
            raster_loader.read_raster_files(_dummy_files(tmp_path, 2),
                                            default_crs="EPSG:32632")

        assert first.closed and no_crs.closed and memfile.closed

    def test_read_raster_files_success_hands_the_open_handles_to_the_caller(
            self, tmp_path, opened):
        first, second = _Handle(), _Handle()
        opened(first, second)

        datasets, memory_files = raster_loader.read_raster_files(_dummy_files(tmp_path, 2))

        assert datasets == [first, second] and memory_files == []
        assert not first.closed and not second.closed

    def test_combine_rasters_closes_the_datasets_on_success(self, fake_merge):
        first, second = _Handle(), _Handle()
        memfile = _Handle()

        raster_loader.combine_rasters(([first, second], [memfile]))

        assert first.closed and second.closed and memfile.closed

    def test_combine_rasters_closes_the_datasets_when_merge_fails(self, fake_merge):
        first, second = _Handle(), _Handle()
        memfile = _Handle()
        fake_merge.side_effect = RuntimeError("merge failed")

        with pytest.raises(RuntimeError, match="merge failed"):
            raster_loader.combine_rasters(([first, second], [memfile]))

        assert first.closed and second.closed and memfile.closed

    def test_combine_rasters_closes_the_datasets_when_saving_fails(
            self, fake_merge, tmp_path):
        first = _Handle()
        blocker = tmp_path / "file_not_dir"
        blocker.write_text("x")                       # parent of save_path is a file

        with pytest.raises(OSError):
            raster_loader.combine_rasters([first], save_path=blocker / "out.tif")

        assert first.closed

    def test_combine_rasters_closes_the_datasets_on_a_crs_mismatch(self, fake_merge):
        first, second = _Handle(), _Handle(crs=CRS.from_epsg(4326))

        with pytest.raises(ValueError, match="same CRS"):
            raster_loader.combine_rasters([first, second])

        assert first.closed and second.closed
        fake_merge.assert_not_called()

    def test_create_combined_raster_dataset_closes_on_both_paths(
            self, tmp_path, opened, fake_merge):
        first, second = _Handle(), _Handle()
        opened(first, second)
        result = raster_loader.create_combined_raster_dataset(_dummy_files(tmp_path, 2))
        assert isinstance(result, InMemoryRasterDataset)
        assert first.closed and second.closed

        third, fourth = _Handle(), _Handle()
        opened(third, fourth)
        fake_merge.side_effect = RuntimeError("merge failed")
        with pytest.raises(RuntimeError, match="merge failed"):
            raster_loader.create_combined_raster_dataset(_dummy_files(tmp_path, 2))
        assert third.closed and fourth.closed


class TestRasterLoaderMergeSettings:

    @staticmethod
    def _tiles(tmp_path):
        left = _write_tif(tmp_path / "left.tif", value=7)
        right = tmp_path / "right.tif"
        with rasterio.open(right, "w", driver="GTiff", height=3, width=3, count=1,
                           dtype="uint16", crs="EPSG:32632",
                           transform=from_origin(500003, 5600000, 1, 1)) as dst:
            dst.write(np.full((1, 3, 3), 9, dtype="uint16"))
        return [left, right]

    def test_nodata_dtype_and_resampling_reach_merge(self, tmp_path):
        with patch.object(raster_loader, "merge", wraps=raster_loader.merge) as spy:
            result = raster_loader.create_combined_raster_dataset(
                self._tiles(tmp_path), nodata=0, dtype="uint8",
                resampling=Resampling.nearest)

        kwargs = spy.call_args.kwargs
        assert kwargs["nodata"] == 0
        assert kwargs["dtype"] == "uint8"
        assert kwargs["resampling"] == Resampling.nearest
        assert result.data.dtype == np.uint8
        assert result.data.shape == (1, 3, 6)

    def test_defaults_equal_the_behaviour_before(self, tmp_path):
        with patch.object(raster_loader, "merge", wraps=raster_loader.merge) as spy:
            result = raster_loader.create_combined_raster_dataset(self._tiles(tmp_path))

        kwargs = spy.call_args.kwargs
        assert kwargs["nodata"] is None
        assert kwargs["resampling"] == Resampling.bilinear
        assert kwargs["method"] == "first"
        assert result.data.dtype == np.uint16
        assert sorted(np.unique(result.data)) == [7, 9]
