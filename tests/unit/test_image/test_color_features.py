"""Histogram compatibility with OpenCV 4 and 5."""

import numpy as np
import pytest


cv2 = pytest.importorskip("cv2")

from lightautoml.image.image import ColorFeatures


@pytest.mark.parametrize("column_vector", [False, True])
def test_opencv_histogram_output_shapes(monkeypatch, column_vector):
    counts = np.array([1, 2, 1], dtype=np.float32)
    monkeypatch.setattr(cv2, "calcHist", lambda *args, **kwargs: counts[:, None] if column_vector else counts)
    actual = ColorFeatures(hist_size=3).compute_histogram(np.zeros((4, 4), dtype=np.uint8))
    np.testing.assert_allclose(actual, [0.25, 0.5, 0.25])


@pytest.mark.parametrize("value", [0, 128, 255])
def test_histogram_includes_all_uint8_values(value):
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    features = np.asarray(ColorFeatures(hist_size=8, is_hsv=False).get_features(image)).reshape(3, 8)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features.sum(axis=1), 1)
