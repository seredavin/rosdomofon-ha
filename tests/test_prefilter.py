"""Тесты предварительного фильтра кадров (движение + лицо)."""

import io

import numpy as np
import pytest

from custom_components.rosdomofon import prefilter


def _jpeg(color: int, size=(320, 240)) -> bytes:
    """Возвращает JPEG-кадр заданной равномерной яркости (требует Pillow)."""
    from PIL import Image

    img = Image.new("L", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# --- Детекция движения --------------------------------------------------------


def test_downscale_gray_shape():
    pytest.importorskip("PIL")
    arr = prefilter.downscale_gray(_jpeg(128))
    assert arr is not None
    assert arr.shape == (120, 160)  # (высота, ширина)


def test_downscale_gray_invalid_returns_none():
    pytest.importorskip("PIL")
    assert prefilter.downscale_gray(b"not-an-image") is None


def test_has_motion_true_on_big_change():
    dark = np.zeros((120, 160), dtype=np.int16)
    bright = np.full((120, 160), 255, dtype=np.int16)
    assert prefilter.has_motion(dark, bright) is True


def test_has_motion_false_on_identical():
    frame = np.full((120, 160), 100, dtype=np.int16)
    assert prefilter.has_motion(frame, frame.copy()) is False


def test_has_motion_false_on_tiny_noise():
    rng = np.zeros((120, 160), dtype=np.int16)
    noisy = rng.copy()
    # Меняем всего несколько пикселей — ниже порога доли
    noisy[0, :5] = 255
    assert prefilter.has_motion(rng, noisy) is False


# --- Детекция лица (OpenCV) ---------------------------------------------------


def test_has_face_false_on_blank():
    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    prefilter._cascade = None
    prefilter._cascade_loaded = False
    # Пустой серый кадр — лиц нет
    assert prefilter.has_face(_jpeg(120)) is False


def test_has_face_fail_open_on_garbage():
    pytest.importorskip("cv2")
    prefilter._cascade = None
    prefilter._cascade_loaded = False
    # Нечитаемый кадр -> fail-open (True), чтобы не потерять лицо
    assert prefilter.has_face(b"not-an-image") is True


def test_has_face_unavailable_without_cv2(monkeypatch):
    """Без OpenCV has_face должен бросать FaceDetectUnavailable."""
    pytest.importorskip("PIL")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("no cv2")
        return real_import(name, *args, **kwargs)

    prefilter._cascade = None
    prefilter._cascade_loaded = False
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(prefilter.FaceDetectUnavailable):
        prefilter.has_face(_jpeg(120))
    # Сбрасываем кэш, чтобы не влиять на другие тесты
    prefilter._cascade = None
    prefilter._cascade_loaded = False
