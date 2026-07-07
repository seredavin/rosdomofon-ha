"""
Предварительный фильтр кадров перед вызовом DeepFace.

Двухступенчатый лёгкий фильтр, чтобы не гонять тяжёлое распознавание на каждом
кадре:
  1. Детекция движения (numpy/Pillow) — сравнение с предыдущим кадром камеры.
  2. Детекция лица в кадре (OpenCV Haar cascade).

DeepFace вызывается, только если оба этапа пройдены. Функции синхронные (CPU) —
вызывать из executor.
"""

import io
import logging

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Размер, до которого уменьшаем кадр для детекции движения (Ш×В)
_MOTION_SIZE = (160, 120)
# Яркостная разница пикселя, выше которой пиксель считается изменившимся
_PIXEL_DELTA = 25
# Доля изменившихся пикселей, выше которой считаем, что есть движение
_MOTION_FRACTION = 0.02


class FaceDetectUnavailable(Exception):
    """OpenCV недоступен — детекцию лица выполнить нельзя."""


# Ленивая загрузка Haar-каскада: держим один экземпляр на процесс.
_cascade = None
_cascade_loaded = False


def downscale_gray(image_bytes: bytes):
    """Уменьшает кадр и переводит в оттенки серого для детекции движения.

    Возвращает numpy-массив (int16) либо None, если Pillow недоступен или кадр не
    удалось декодировать (тогда детекция движения просто пропускается).
    """
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            small = img.convert("L").resize(_MOTION_SIZE)
            return np.asarray(small, dtype=np.int16)
    except (UnidentifiedImageError, OSError) as exc:
        _LOGGER.debug("Не удалось декодировать кадр для детекции движения: %s", exc)
        return None


def has_motion(prev_gray, cur_gray) -> bool:
    """True, если между кадрами достаточно изменившихся пикселей."""
    diff = np.abs(cur_gray - prev_gray)
    changed = int(np.count_nonzero(diff > _PIXEL_DELTA))
    return changed / diff.size > _MOTION_FRACTION


def _get_cascade():
    """Лениво загружает Haar-каскад лиц.

    Бросает FaceDetectUnavailable, если OpenCV не установлен или каскад не читается.
    """
    global _cascade, _cascade_loaded
    if _cascade_loaded:
        if _cascade is None:
            raise FaceDetectUnavailable("OpenCV недоступен")
        return _cascade

    _cascade_loaded = True
    try:
        import cv2
    except ImportError as exc:
        _cascade = None
        raise FaceDetectUnavailable(
            "Не установлен opencv-python-headless"
        ) from exc

    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        _cascade = None
        raise FaceDetectUnavailable("Не удалось загрузить каскад лиц")
    _cascade = cascade
    return _cascade


def has_face(image_bytes: bytes) -> bool:
    """True, если в кадре найдено хотя бы одно лицо (Haar cascade).

    Бросает FaceDetectUnavailable, если OpenCV недоступен. Если кадр не удалось
    декодировать — возвращает True (fail-open), чтобы не отбросить возможное лицо.
    """
    cascade = _get_cascade()  # бросит FaceDetectUnavailable без cv2
    import cv2  # здесь cv2 гарантированно доступен

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        _LOGGER.debug("OpenCV не смог декодировать кадр — пропускаю фильтр лица")
        return True
    img = cv2.equalizeHist(img)
    faces = cascade.detectMultiScale(
        img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    return len(faces) > 0
