"""
Обрезка лица по области, найденной DeepFace.

Из полного кадра вырезает лицо с запасом (margin) и уменьшает до разумного
размера — так эталонное фото чище и легче, а распознавание точнее. Использует
Pillow (в Home Assistant есть штатно). Функция синхронная — вызывать из executor.
"""

import io
import logging

_LOGGER = logging.getLogger(__name__)

# Запас вокруг лица (доля от размера лица) и максимальная сторона эскиза
_DEFAULT_MARGIN = 0.4
_MAX_SIDE = 400


def crop_face(
    image: bytes,
    facial_area: dict | None,
    margin: float = _DEFAULT_MARGIN,
    max_side: int = _MAX_SIDE,
) -> bytes:
    """Обрезает изображение по области лица с запасом.

    Если область не задана или некорректна — возвращает исходное изображение,
    уменьшенное до max_side (fail-open, чтобы всегда было что показать/сохранить).
    """
    try:
        from PIL import Image
    except ImportError:
        return image

    try:
        with Image.open(io.BytesIO(image)) as img:
            img = img.convert("RGB")
            width, height = img.size

            box = _area_to_box(facial_area, width, height, margin)
            crop = img.crop(box) if box else img
            crop.thumbnail((max_side, max_side))

            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
    except (OSError, ValueError) as exc:
        _LOGGER.debug("Не удалось обрезать лицо: %s", exc)
        return image


def _area_to_box(
    facial_area: dict | None, width: int, height: int, margin: float
) -> tuple[int, int, int, int] | None:
    """Преобразует область лица {x,y,w,h} в рамку обрезки с запасом."""
    if not facial_area:
        return None
    try:
        x = int(facial_area["x"])
        y = int(facial_area["y"])
        w = int(facial_area["w"])
        h = int(facial_area["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None

    mx = int(w * margin)
    my = int(h * margin)
    left = max(0, x - mx)
    top = max(0, y - my)
    right = min(width, x + w + mx)
    bottom = min(height, y + h + my)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
