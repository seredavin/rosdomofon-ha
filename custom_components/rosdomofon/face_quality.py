"""
Оценка качества кадра лица перед распознаванием.

С плохой камеры выгоднее распознавать не каждый кадр, а только пригодные:
достаточно крупное, резкое и уверенно детектированное лицо. Мутные, мелкие и
«зашумлённые» кадры дают неустойчивые эмбеддинги и ложные срабатывания.

Метрики (все считаются по области лица, а не по всему кадру):
  - sharpness — дисперсия отклика Лапласиана (чем выше, тем резче кадр);
  - min_side — меньшая сторона лица в пикселях (размер лица в кадре).

Функция синхронная (CPU + Pillow) — вызывать из executor. При недоступности
Pillow или ошибке декодирования возвращает None-метрики (fail-open — тогда
качество-фильтр такой кадр не отбрасывает).
"""

import io
import logging

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Ядро Лапласиана 3x3 — отклик тем сильнее, чем больше высокочастотных деталей
# (резких границ). Дисперсия отклика — классическая мера чёткости кадра.
_LAPLACIAN_KERNEL = (0, 1, 0, 1, -4, 1, 0, 1, 0)


def assess(
    image_bytes: bytes, facial_area: dict | None
) -> tuple[float | None, int | None]:
    """Возвращает (sharpness, min_side) для области лица.

    sharpness — дисперсия лапласиана (мера резкости), min_side — меньшая сторона
    лица в пикселях. None означает, что метрику посчитать не удалось (fail-open).
    """
    try:
        from PIL import Image, ImageFilter, UnidentifiedImageError
    except ImportError:
        return None, None
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            gray = img.convert("L")
            crop, min_side = _crop_face(gray, facial_area)
            lap = crop.filter(
                ImageFilter.Kernel((3, 3), _LAPLACIAN_KERNEL, scale=1)
            )
            sharpness = float(np.asarray(lap, dtype=np.float64).var())
            return sharpness, min_side
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        _LOGGER.debug("Не удалось оценить качество кадра: %s", exc)
        return None, None


def _crop_face(gray, facial_area: dict | None):
    """Вырезает область лица; возвращает (crop, min_side_px).

    Без валидной области лица метрики считаются по всему кадру.
    """
    width, height = gray.size
    if facial_area:
        try:
            x = max(0, int(facial_area["x"]))
            y = max(0, int(facial_area["y"]))
            fw = int(facial_area["w"])
            fh = int(facial_area["h"])
        except (KeyError, TypeError, ValueError):
            return gray, min(width, height)
        if fw > 0 and fh > 0:
            box = (x, y, min(width, x + fw), min(height, y + fh))
            return gray.crop(box), min(fw, fh)
    return gray, min(width, height)
