"""
Хранилище эталонных лиц для интеграции Росдомофон.

Хранит по каждому человеку список эталонных фото: эмбеддинг (посчитанный
DeepFace), обрезанное фото лица и id (для поштучного удаления). Матчинг —
чистая функция косинусного расстояния без обращений к сети.

Эмбеддинги зависят и от модели, и от детектора лиц (детектор задаёт
кадрирование и выравнивание лица перед вычислением эмбеддинга). Поэтому в
хранилище фиксируется и то, и другое: при смене любого из них сохранённые
эталоны несовместимы с живыми кадрами и сбрасываются (см. async_sync_config).

Формат хранения (Store):
    {
      "model": str,
      "detector": str,
      "people": {
        name: [
          {"id": hex, "embedding": [float, ...], "photo": base64_jpeg | None},
          ...
        ]
      }
    }

Старый формат (людям соответствовал список «голых» эмбеддингов) мигрируется при
загрузке.
"""

import base64
import logging
import math
import uuid

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from . import deepface_client, face_crop
from .const import FACE_STORE_KEY, FACE_STORE_VERSION

_LOGGER = logging.getLogger(__name__)


def cosine_distance(a: list[float], b: list[float]) -> float:
    """Косинусное расстояние (0 — идентичны, 2 — противоположны)."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return 2.0
    return 1.0 - dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class FaceStore:
    """Персистентное хранилище эталонных фото людей (эмбеддинг + фото)."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        self._store: Store = Store(hass, FACE_STORE_VERSION, FACE_STORE_KEY)
        self._data: dict = {"model": None, "detector": None, "people": {}}

    async def async_load(self) -> None:
        """Загружает данные из Store, мигрируя старый формат при необходимости."""
        stored = await self._store.async_load()
        if stored:
            self._data = stored
            self._data.setdefault("people", {})
            self._data.setdefault("model", None)
            self._data.setdefault("detector", None)
            self._migrate()

    async def async_sync_config(
        self, model_name: str, detector_backend: str
    ) -> str | None:
        """Сверяет модель/детектор с текущими настройками, сбрасывая эталоны при смене.

        Эмбеддинги эталонов зависят и от модели, и от детектора: сменился любой —
        сохранённые эмбеддинги больше несовместимы с живыми кадрами, иначе
        распознавание молча «плывёт». Возвращает "модель"/"детектор", если эталоны
        были сброшены, иначе None.

        Прежний детектор может быть неизвестен (миграция со старого формата без
        поля detector) — в этом случае эталоны НЕ трогаем, а просто фиксируем
        текущий детектор как эталонный (считаем, что фото сняты именно им).
        """
        people = self._data.setdefault("people", {})
        has_embeddings = any(entries for entries in people.values())
        stored_model = self._data.get("model")
        stored_detector = self._data.get("detector")

        changed: str | None = None
        if has_embeddings and stored_model and stored_model != model_name:
            changed = "модель"
        elif has_embeddings and stored_detector and stored_detector != detector_backend:
            changed = "детектор"

        dirty = False
        if changed:
            self._data["people"] = {}
            dirty = True
            _LOGGER.warning(
                "Сменился %s распознавания (было %s/%s, стало %s/%s) — "
                "эталоны сброшены, требуется пересоздать фото",
                changed,
                stored_model,
                stored_detector,
                model_name,
                detector_backend,
            )
        if stored_model != model_name or stored_detector != detector_backend:
            self._data["model"] = model_name
            self._data["detector"] = detector_backend
            dirty = True

        if dirty:
            await self._store.async_save(self._data)
        return changed

    def _migrate(self) -> None:
        """Приводит записи к новому формату (эмбеддинг -> запись с id и фото)."""
        people = self._data.get("people", {})
        changed = False
        for name, entries in people.items():
            migrated = []
            for entry in entries:
                if isinstance(entry, dict) and "embedding" in entry:
                    entry.setdefault("id", uuid.uuid4().hex)
                    entry.setdefault("photo", None)
                    migrated.append(entry)
                else:
                    # Старый формат: элемент — это сам эмбеддинг (список чисел).
                    migrated.append(
                        {"id": uuid.uuid4().hex, "embedding": entry, "photo": None}
                    )
                    changed = True
            people[name] = migrated
        if changed:
            _LOGGER.info("Хранилище лиц мигрировано в новый формат (эмбеддинг + фото)")

    @property
    def model(self) -> str | None:
        """Модель, которой посчитаны сохранённые эмбеддинги."""
        return self._data.get("model")

    @property
    def people(self) -> list[str]:
        """Имена загруженных людей."""
        return sorted(self._data.get("people", {}).keys())

    def photo_count(self, name: str) -> int:
        """Сколько фото сохранено для человека."""
        return len(self._data.get("people", {}).get(name, []))

    def photos(self, name: str) -> list[dict]:
        """Список фото человека: [{"id": hex, "photo": base64 | None}, ...]."""
        return [
            {"id": entry["id"], "photo": entry.get("photo")}
            for entry in self._data.get("people", {}).get(name, [])
        ]

    async def async_add_person(
        self,
        name: str,
        image: bytes,
        base_url: str,
        model_name: str,
        detector_backend: str,
    ) -> bytes:
        """Считает эмбеддинг лица с фото, обрезает лицо и добавляет человеку.

        Возвращает обрезанное фото лица (JPEG) — для показа. Поднимает
        deepface_client.DeepFaceError, если лицо не найдено или сервис недоступен.
        Anti-spoofing на эталонных фото не применяется.
        """
        faces = await self._hass.async_add_executor_job(
            deepface_client.represent_faces,
            base_url,
            image,
            model_name,
            detector_backend,
            False,  # anti_spoofing выключен для эталонных фото
        )
        if not faces:
            raise deepface_client.NoFaceError("На фото не найдено лицо")

        # Если лиц несколько — берём самое крупное (ближе к камере).
        face = max(faces, key=lambda f: _area_size(f.get("facial_area")))

        # Авто-обрезка лица с запасом — чище эталон, точнее распознавание.
        cropped = await self._hass.async_add_executor_job(
            face_crop.crop_face, image, face.get("facial_area")
        )

        # Если сменили модель или детектор — старые эмбеддинги несовместимы,
        # очищаем (обычно это уже сделал async_sync_config при перезагрузке).
        stored_model = self._data.get("model")
        stored_detector = self._data.get("detector")
        if (stored_model and stored_model != model_name) or (
            stored_detector and stored_detector != detector_backend
        ):
            _LOGGER.warning(
                "Сменились параметры распознавания (%s/%s -> %s/%s), эталоны сброшены",
                stored_model,
                stored_detector,
                model_name,
                detector_backend,
            )
            self._data["people"] = {}
        self._data["model"] = model_name
        self._data["detector"] = detector_backend

        people = self._data.setdefault("people", {})
        people.setdefault(name, []).append(
            {
                "id": uuid.uuid4().hex,
                "embedding": face["embedding"],
                "photo": base64.b64encode(cropped).decode("ascii"),
            }
        )
        await self._store.async_save(self._data)
        return cropped

    async def async_create_person(self, name: str) -> None:
        """Создаёт человека без фото (пустой список эталонов)."""
        people = self._data.setdefault("people", {})
        if name not in people:
            people[name] = []
            await self._store.async_save(self._data)

    async def async_remove_person(self, name: str) -> None:
        """Удаляет человека и все его фото."""
        people = self._data.setdefault("people", {})
        if name in people:
            del people[name]
            await self._store.async_save(self._data)

    async def async_remove_photo(self, name: str, photo_id: str) -> None:
        """Удаляет одно фото человека по id. Если фото не осталось — удаляет человека."""
        people = self._data.setdefault("people", {})
        entries = people.get(name)
        if not entries:
            return
        people[name] = [e for e in entries if e.get("id") != photo_id]
        if not people[name]:
            del people[name]
        await self._store.async_save(self._data)

    def match(
        self, embedding: list[float], threshold: float
    ) -> tuple[str, float] | None:
        """Находит ближайшего человека, если расстояние меньше порога.

        Возвращает (имя, расстояние) либо None, если совпадений нет.
        """
        best_name: str | None = None
        best_distance = threshold
        for name, entries in self._data.get("people", {}).items():
            for entry in entries:
                distance = cosine_distance(embedding, entry["embedding"])
                if distance < best_distance:
                    best_distance = distance
                    best_name = name
        if best_name is None:
            return None
        return best_name, best_distance

    def nearest(self, embeddings: list[list[float]]) -> tuple[str, float] | None:
        """Ближайший человек по всем лицам без учёта порога (для отладки)."""
        best: tuple[str, float] | None = None
        for embedding in embeddings:
            for name, entries in self._data.get("people", {}).items():
                for entry in entries:
                    distance = cosine_distance(embedding, entry["embedding"])
                    if best is None or distance < best[1]:
                        best = (name, distance)
        return best


def _area_size(facial_area: dict | None) -> int:
    """Площадь области лица (для выбора самого крупного лица)."""
    if not facial_area:
        return 0
    try:
        return int(facial_area["w"]) * int(facial_area["h"])
    except (KeyError, TypeError, ValueError):
        return 0
