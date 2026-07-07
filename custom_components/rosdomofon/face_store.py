"""
Хранилище эталонных лиц для интеграции Росдомофон.

Хранит по каждому человеку список эталонных фото: эмбеддинг (посчитанный
DeepFace), обрезанное фото лица и id (для поштучного удаления). Матчинг —
чистая функция косинусного расстояния без обращений к сети.

Формат хранения (Store):
    {
      "model": str,
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
        self._data: dict = {"model": None, "people": {}}

    async def async_load(self) -> None:
        """Загружает данные из Store, мигрируя старый формат при необходимости."""
        stored = await self._store.async_load()
        if stored:
            self._data = stored
            self._data.setdefault("people", {})
            self._data.setdefault("model", None)
            self._migrate()

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

        # Если сменили модель — старые эмбеддинги несовместимы, очищаем.
        if self._data.get("model") and self._data["model"] != model_name:
            _LOGGER.warning(
                "Сменилась модель распознавания (%s -> %s), эталоны сброшены",
                self._data["model"],
                model_name,
            )
            self._data["people"] = {}
        self._data["model"] = model_name

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
