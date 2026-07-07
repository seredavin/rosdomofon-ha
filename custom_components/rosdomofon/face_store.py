"""
Хранилище эталонных лиц для интеграции Росдомофон.

Хранит эмбеддинги «своих» людей (посчитанные DeepFace при загрузке фото) в
персистентном Store и умеет находить ближайшее совпадение по косинусному
расстоянию. Матчинг — чистая функция без обращений к сети.
"""

import logging
import math

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from . import deepface_client
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
    """Персистентное хранилище эталонных эмбеддингов людей."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        self._store: Store = Store(hass, FACE_STORE_VERSION, FACE_STORE_KEY)
        # {"model": str, "people": {name: [embedding, ...]}}
        self._data: dict = {"model": None, "people": {}}

    async def async_load(self) -> None:
        """Загружает данные из Store."""
        stored = await self._store.async_load()
        if stored:
            self._data = stored
            self._data.setdefault("people", {})
            self._data.setdefault("model", None)

    @property
    def model(self) -> str | None:
        """Модель, которой посчитаны сохранённые эмбеддинги."""
        return self._data.get("model")

    @property
    def people(self) -> list[str]:
        """Имена загруженных людей."""
        return sorted(self._data.get("people", {}).keys())

    def photo_count(self, name: str) -> int:
        """Сколько фото (эмбеддингов) сохранено для человека."""
        return len(self._data.get("people", {}).get(name, []))

    async def async_add_person(
        self,
        name: str,
        image: bytes,
        base_url: str,
        model_name: str,
        detector_backend: str,
    ) -> None:
        """Считает эмбеддинг лица с фото и добавляет его человеку.

        Поднимает deepface_client.DeepFaceError, если лицо не найдено или сервис
        недоступен. Anti-spoofing на эталонных фото не применяется.
        """
        embeddings = await self._hass.async_add_executor_job(
            deepface_client.represent,
            base_url,
            image,
            model_name,
            detector_backend,
            False,  # anti_spoofing выключен для эталонных фото
        )
        if not embeddings:
            raise deepface_client.NoFaceError("На фото не найдено лицо")
        if len(embeddings) > 1:
            _LOGGER.warning(
                "На фото %s найдено несколько лиц, берём первое", name
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
        people.setdefault(name, []).append(embeddings[0])
        await self._store.async_save(self._data)

    async def async_remove_person(self, name: str) -> None:
        """Удаляет человека и все его эмбеддинги."""
        people = self._data.setdefault("people", {})
        if name in people:
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
        for name, refs in self._data.get("people", {}).items():
            for ref in refs:
                distance = cosine_distance(embedding, ref)
                if distance < best_distance:
                    best_distance = distance
                    best_name = name
        if best_name is None:
            return None
        return best_name, best_distance
