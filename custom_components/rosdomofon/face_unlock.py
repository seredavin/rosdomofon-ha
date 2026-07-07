"""
Координатор авто-открытия двери по лицу для интеграции Росдомофон.

Периодически берёт кадр с включённых камер, отправляет в DeepFace (с проверкой
на подделку), ищет совпадение среди эталонных лиц и при успехе открывает
привязанный замок. Соблюдает кулдаун и шлёт уведомление.
"""

import logging
from datetime import timedelta

from homeassistant.components import persistent_notification
from homeassistant.components.camera import async_get_image
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_time_interval

from . import deepface_client
from .const import (
    CONF_ANTISPOOF,
    CONF_CAMERAS,
    CONF_COOLDOWN,
    CONF_DEEPFACE_URL,
    CONF_INTERVAL,
    CONF_MODEL,
    CONF_THRESHOLD,
    DEFAULT_ANTISPOOF,
    DEFAULT_COOLDOWN,
    DEFAULT_DETECTOR,
    DEFAULT_INTERVAL,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    DOMAIN,
)
from .face_store import FaceStore

_LOGGER = logging.getLogger(__name__)

# Сигнал обновления состояния для sensor/switch
SIGNAL_FACE_UPDATE = f"{DOMAIN}_face_update"


class FaceUnlockCoordinator:
    """Опрашивает камеры и открывает замки по распознанному лицу."""

    def __init__(
        self,
        hass: HomeAssistant,
        face_store: FaceStore,
        options: dict,
    ) -> None:
        self._hass = hass
        self._face_store = face_store
        self._unsub = None
        self._busy: set[str] = set()
        self._cooldown_until: dict[str, float] = {}
        self._enabled: dict[str, bool] = {}
        # Последнее распознавание (для sensor)
        self.last_person: str | None = None

        self._apply_options(options)

    def _apply_options(self, options: dict) -> None:
        """Считывает настройки из options config entry."""
        self._url = options.get(CONF_DEEPFACE_URL, "")
        self._model = options.get(CONF_MODEL, DEFAULT_MODEL)
        self._threshold = float(options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD))
        self._interval = int(options.get(CONF_INTERVAL, DEFAULT_INTERVAL))
        self._cooldown = int(options.get(CONF_COOLDOWN, DEFAULT_COOLDOWN))
        self._anti_spoofing = bool(options.get(CONF_ANTISPOOF, DEFAULT_ANTISPOOF))
        # {camera_entity_id: lock_entity_id}
        self._cameras: dict[str, str] = dict(options.get(CONF_CAMERAS, {}))
        # Сохраняем ранее выставленные переключатели, для новых камер — включено.
        self._enabled = {
            cam: self._enabled.get(cam, True) for cam in self._cameras
        }

    # -- Управление жизненным циклом -------------------------------------

    @callback
    def start(self) -> None:
        """Запускает периодический опрос."""
        self.stop()
        if not self._url or not self._cameras:
            _LOGGER.debug("Авто-открытие по лицу не запущено (нет URL/камер)")
            return
        self._unsub = async_track_time_interval(
            self._hass, self._async_tick, timedelta(seconds=self._interval)
        )
        _LOGGER.info(
            "Авто-открытие по лицу активно для камер: %s",
            ", ".join(self._cameras),
        )

    @callback
    def stop(self) -> None:
        """Останавливает опрос."""
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    @callback
    def update_options(self, options: dict) -> None:
        """Перечитывает настройки и перезапускает опрос."""
        self._apply_options(options)
        self.start()

    # -- Переключатели камер (используются switch-сущностями) ------------

    @property
    def cameras(self) -> list[str]:
        """Камеры с настроенным авто-открытием."""
        return list(self._cameras)

    def lock_for(self, camera: str) -> str | None:
        """Замок, привязанный к камере."""
        return self._cameras.get(camera)

    def is_enabled(self, camera: str) -> bool:
        return self._enabled.get(camera, False)

    @callback
    def set_enabled(self, camera: str, enabled: bool) -> None:
        self._enabled[camera] = enabled
        async_dispatcher_send(self._hass, SIGNAL_FACE_UPDATE)

    # -- Опрос ------------------------------------------------------------

    async def _async_tick(self, _now) -> None:
        """Обрабатывает все включённые камеры за один тик."""
        for camera, lock in self._cameras.items():
            if not self._enabled.get(camera, False):
                continue
            if camera in self._busy:
                continue  # предыдущий кадр ещё обрабатывается
            self._busy.add(camera)
            self._hass.async_create_task(self._process_camera(camera, lock))

    async def _process_camera(self, camera: str, lock: str) -> None:
        """Берёт кадр, распознаёт лицо и при совпадении открывает замок."""
        try:
            now = self._hass.loop.time()
            if now < self._cooldown_until.get(camera, 0):
                return

            try:
                image = await async_get_image(self._hass, camera, timeout=10)
            except Exception as exc:  # noqa: BLE001 — не спамим при недоступности
                _LOGGER.debug("Не удалось получить кадр с %s: %s", camera, exc)
                return

            try:
                embeddings = await self._hass.async_add_executor_job(
                    deepface_client.represent,
                    self._url,
                    image.content,
                    self._model,
                    DEFAULT_DETECTOR,
                    self._anti_spoofing,
                )
            except deepface_client.SpoofDetected:
                _LOGGER.warning("%s: обнаружена подделка (фото/экран), пропуск", camera)
                return
            except deepface_client.DeepFaceError as exc:
                _LOGGER.debug("%s: ошибка DeepFace: %s", camera, exc)
                return

            match = None
            for embedding in embeddings:
                candidate = self._face_store.match(embedding, self._threshold)
                if candidate and (match is None or candidate[1] < match[1]):
                    match = candidate

            if match is None:
                return

            name, distance = match
            await self._unlock(camera, lock, name, distance)
        finally:
            self._busy.discard(camera)

    async def _unlock(
        self, camera: str, lock: str, name: str, distance: float
    ) -> None:
        """Открывает замок и уведомляет пользователя."""
        self._cooldown_until[camera] = self._hass.loop.time() + self._cooldown
        self.last_person = name
        async_dispatcher_send(self._hass, SIGNAL_FACE_UPDATE)

        _LOGGER.info(
            "Распознан %s (расстояние %.3f) на %s — открываю %s",
            name,
            distance,
            camera,
            lock,
        )
        await self._hass.services.async_call(
            "lock", "unlock", {"entity_id": lock}, blocking=False
        )
        persistent_notification.async_create(
            self._hass,
            f"Распознан **{name}** — дверь открыта ({lock}).",
            title="Росдомофон: авто-открытие по лицу 👤🔓",
            notification_id=f"rosdomofon_face_{camera}",
        )
