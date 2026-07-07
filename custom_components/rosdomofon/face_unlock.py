"""
Координатор авто-открытия двери по лицу для интеграции Росдомофон.

Периодически берёт кадр с включённых камер, отправляет в DeepFace (с проверкой
на подделку), ищет совпадение среди эталонных лиц и при успехе открывает
привязанный замок. Соблюдает кулдаун и шлёт уведомление.
"""

import logging
from datetime import timedelta

from datetime import datetime

from homeassistant.components import persistent_notification
from homeassistant.components.camera import async_get_image
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util

from . import deepface_client
from .const import (
    CONF_ANTISPOOF,
    CONF_CAMERAS,
    CONF_COOLDOWN,
    CONF_DEEPFACE_URL,
    CONF_DETECTOR,
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
    EVENT_FACE_RECOGNIZED,
    EVENT_FACE_UNKNOWN,
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
        # Отдельный кулдаун на запись кадров неизвестных лиц, чтобы не засорять ленту
        self._unknown_cooldown_until: dict[str, float] = {}
        self._enabled: dict[str, bool] = {}
        # Один раз предупреждаем, если антиспуфинг недоступен (нет torch)
        self._antispoof_warned = False
        # Последнее распознавание (для sensor)
        self.last_person: str | None = None

        # Последние кадры для ленты активности (image-сущности).
        # Кадры держим в памяти — при перезапуске HA история кадров очищается.
        self.last_recognized_image: bytes | None = None
        self.last_recognized_at: datetime | None = None
        self.last_recognized_name: str | None = None
        self.last_recognized_camera: str | None = None
        self.last_unknown_image: bytes | None = None
        self.last_unknown_at: datetime | None = None
        self.last_unknown_camera: str | None = None

        self._apply_options(options)

    def _apply_options(self, options: dict) -> None:
        """Считывает настройки из options config entry."""
        self._url = options.get(CONF_DEEPFACE_URL, "")
        self._model = options.get(CONF_MODEL, DEFAULT_MODEL)
        self._threshold = float(options.get(CONF_THRESHOLD, DEFAULT_THRESHOLD))
        self._interval = int(options.get(CONF_INTERVAL, DEFAULT_INTERVAL))
        self._cooldown = int(options.get(CONF_COOLDOWN, DEFAULT_COOLDOWN))
        self._anti_spoofing = bool(options.get(CONF_ANTISPOOF, DEFAULT_ANTISPOOF))
        self._detector = options.get(CONF_DETECTOR, DEFAULT_DETECTOR)
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
                    self._detector,
                    self._anti_spoofing,
                )
            except deepface_client.AntiSpoofUnavailable:
                # В образе DeepFace нет torch — продолжаем без антиспуфинга.
                if not self._antispoof_warned:
                    _LOGGER.warning(
                        "Антиспуфинг недоступен в DeepFace (не установлен torch). "
                        "Продолжаю распознавание без проверки на подделку. "
                        "Установите torch в сервисе, чтобы включить защиту от фото/экрана."
                    )
                    self._antispoof_warned = True
                self._anti_spoofing = False
                return
            except deepface_client.SpoofDetected:
                _LOGGER.warning("%s: обнаружена подделка (фото/экран), пропуск", camera)
                return
            except deepface_client.DeepFaceError as exc:
                _LOGGER.debug("%s: ошибка DeepFace: %s", camera, exc)
                return

            # Лицо на кадре не найдено — в ленту ничего не пишем.
            if not embeddings:
                return

            match = None
            for embedding in embeddings:
                candidate = self._face_store.match(embedding, self._threshold)
                if candidate and (match is None or candidate[1] < match[1]):
                    match = candidate

            if match is None:
                self._handle_unknown(camera, image.content)
                return

            name, distance = match
            await self._unlock(camera, lock, name, distance, image.content)
        finally:
            self._busy.discard(camera)

    @callback
    def _handle_unknown(self, camera: str, image_bytes: bytes) -> None:
        """Сохраняет кадр нераспознанного лица в ленту активности.

        Работает с кулдауном на камеру, чтобы одно и то же неизвестное лицо
        не порождало запись в ленте на каждом кадре.
        """
        now = self._hass.loop.time()
        if now < self._unknown_cooldown_until.get(camera, 0):
            return
        self._unknown_cooldown_until[camera] = now + self._cooldown

        self.last_unknown_image = image_bytes
        self.last_unknown_at = dt_util.utcnow()
        self.last_unknown_camera = camera
        async_dispatcher_send(self._hass, SIGNAL_FACE_UPDATE)
        self._hass.bus.async_fire(EVENT_FACE_UNKNOWN, {"camera": camera})

        _LOGGER.info("Обнаружено неизвестное лицо на %s", camera)

    async def _unlock(
        self, camera: str, lock: str, name: str, distance: float, image_bytes: bytes
    ) -> None:
        """Открывает замок, сохраняет кадр в ленту и уведомляет пользователя."""
        self._cooldown_until[camera] = self._hass.loop.time() + self._cooldown
        self.last_person = name
        self.last_recognized_image = image_bytes
        self.last_recognized_at = dt_util.utcnow()
        self.last_recognized_name = name
        self.last_recognized_camera = camera
        async_dispatcher_send(self._hass, SIGNAL_FACE_UPDATE)
        self._hass.bus.async_fire(
            EVENT_FACE_RECOGNIZED,
            {"camera": camera, "name": name, "distance": round(distance, 3)},
        )

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
