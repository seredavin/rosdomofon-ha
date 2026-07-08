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
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.util import dt as dt_util

from . import deepface_client, prefilter
from .const import (
    CONF_ANTISPOOF,
    CONF_CAMERAS,
    CONF_COOLDOWN,
    CONF_DEEPFACE_URL,
    CONF_DETECTOR,
    CONF_INTERVAL,
    CONF_DEBUG,
    CONF_MODEL,
    CONF_PREFILTER,
    CONF_THRESHOLD,
    DEFAULT_ANTISPOOF,
    DEFAULT_COOLDOWN,
    DEFAULT_DEBUG,
    DEFAULT_DETECTOR,
    DEFAULT_INTERVAL,
    DEFAULT_MODEL,
    DEFAULT_PREFILTER,
    DEFAULT_THRESHOLD,
    DOMAIN,
    EVENT_FACE_RECOGNIZED,
    EVENT_FACE_UNKNOWN,
)
from .debug_view import get_debug_log
from .face_store import FaceStore
from .stream_grabber import StreamGrabber

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
        # Постоянные ffmpeg-читатели потока по камере (свежие кадры для распознавания)
        self._grabbers: dict[str, StreamGrabber] = {}
        # Предыдущие уменьшенные кадры для детекции движения (по камере)
        self._prev_gray: dict = {}
        # Доступность OpenCV-детектора лиц (fail-open, если opencv не установлен)
        self._face_detect_available = True
        self._face_detect_warned = False
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
        self._prefilter = bool(options.get(CONF_PREFILTER, DEFAULT_PREFILTER))
        self._debug = bool(options.get(CONF_DEBUG, DEFAULT_DEBUG))
        # {camera_entity_id: lock_entity_id}
        self._cameras: dict[str, str] = dict(options.get(CONF_CAMERAS, {}))
        # Сохраняем ранее выставленные переключатели, для новых камер — включено.
        self._enabled = {
            cam: self._enabled.get(cam, True) for cam in self._cameras
        }

    # -- Управление жизненным циклом -------------------------------------

    @callback
    def start(self) -> None:
        """Запускает периодический опрос и постоянные читатели потоков."""
        self._stop_polling()
        if not self._url or not self._cameras:
            _LOGGER.debug("Авто-открытие по лицу не запущено (нет URL/камер)")
            return
        self._unsub = async_track_time_interval(
            self._hass, self._async_tick, timedelta(seconds=self._interval)
        )
        self._reconcile_grabbers()
        _LOGGER.info(
            "Авто-открытие по лицу активно для камер: %s",
            ", ".join(self._cameras),
        )

    @callback
    def _stop_polling(self) -> None:
        """Останавливает только периодический опрос (без читателей потоков)."""
        if self._unsub is not None:
            self._unsub()
            self._unsub = None

    @callback
    def stop(self) -> None:
        """Останавливает опрос и планирует остановку читателей потоков."""
        self._stop_polling()
        for camera in list(self._grabbers):
            self._stop_grabber(camera)

    async def async_shutdown(self) -> None:
        """Полная остановка с ожиданием завершения ffmpeg (для выгрузки entry)."""
        self._stop_polling()
        grabbers = list(self._grabbers.values())
        self._grabbers.clear()
        for grabber in grabbers:
            await grabber.async_stop()

    @callback
    def update_options(self, options: dict) -> None:
        """Перечитывает настройки и перезапускает опрос."""
        self._apply_options(options)
        self.start()

    # -- Постоянные читатели потоков ------------------------------------

    @callback
    def _reconcile_grabbers(self) -> None:
        """Приводит набор читателей потоков в соответствие включённым камерам."""
        for camera in self._cameras:
            if self._url and self._enabled.get(camera, False):
                self._ensure_grabber(camera)
        for camera in list(self._grabbers):
            if camera not in self._cameras or not self._enabled.get(camera, False):
                self._stop_grabber(camera)

    @callback
    def _ensure_grabber(self, camera: str) -> None:
        """Создаёт (при необходимости) и запускает читатель потока камеры."""
        grabber = self._grabbers.get(camera)
        if grabber is None:
            grabber = StreamGrabber(self._hass, camera)
            self._grabbers[camera] = grabber
        grabber.start()

    @callback
    def _stop_grabber(self, camera: str) -> None:
        """Останавливает и убирает читатель потока камеры."""
        grabber = self._grabbers.pop(camera, None)
        if grabber is not None:
            self._hass.async_create_task(grabber.async_stop())

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
        # Читатель потока держим только для включённых камер.
        if enabled and self._url and self._unsub is not None:
            self._ensure_grabber(camera)
        elif not enabled:
            self._stop_grabber(camera)
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

            # Свежий кадр из постоянного читателя потока (не async_get_image:
            # тот дёргает поток по требованию, из-за чего RDVA гасит поток и
            # кадры приходят рвано/битые). None — поток ещё не готов или завис.
            grabber = self._grabbers.get(camera)
            image_bytes = grabber.latest_frame() if grabber is not None else None
            if image_bytes is None:
                return

            # Дешёвый предфильтр: не гоняем DeepFace на пустых/статичных кадрах.
            if self._prefilter and not await self._passes_prefilter(
                camera, image_bytes
            ):
                return

            sent_at = self._hass.loop.time()
            try:
                embeddings = await self._hass.async_add_executor_job(
                    deepface_client.represent,
                    self._url,
                    image_bytes,
                    self._model,
                    self._detector,
                    self._anti_spoofing,
                )
            except deepface_client.AntiSpoofUnavailable:
                # В образе DeepFace нет torch — продолжаем без антиспуфинга.
                self._record_debug(
                    camera, image_bytes, "антиспуфинг недоступен (нет torch)", sent_at
                )
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
                self._record_debug(
                    camera, image_bytes, "подделка (spoof)", sent_at
                )
                _LOGGER.warning("%s: обнаружена подделка (фото/экран), пропуск", camera)
                return
            except deepface_client.DeepFaceError as exc:
                self._record_debug(
                    camera, image_bytes, f"ошибка DeepFace: {exc}", sent_at
                )
                _LOGGER.debug("%s: ошибка DeepFace: %s", camera, exc)
                return

            # Лицо на кадре не найдено — в ленту ничего не пишем.
            if not embeddings:
                self._record_debug(camera, image_bytes, "лицо не найдено (0)", sent_at)
                return

            match = None
            for embedding in embeddings:
                candidate = self._face_store.match(embedding, self._threshold)
                if candidate and (match is None or candidate[1] < match[1]):
                    match = candidate

            self._record_debug(
                camera, image_bytes, self._match_summary(embeddings, match), sent_at
            )

            if match is None:
                self._handle_unknown(camera, image_bytes)
                return

            name, distance = match
            await self._unlock(camera, lock, name, distance, image_bytes)
        finally:
            self._busy.discard(camera)

    async def _passes_prefilter(self, camera: str, image_bytes: bytes) -> bool:
        """Лёгкий фильтр перед DeepFace: движение + лицо в кадре.

        Возвращает True, если кадр стоит отправлять в DeepFace. Работает
        fail-open: при ошибках декодирования или отсутствии OpenCV пропускает
        кадр дальше, чтобы не потерять реальное лицо.
        """
        # 1. Детекция движения — сравниваем с предыдущим кадром камеры.
        cur_gray = await self._hass.async_add_executor_job(
            prefilter.downscale_gray, image_bytes
        )
        if cur_gray is not None:
            prev_gray = self._prev_gray.get(camera)
            self._prev_gray[camera] = cur_gray
            if prev_gray is not None and not prefilter.has_motion(
                prev_gray, cur_gray
            ):
                return False

        # 2. Детекция лица (OpenCV Haar). Без OpenCV — только движение.
        if self._face_detect_available:
            try:
                return await self._hass.async_add_executor_job(
                    prefilter.has_face, image_bytes
                )
            except prefilter.FaceDetectUnavailable:
                if not self._face_detect_warned:
                    _LOGGER.info(
                        "Локальный детектор лиц OpenCV недоступен — это нормально "
                        "(в свежих сборках Home Assistant opencv не ставится). "
                        "Проверка на лицо выполняется на стороне DeepFace, "
                        "локальный предфильтр работает по детекции движения."
                    )
                    self._face_detect_warned = True
                self._face_detect_available = False

        return True

    def _match_summary(self, embeddings: list, match) -> str:
        """Текстовый результат распознавания для отладочной галереи."""
        if not self._debug:
            return ""
        parts = [f"{len(embeddings)} лиц"]
        if match is not None:
            parts.append(
                f"→ {match[0]} d={match[1]:.3f} (порог {self._threshold})"
            )
        else:
            nearest = self._face_store.nearest(embeddings)
            if nearest is not None:
                parts.append(
                    f"→ нет совпадения, ближайший {nearest[0]} "
                    f"d={nearest[1]:.3f} (порог {self._threshold})"
                )
            else:
                parts.append("→ эталонов нет")
        return ", ".join(parts)

    def _record_debug(
        self, camera: str, image_bytes: bytes, summary: str, sent_at: float
    ) -> None:
        """Пишет кадр и результат в отладочную галерею (если отладка включена)."""
        if not self._debug:
            return
        debug_log = get_debug_log(self._hass)
        if debug_log is None:
            return
        elapsed_ms = int((self._hass.loop.time() - sent_at) * 1000)
        when = dt_util.now().strftime("%H:%M:%S")
        debug_log.add(camera, image_bytes, summary, elapsed_ms, when)

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
