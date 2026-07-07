from homeassistant.exceptions import ConfigEntryNotReady
"""
Интеграция Росдомофон для Home Assistant.

Обеспечивает управление замками (двери, шлагбаумы, ворота, калитки)
через облачный API Росдомофон.
Поддерживает генерацию временных гостевых ссылок для открытия дверей.
"""

import logging

import voluptuous as vol
from homeassistant.components import persistent_notification
from homeassistant.helpers import config_validation as cv

from .const import (
    CONF_DEBUG,
    CONF_DEEPFACE_URL,
    CONF_DETECTOR,
    CONF_MODEL,
    DATA_FACE_STORE,
    DEFAULT_DEBUG,
    DEFAULT_DETECTOR,
    DEFAULT_MODEL,
    DOMAIN,
    ENROLL_LINK_DEFAULT_TTL_HOURS,
    SHARE_LINK_DEFAULT_TTL_HOURS,
)
from .debug_view import debug_gallery_url, setup_debug_view
from .enroll import EnrollLinkManager
from .face_store import FaceStore
from .face_unlock import FaceUnlockCoordinator
from .share import ExternalURLNotAvailable, ShareLinkManager
from .stream_proxy import setup_stream_proxy
from .token_manager import TokenManager

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[str] = ["lock", "button", "camera", "switch", "sensor", "image"]

# Схема сервиса генерации гостевой ссылки
SERVICE_GENERATE_LINK = "generate_share_link"
SERVICE_GENERATE_LINK_SCHEMA = vol.Schema({
    vol.Required("entity_id"): cv.entity_id,
    vol.Optional("ttl_hours", default=SHARE_LINK_DEFAULT_TTL_HOURS): vol.All(
        vol.Coerce(float), vol.Range(min=0.5, max=168)
    ),
})

# Схема сервиса генерации ссылки на добавление лица
SERVICE_GENERATE_ENROLL = "generate_enroll_link"
SERVICE_GENERATE_ENROLL_SCHEMA = vol.Schema({
    vol.Required("camera"): cv.entity_id,
    vol.Required("person"): vol.All(cv.string, vol.Length(min=1)),
    vol.Optional("ttl_hours", default=ENROLL_LINK_DEFAULT_TTL_HOURS): vol.All(
        vol.Coerce(float), vol.Range(min=0.5, max=168)
    ),
})


async def async_setup_entry(hass, entry) -> bool:
    """Настройка интеграции при добавлении config entry."""
    token_manager = TokenManager(hass, entry)

    if not await token_manager.ensure_valid_token():
        _LOGGER.error("Не удалось обновить токен при старте")
        raise ConfigEntryNotReady("Не удалось обновить токен при старте")

    share_manager = ShareLinkManager(hass)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        "token_manager": token_manager,
        "share_manager": share_manager,
    }

    # Распознавание лиц (авто-открытие по лицу). Хранилище лиц — одно на домен.
    face_store = hass.data[DOMAIN].get(DATA_FACE_STORE)
    if face_store is None:
        face_store = FaceStore(hass)
        await face_store.async_load()
        hass.data[DOMAIN][DATA_FACE_STORE] = face_store

    coordinator = FaceUnlockCoordinator(hass, face_store, dict(entry.options))
    hass.data[DOMAIN][entry.entry_id]["face_coordinator"] = coordinator

    # Менеджер ссылок для самостоятельного добавления лиц.
    deepface_config = {
        "url": entry.options.get(CONF_DEEPFACE_URL, ""),
        "model": entry.options.get(CONF_MODEL, DEFAULT_MODEL),
        "detector": entry.options.get(CONF_DETECTOR, DEFAULT_DETECTOR),
    }
    enroll_manager = EnrollLinkManager(hass, face_store, deepface_config)
    hass.data[DOMAIN][entry.entry_id]["enroll_manager"] = enroll_manager

    entry.async_on_unload(entry.add_update_listener(_async_options_updated))

    # Регистрируем прокси для HLS потоков (один раз на домен)
    if "_stream_proxy_registered" not in hass.data[DOMAIN]:
        setup_stream_proxy(hass)
        hass.data[DOMAIN]["_stream_proxy_registered"] = True

    # Регистрируем отладочную галерею DeepFace (один раз на домен)
    if "_debug_registered" not in hass.data[DOMAIN]:
        setup_debug_view(hass)
        hass.data[DOMAIN]["_debug_registered"] = True

    # При включённой отладке показываем ссылку на галерею кадров.
    if entry.options.get(CONF_DEBUG, DEFAULT_DEBUG):
        url = debug_gallery_url(hass)
        if url:
            persistent_notification.async_create(
                hass,
                f"Отладка распознавания включена.\n\n"
                f"Галерея кадров, отправленных в DeepFace:\n\n{url}",
                title="Росдомофон: отладка распознавания 🐞",
                notification_id="rosdomofon_face_debug",
            )
        else:
            _LOGGER.warning(
                "Отладка включена, но не удалось построить ссылку на галерею "
                "(не настроен внутренний/внешний URL Home Assistant)."
            )
    else:
        persistent_notification.async_dismiss(hass, "rosdomofon_face_debug")

    # Регистрируем сервис генерации ссылки (один раз на домен)
    if not hass.services.has_service(DOMAIN, SERVICE_GENERATE_LINK):
        async def handle_generate_link(call):
            """Обработчик сервиса rosdomofon.generate_share_link."""
            entity_id = call.data["entity_id"]
            ttl_hours = call.data.get("ttl_hours", SHARE_LINK_DEFAULT_TTL_HOURS)

            # Находим share_manager для любого активного entry
            mgr = None
            for _eid, data in hass.data.get(DOMAIN, {}).items():
                if isinstance(data, dict) and "share_manager" in data:
                    mgr = data["share_manager"]
                    break

            if mgr is None:
                _LOGGER.error("Интеграция не настроена")
                return

            try:
                url = mgr.generate(entity_id, ttl_hours)
            except ExternalURLNotAvailable:
                persistent_notification.async_create(
                    hass,
                    "Невозможно создать гостевую ссылку: "
                    "в Home Assistant не настроен внешний доступ. "
                    "Настройте External URL или подключите Home Assistant Cloud (Nabu Casa).",
                    title="Росдомофон: внешний доступ не настроен",
                    notification_id="rosdomofon_no_external_url",
                )
                return

            ttl_text = f"{int(ttl_hours)} ч" if ttl_hours == int(ttl_hours) else f"{ttl_hours} ч"
            persistent_notification.async_create(
                hass,
                f"Ссылка для открытия **{entity_id}** "
                f"(действительна {ttl_text}):\n\n"
                f"`{url}`\n\n"
                f"Скопируйте и отправьте гостю.",
                title="Росдомофон: гостевая ссылка создана 🔗",
                notification_id=f"rosdomofon_share_{entity_id}",
            )

        hass.services.async_register(
            DOMAIN,
            SERVICE_GENERATE_LINK,
            handle_generate_link,
            schema=SERVICE_GENERATE_LINK_SCHEMA,
        )

    # Регистрируем сервис генерации ссылки на добавление лица (один раз на домен)
    if not hass.services.has_service(DOMAIN, SERVICE_GENERATE_ENROLL):
        async def handle_generate_enroll(call):
            """Обработчик сервиса rosdomofon.generate_enroll_link."""
            camera = call.data["camera"]
            person = call.data["person"].strip()
            ttl_hours = call.data.get("ttl_hours", ENROLL_LINK_DEFAULT_TTL_HOURS)

            mgr = None
            for _eid, data in hass.data.get(DOMAIN, {}).items():
                if isinstance(data, dict) and "enroll_manager" in data:
                    mgr = data["enroll_manager"]
                    break

            if mgr is None:
                _LOGGER.error("Интеграция не настроена")
                return

            try:
                url = mgr.generate(camera, person, ttl_hours)
            except ExternalURLNotAvailable:
                persistent_notification.async_create(
                    hass,
                    "Невозможно создать ссылку: в Home Assistant не настроен внешний доступ. "
                    "Настройте External URL или подключите Home Assistant Cloud (Nabu Casa).",
                    title="Росдомофон: внешний доступ не настроен",
                    notification_id="rosdomofon_no_external_url",
                )
                return

            ttl_text = f"{int(ttl_hours)} ч" if ttl_hours == int(ttl_hours) else f"{ttl_hours} ч"
            persistent_notification.async_create(
                hass,
                f"Ссылка для добавления лица **{person}** "
                f"(камера {camera}, действительна {ttl_text}):\n\n"
                f"`{url}`\n\n"
                f"Откройте на телефоне: снимите лицо с камеры или загрузите фото. "
                f"⚠️ По ссылке можно добавить лицо в список «своих» — не пересылайте посторонним.",
                title="Росдомофон: ссылка для добавления лица 👤➕",
                notification_id=f"rosdomofon_enroll_{person}",
            )

        hass.services.async_register(
            DOMAIN,
            SERVICE_GENERATE_ENROLL,
            handle_generate_enroll,
            schema=SERVICE_GENERATE_ENROLL_SCHEMA,
        )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Запускаем опрос камер после создания сущностей.
    coordinator.start()
    return True


async def _async_options_updated(hass, entry) -> None:
    """Перезагружает интеграцию при изменении настроек распознавания."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass, entry) -> bool:
    """Выгрузка интеграции при удалении config entry."""
    coordinator = hass.data[DOMAIN].get(entry.entry_id, {}).get("face_coordinator")
    if coordinator:
        coordinator.stop()

    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    if unload_ok:
        data = hass.data[DOMAIN].pop(entry.entry_id, {})
        share_manager = data.get("share_manager")
        if share_manager:
            share_manager.revoke_all()
        enroll_manager = data.get("enroll_manager")
        if enroll_manager:
            enroll_manager.revoke_all()

        # Если больше нет активных entry, удаляем сервисы
        if not any(
            isinstance(v, dict) and "token_manager" in v
            for v in hass.data.get(DOMAIN, {}).values()
        ):
            hass.services.async_remove(DOMAIN, SERVICE_GENERATE_LINK)
            hass.services.async_remove(DOMAIN, SERVICE_GENERATE_ENROLL)

    return unload_ok
