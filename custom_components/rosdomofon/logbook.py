"""
Описание событий распознавания лиц для Логбука (ленты активности).

Превращает события rosdomofon_face_recognized / rosdomofon_face_unknown в
читаемые строки Логбука и привязывает их к image-сущности с кадром, чтобы из
ленты можно было сразу открыть снимок.
"""

from homeassistant.components.logbook import (
    LOGBOOK_ENTRY_ENTITY_ID,
    LOGBOOK_ENTRY_MESSAGE,
    LOGBOOK_ENTRY_NAME,
)
from homeassistant.core import Event, HomeAssistant, callback

from .const import (
    DOMAIN,
    EVENT_FACE_RECOGNIZED,
    EVENT_FACE_UNKNOWN,
    IMAGE_RECOGNIZED_ENTITY_ID,
    IMAGE_UNKNOWN_ENTITY_ID,
)


@callback
def async_describe_events(hass: HomeAssistant, async_describe_event) -> None:
    """Регистрирует форматирование событий лиц для Логбука."""

    @callback
    def describe_recognized(event: Event) -> dict:
        data = event.data
        name = data.get("name") or "лицо"
        camera = data.get("camera")
        message = f"распознан(а) {name}"
        if camera:
            message += f" ({camera})"
        return {
            LOGBOOK_ENTRY_NAME: "Росдомофон",
            LOGBOOK_ENTRY_MESSAGE: message,
            LOGBOOK_ENTRY_ENTITY_ID: IMAGE_RECOGNIZED_ENTITY_ID,
        }

    @callback
    def describe_unknown(event: Event) -> dict:
        camera = event.data.get("camera")
        message = "обнаружено неизвестное лицо"
        if camera:
            message += f" ({camera})"
        return {
            LOGBOOK_ENTRY_NAME: "Росдомофон",
            LOGBOOK_ENTRY_MESSAGE: message,
            LOGBOOK_ENTRY_ENTITY_ID: IMAGE_UNKNOWN_ENTITY_ID,
        }

    async_describe_event(DOMAIN, EVENT_FACE_RECOGNIZED, describe_recognized)
    async_describe_event(DOMAIN, EVENT_FACE_UNKNOWN, describe_unknown)
