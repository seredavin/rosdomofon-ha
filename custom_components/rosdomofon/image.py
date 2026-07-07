"""
Платформа image для интеграции Росдомофон.

Показывает последние кадры распознанного и нераспознанного лиц. Сущности
нативно отдают JPEG через HA, видны на дашборде, а их изменения попадают в
Логбук (ленту активности) — по клику можно посмотреть сам кадр.
"""

import logging

from homeassistant.components.image import ImageEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    DOMAIN,
    IMAGE_RECOGNIZED_OBJECT_ID,
    IMAGE_UNKNOWN_OBJECT_ID,
)
from .face_unlock import SIGNAL_FACE_UPDATE

_LOGGER = logging.getLogger(__name__)

# Параметры сущностей по типу кадра: (имя, object_id, иконка)
_KIND_META = {
    "recognized": (
        "Последнее распознанное лицо",
        IMAGE_RECOGNIZED_OBJECT_ID,
        "mdi:face-recognition",
    ),
    "unknown": (
        "Последнее нераспознанное лицо",
        IMAGE_UNKNOWN_OBJECT_ID,
        "mdi:account-question",
    ),
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Создаёт image-сущности кадров, если настроено авто-открытие по лицу."""
    coordinator = hass.data[DOMAIN][entry.entry_id].get("face_coordinator")
    if coordinator is None or not coordinator.cameras:
        return
    async_add_entities(
        [
            RosdomofonFaceImage(hass, coordinator, "recognized"),
            RosdomofonFaceImage(hass, coordinator, "unknown"),
        ]
    )


class RosdomofonFaceImage(ImageEntity):
    """Последний кадр распознанного или нераспознанного лица."""

    _attr_has_entity_name = False

    def __init__(self, hass: HomeAssistant, coordinator, kind: str) -> None:
        super().__init__(hass)
        self._coordinator = coordinator
        self._kind = kind
        name, object_id, icon = _KIND_META[kind]
        self._attr_name = name
        self._attr_icon = icon
        self._attr_unique_id = f"{object_id}_image"
        # Явный стабильный entity_id — на него ссылаются события Логбука.
        self.entity_id = f"image.{object_id}"

    def _snapshot(self):
        """Возвращает (байты кадра, время обновления) для своего типа."""
        c = self._coordinator
        if self._kind == "recognized":
            return c.last_recognized_image, c.last_recognized_at
        return c.last_unknown_image, c.last_unknown_at

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        _, updated_at = self._snapshot()
        self._attr_image_last_updated = updated_at
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, SIGNAL_FACE_UPDATE, self._handle_update
            )
        )

    @callback
    def _handle_update(self) -> None:
        _, updated_at = self._snapshot()
        # Обновляем состояние только когда появился новый кадр.
        if updated_at != self._attr_image_last_updated:
            self._attr_image_last_updated = updated_at
            self.async_write_ha_state()

    async def async_image(self) -> bytes | None:
        image, _ = self._snapshot()
        return image

    @property
    def extra_state_attributes(self) -> dict:
        c = self._coordinator
        if self._kind == "recognized":
            return {
                "name": c.last_recognized_name,
                "camera": c.last_recognized_camera,
            }
        return {"camera": c.last_unknown_camera}
