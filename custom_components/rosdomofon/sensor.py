"""
Платформа sensor для интеграции Росдомофон.

Сенсор показывает последнее распознанное лицо (для автоматизаций и логов).
"""

import logging

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .face_unlock import SIGNAL_FACE_UPDATE

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Создаёт сенсор последнего распознанного лица."""
    coordinator = hass.data[DOMAIN][entry.entry_id].get("face_coordinator")
    if coordinator is None or not coordinator.cameras:
        return
    async_add_entities([RosdomofonLastFaceSensor(coordinator)])


class RosdomofonLastFaceSensor(SensorEntity):
    """Последнее распознанное лицо."""

    _attr_icon = "mdi:face-recognition"
    _attr_name = "Последнее распознанное лицо"

    def __init__(self, coordinator) -> None:
        self._coordinator = coordinator
        self._attr_unique_id = "rosdomofon_last_recognized_face"

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, SIGNAL_FACE_UPDATE, self._handle_update
            )
        )

    @callback
    def _handle_update(self) -> None:
        self.async_write_ha_state()

    @property
    def native_value(self) -> str | None:
        return self._coordinator.last_person
