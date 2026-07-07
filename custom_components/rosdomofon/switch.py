"""
Платформа switch для интеграции Росдомофон.

Переключатель включает/выключает авто-открытие по лицу на конкретной камере.
"""

import logging

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import DOMAIN
from .face_unlock import SIGNAL_FACE_UPDATE

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Создаёт переключатели авто-открытия для настроенных камер."""
    coordinator = hass.data[DOMAIN][entry.entry_id].get("face_coordinator")
    if coordinator is None:
        return

    entities = [
        RosdomofonFaceUnlockSwitch(coordinator, camera)
        for camera in coordinator.cameras
    ]
    async_add_entities(entities)


class RosdomofonFaceUnlockSwitch(SwitchEntity, RestoreEntity):
    """Вкл/выкл авто-открытие по лицу для камеры."""

    _attr_icon = "mdi:face-recognition"
    _attr_has_entity_name = False

    def __init__(self, coordinator, camera_entity: str) -> None:
        self._coordinator = coordinator
        self._camera = camera_entity
        self._attr_unique_id = f"rosdomofon_face_unlock_{camera_entity}"
        self._attr_name = f"Авто-открытие по лицу ({camera_entity})"

    async def async_added_to_hass(self) -> None:
        """Восстанавливает состояние и подписывается на обновления."""
        await super().async_added_to_hass()

        last_state = await self.async_get_last_state()
        if last_state is not None:
            self._coordinator.set_enabled(self._camera, last_state.state == "on")

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, SIGNAL_FACE_UPDATE, self._handle_update
            )
        )

    @callback
    def _handle_update(self) -> None:
        self.async_write_ha_state()

    @property
    def is_on(self) -> bool:
        return self._coordinator.is_enabled(self._camera)

    async def async_turn_on(self, **kwargs) -> None:
        self._coordinator.set_enabled(self._camera, True)
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs) -> None:
        self._coordinator.set_enabled(self._camera, False)
        self.async_write_ha_state()
