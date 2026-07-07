"""Тесты image-сущностей ленты активности (кадры распознанных/неизвестных лиц)."""

from unittest.mock import MagicMock

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.rosdomofon.image import RosdomofonFaceImage
from custom_components.rosdomofon.const import (
    IMAGE_RECOGNIZED_ENTITY_ID,
    IMAGE_UNKNOWN_ENTITY_ID,
)


def _coordinator():
    c = MagicMock()
    c.last_recognized_image = None
    c.last_recognized_at = None
    c.last_recognized_name = None
    c.last_recognized_camera = None
    c.last_unknown_image = None
    c.last_unknown_at = None
    c.last_unknown_camera = None
    return c


@pytest.mark.asyncio
async def test_recognized_image_entity(hass: HomeAssistant):
    coord = _coordinator()
    entity = RosdomofonFaceImage(hass, coord, "recognized")

    # Стабильный entity_id — на него ссылается Логбук
    assert entity.entity_id == IMAGE_RECOGNIZED_ENTITY_ID
    assert await entity.async_image() is None

    # Появился кадр -> сущность отдаёт байты и время обновления
    ts = dt_util.utcnow()
    coord.last_recognized_image = b"jpeg-bytes"
    coord.last_recognized_at = ts
    coord.last_recognized_name = "Alice"
    coord.last_recognized_camera = "camera.podezd"

    assert await entity.async_image() == b"jpeg-bytes"
    assert entity.extra_state_attributes == {
        "name": "Alice",
        "camera": "camera.podezd",
    }


@pytest.mark.asyncio
async def test_unknown_image_entity(hass: HomeAssistant):
    coord = _coordinator()
    entity = RosdomofonFaceImage(hass, coord, "unknown")

    assert entity.entity_id == IMAGE_UNKNOWN_ENTITY_ID
    coord.last_unknown_image = b"unknown-jpeg"
    coord.last_unknown_camera = "camera.dvor"
    assert await entity.async_image() == b"unknown-jpeg"
    assert entity.extra_state_attributes == {"camera": "camera.dvor"}


@pytest.mark.asyncio
async def test_image_updates_on_new_frame(hass: HomeAssistant):
    """_handle_update меняет состояние только при новом кадре."""
    coord = _coordinator()
    entity = RosdomofonFaceImage(hass, coord, "recognized")
    entity.hass = hass
    entity.entity_id = IMAGE_RECOGNIZED_ENTITY_ID
    entity.async_write_ha_state = MagicMock()

    # Синхронизируем начальное значение (как в async_added_to_hass)
    entity._attr_image_last_updated = coord.last_recognized_at

    # Нет нового кадра -> состояние не пишется
    entity._handle_update()
    entity.async_write_ha_state.assert_not_called()

    # Новый кадр -> обновляем last_updated и пишем состояние
    coord.last_recognized_at = dt_util.utcnow()
    entity._handle_update()
    entity.async_write_ha_state.assert_called_once()
    assert entity._attr_image_last_updated == coord.last_recognized_at
