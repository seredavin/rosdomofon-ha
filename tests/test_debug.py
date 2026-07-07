"""Тесты отладочной галереи кадров DeepFace."""

from unittest.mock import MagicMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.rosdomofon.const import DOMAIN
from custom_components.rosdomofon.debug_view import (
    DATA_DEBUG_LOG,
    DebugLog,
    RosdomofonDebugView,
    get_debug_log,
)


# --- DebugLog -----------------------------------------------------------------


def test_debug_log_order_and_cap():
    log = DebugLog(maxlen=2)
    log.add("cam", b"1", "первый", 10, "09:00:00")
    log.add("cam", b"2", "второй", 20, "09:00:01")
    log.add("cam", b"3", "третий", 30, "09:00:02")

    entries = log.entries()
    # Новые сверху, старый вытеснен по лимиту
    assert [e["summary"] for e in entries] == ["третий", "второй"]
    assert entries[0]["image"] == b"3"
    assert entries[0]["elapsed_ms"] == 30


def test_debug_log_clear():
    log = DebugLog()
    log.add("cam", b"1", "x", 1, "09:00:00")
    log.clear()
    assert log.entries() == []


# --- HTTP View ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_debug_view_rejects_without_signature(hass: HomeAssistant):
    """Без корректной подписи галерея отдаёт 401."""
    view = RosdomofonDebugView(hass)
    request = MagicMock(path="/api/rosdomofon/debug", query={})
    request.get.return_value = False  # не аутентифицирован
    response = await view.get(request)
    assert response.status == 401


@pytest.mark.asyncio
async def test_debug_view_renders_gallery(hass: HomeAssistant):
    """С валидной подписью галерея отдаёт HTML с кадрами и результатом."""
    from custom_components.rosdomofon.stream_proxy import sign_proxy_path

    log = DebugLog()
    log.add("camera.dvor", b"\xff\xd8jpeg", "1 лиц → Иван d=0.190", 240, "09:12:00")
    hass.data.setdefault(DOMAIN, {})[DATA_DEBUG_LOG] = log

    signed = sign_proxy_path(hass, "/api/rosdomofon/debug")
    sig = signed.split("sig=", 1)[1]
    request = MagicMock(path="/api/rosdomofon/debug", query={"sig": sig})
    request.get.return_value = False

    view = RosdomofonDebugView(hass)
    response = await view.get(request)

    assert response.status == 200
    body = response.text
    assert "camera.dvor" in body
    assert "Иван" in body
    assert "data:image/jpeg;base64," in body


def test_get_debug_log(hass: HomeAssistant):
    assert get_debug_log(hass) is None
    log = DebugLog()
    hass.data.setdefault(DOMAIN, {})[DATA_DEBUG_LOG] = log
    assert get_debug_log(hass) is log
