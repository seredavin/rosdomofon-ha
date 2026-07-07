"""Тесты распознавания лиц: клиент DeepFace, хранилище, координатор."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from homeassistant.core import HomeAssistant

from custom_components.rosdomofon import deepface_client
from custom_components.rosdomofon.face_store import FaceStore, cosine_distance
from custom_components.rosdomofon.face_unlock import FaceUnlockCoordinator
from custom_components.rosdomofon.const import (
    CONF_CAMERAS,
    CONF_COOLDOWN,
    CONF_DEEPFACE_URL,
    CONF_THRESHOLD,
)


def _response(status, payload):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = str(payload)
    return resp


# --- deepface_client ---------------------------------------------------------


def test_represent_success():
    with patch("requests.post", return_value=_response(200, {"results": [{"embedding": [1.0, 2.0, 3.0]}]})):
        result = deepface_client.represent("http://df", b"img", "Facenet512", "retinaface", True)
    assert result == [[1.0, 2.0, 3.0]]


def test_represent_spoof_raises():
    with patch("requests.post", return_value=_response(400, {"error": "Spoof detected in given image."})):
        with pytest.raises(deepface_client.SpoofDetected):
            deepface_client.represent("http://df", b"img", "Facenet512", "retinaface", True)


def test_represent_no_face_returns_empty():
    with patch("requests.post", return_value=_response(400, {"error": "Face could not be detected."})):
        result = deepface_client.represent("http://df", b"img", "Facenet512", "retinaface", True)
    assert result == []


def test_represent_other_error_raises():
    with patch("requests.post", return_value=_response(500, {"error": "boom"})):
        with pytest.raises(deepface_client.DeepFaceError):
            deepface_client.represent("http://df", b"img", "Facenet512", "retinaface", True)


def test_represent_network_error_raises():
    with patch("requests.post", side_effect=requests.RequestException("down")):
        with pytest.raises(deepface_client.DeepFaceError):
            deepface_client.represent("http://df", b"img", "Facenet512", "retinaface", True)


# --- face_store --------------------------------------------------------------


def test_cosine_distance():
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_face_store_add_and_match(hass: HomeAssistant):
    store = FaceStore(hass)
    await store.async_load()

    with patch(
        "custom_components.rosdomofon.deepface_client.represent",
        return_value=[[1.0, 0.0, 0.0]],
    ):
        await store.async_add_person("Alice", b"img", "http://df", "Facenet512", "retinaface")

    assert store.people == ["Alice"]
    assert store.photo_count("Alice") == 1

    # Похожий эмбеддинг -> совпадение
    match = store.match([0.99, 0.01, 0.0], threshold=0.3)
    assert match is not None and match[0] == "Alice"

    # Непохожий -> нет совпадения
    assert store.match([0.0, 1.0, 0.0], threshold=0.3) is None

    await store.async_remove_person("Alice")
    assert store.people == []


# --- FaceUnlockCoordinator ---------------------------------------------------


def _coordinator(hass, face_store):
    options = {
        CONF_DEEPFACE_URL: "http://df",
        CONF_THRESHOLD: 0.3,
        CONF_COOLDOWN: 30,
        CONF_CAMERAS: {"camera.podezd": "lock.dver"},
    }
    return FaceUnlockCoordinator(hass, face_store, options)


def _track_unlock(hass):
    """Регистрирует мок-сервис lock.unlock и возвращает список вызовов."""
    calls: list = []

    async def handler(call):
        calls.append(call)

    hass.services.async_register("lock", "unlock", handler)
    return calls


@pytest.mark.asyncio
async def test_coordinator_unlocks_on_match(hass: HomeAssistant):
    face_store = MagicMock()
    face_store.match.return_value = ("Alice", 0.1)
    coord = _coordinator(hass, face_store)
    calls = _track_unlock(hass)

    image = MagicMock(content=b"frame")
    with patch("custom_components.rosdomofon.face_unlock.async_get_image", AsyncMock(return_value=image)), \
         patch("custom_components.rosdomofon.deepface_client.represent", return_value=[[1.0, 0.0]]), \
         patch("custom_components.rosdomofon.face_unlock.persistent_notification.async_create"):
        await coord._process_camera("camera.podezd", "lock.dver")
        await hass.async_block_till_done()

    assert len(calls) == 1
    assert calls[0].data["entity_id"] == "lock.dver"
    assert coord.last_person == "Alice"
    # Кадр распознанного лица сохранён для ленты активности
    assert coord.last_recognized_image == b"frame"
    assert coord.last_recognized_name == "Alice"
    assert coord.last_recognized_camera == "camera.podezd"
    assert coord.last_recognized_at is not None


@pytest.mark.asyncio
async def test_coordinator_spoof_does_not_unlock(hass: HomeAssistant):
    face_store = MagicMock()
    coord = _coordinator(hass, face_store)
    calls = _track_unlock(hass)

    image = MagicMock(content=b"frame")
    with patch("custom_components.rosdomofon.face_unlock.async_get_image", AsyncMock(return_value=image)), \
         patch("custom_components.rosdomofon.deepface_client.represent", side_effect=deepface_client.SpoofDetected()):
        await coord._process_camera("camera.podezd", "lock.dver")
        await hass.async_block_till_done()

    assert len(calls) == 0
    face_store.match.assert_not_called()


@pytest.mark.asyncio
async def test_coordinator_antispoof_unavailable_falls_back(hass: HomeAssistant):
    face_store = MagicMock()
    coord = _coordinator(hass, face_store)
    calls = _track_unlock(hass)
    assert coord._anti_spoofing is True

    image = MagicMock(content=b"frame")
    with patch("custom_components.rosdomofon.face_unlock.async_get_image", AsyncMock(return_value=image)), \
         patch("custom_components.rosdomofon.deepface_client.represent", side_effect=deepface_client.AntiSpoofUnavailable()):
        await coord._process_camera("camera.podezd", "lock.dver")
        await hass.async_block_till_done()

    # Кадр пропущен, но антиспуфинг отключён для последующих опросов
    assert len(calls) == 0
    assert coord._anti_spoofing is False


@pytest.mark.asyncio
async def test_coordinator_no_match_does_not_unlock(hass: HomeAssistant):
    face_store = MagicMock()
    face_store.match.return_value = None
    coord = _coordinator(hass, face_store)
    calls = _track_unlock(hass)

    image = MagicMock(content=b"frame")
    with patch("custom_components.rosdomofon.face_unlock.async_get_image", AsyncMock(return_value=image)), \
         patch("custom_components.rosdomofon.deepface_client.represent", return_value=[[1.0, 0.0]]):
        await coord._process_camera("camera.podezd", "lock.dver")
        await hass.async_block_till_done()

    assert len(calls) == 0
    # Нераспознанное лицо сохранено в ленту как «неизвестное»
    assert coord.last_unknown_image == b"frame"
    assert coord.last_unknown_camera == "camera.podezd"
    assert coord.last_unknown_at is not None


@pytest.mark.asyncio
async def test_coordinator_no_face_not_recorded(hass: HomeAssistant):
    """Пустой кадр (лицо не найдено) не должен попадать в ленту."""
    face_store = MagicMock()
    coord = _coordinator(hass, face_store)
    _track_unlock(hass)

    image = MagicMock(content=b"frame")
    with patch("custom_components.rosdomofon.face_unlock.async_get_image", AsyncMock(return_value=image)), \
         patch("custom_components.rosdomofon.deepface_client.represent", return_value=[]):
        await coord._process_camera("camera.podezd", "lock.dver")
        await hass.async_block_till_done()

    assert coord.last_unknown_image is None
    face_store.match.assert_not_called()


@pytest.mark.asyncio
async def test_coordinator_unknown_cooldown(hass: HomeAssistant):
    """Повторное неизвестное лицо внутри кулдауна не создаёт новую запись."""
    face_store = MagicMock()
    face_store.match.return_value = None
    coord = _coordinator(hass, face_store)
    events: list = []
    hass.bus.async_listen("rosdomofon_face_unknown", lambda e: events.append(e))

    first = MagicMock(content=b"frame1")
    second = MagicMock(content=b"frame2")
    with patch("custom_components.rosdomofon.deepface_client.represent", return_value=[[1.0, 0.0]]):
        with patch("custom_components.rosdomofon.face_unlock.async_get_image", AsyncMock(return_value=first)):
            await coord._process_camera("camera.podezd", "lock.dver")
        with patch("custom_components.rosdomofon.face_unlock.async_get_image", AsyncMock(return_value=second)):
            await coord._process_camera("camera.podezd", "lock.dver")
        await hass.async_block_till_done()

    # Второй кадр в пределах кулдауна проигнорирован — остаётся первый
    assert coord.last_unknown_image == b"frame1"
    assert len(events) == 1


@pytest.mark.asyncio
async def test_coordinator_cooldown_blocks_second_unlock(hass: HomeAssistant):
    face_store = MagicMock()
    face_store.match.return_value = ("Alice", 0.1)
    coord = _coordinator(hass, face_store)
    calls = _track_unlock(hass)

    image = MagicMock(content=b"frame")
    with patch("custom_components.rosdomofon.face_unlock.async_get_image", AsyncMock(return_value=image)), \
         patch("custom_components.rosdomofon.deepface_client.represent", return_value=[[1.0, 0.0]]), \
         patch("custom_components.rosdomofon.face_unlock.persistent_notification.async_create"):
        await coord._process_camera("camera.podezd", "lock.dver")
        await coord._process_camera("camera.podezd", "lock.dver")
        await hass.async_block_till_done()

    # Второй вызов внутри кулдауна не должен открывать повторно
    assert len(calls) == 1
