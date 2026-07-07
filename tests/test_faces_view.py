"""Тесты страницы просмотра и управления эталонными лицами."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.rosdomofon.const import DATA_FACE_STORE, DOMAIN
from custom_components.rosdomofon.face_store import FaceStore
from custom_components.rosdomofon.faces_view import RosdomofonFacesView
from custom_components.rosdomofon.stream_proxy import sign_proxy_path


async def _store_with_people(hass) -> FaceStore:
    store = FaceStore(hass)
    await store.async_load()
    with patch(
        "custom_components.rosdomofon.deepface_client.represent_faces",
        return_value=[{"embedding": [1.0, 0.0], "facial_area": None, "confidence": 0.9}],
    ):
        await store.async_add_person("Иван", b"i1", "http://df", "Facenet512", "opencv")
        await store.async_add_person("Иван", b"i2", "http://df", "Facenet512", "opencv")
        await store.async_add_person("Мария", b"m1", "http://df", "Facenet512", "opencv")
    hass.data.setdefault(DOMAIN, {})[DATA_FACE_STORE] = store
    return store


def _signed_request(hass, method="GET", body=None):
    signed = sign_proxy_path(hass, "/api/rosdomofon/faces")
    sig = signed.split("sig=", 1)[1]
    request = MagicMock(path="/api/rosdomofon/faces", query={"sig": sig})
    request.get.return_value = False
    if body is not None:
        request.json = AsyncMock(return_value=body)
    return request


@pytest.mark.asyncio
async def test_faces_get_rejects_without_signature(hass: HomeAssistant):
    view = RosdomofonFacesView(hass)
    request = MagicMock(path="/api/rosdomofon/faces", query={})
    request.get.return_value = False
    resp = await view.get(request)
    assert resp.status == 401


@pytest.mark.asyncio
async def test_faces_get_renders(hass: HomeAssistant):
    await _store_with_people(hass)
    view = RosdomofonFacesView(hass)
    resp = await view.get(_signed_request(hass))
    assert resp.status == 200
    body = resp.text
    assert "Иван" in body and "Мария" in body
    assert "2 фото" in body  # у Ивана два фото


@pytest.mark.asyncio
async def test_faces_delete_photo(hass: HomeAssistant):
    store = await _store_with_people(hass)
    photo_id = store.photos("Иван")[0]["id"]
    view = RosdomofonFacesView(hass)

    resp = await view.post(
        _signed_request(hass, "POST", {"action": "delete_photo", "name": "Иван", "id": photo_id})
    )
    data = json.loads(resp.body)
    assert data["status"] == "ok"
    assert store.photo_count("Иван") == 1


@pytest.mark.asyncio
async def test_faces_delete_person(hass: HomeAssistant):
    store = await _store_with_people(hass)
    view = RosdomofonFacesView(hass)

    resp = await view.post(
        _signed_request(hass, "POST", {"action": "delete_person", "name": "Мария"})
    )
    data = json.loads(resp.body)
    assert "Мария" not in [p["name"] for p in data["people"]]
    assert "Мария" not in store.people


@pytest.mark.asyncio
async def test_faces_post_rejects_without_signature(hass: HomeAssistant):
    await _store_with_people(hass)
    view = RosdomofonFacesView(hass)
    request = MagicMock(path="/api/rosdomofon/faces", query={})
    request.get.return_value = False
    request.json = AsyncMock(return_value={"action": "list"})
    resp = await view.post(request)
    assert resp.status == 401
