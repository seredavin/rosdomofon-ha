"""Тесты страницы просмотра и управления эталонными лицами."""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.rosdomofon.const import (
    CONF_DEEPFACE_URL,
    CONF_DETECTOR,
    CONF_MODEL,
    DATA_FACE_STORE,
    DOMAIN,
)
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
    request = MagicMock(
        path="/api/rosdomofon/faces", query={"sig": sig}, content_type="application/json"
    )
    request.get.return_value = False
    if body is not None:
        request.json = AsyncMock(return_value=body)
    return request


@pytest.mark.asyncio
async def test_register_faces_panel(hass: HomeAssistant):
    from custom_components.rosdomofon import faces_view

    with patch("homeassistant.components.frontend.async_register_built_in_panel") as reg, \
         patch("homeassistant.components.frontend.async_remove_panel"):
        faces_view.async_register_faces_panel(hass)

    assert reg.called
    kwargs = reg.call_args.kwargs
    assert kwargs["component_name"] == "iframe"
    assert kwargs["frontend_url_path"] == faces_view.FACES_PANEL_URL_PATH
    assert kwargs["require_admin"] is True
    assert "/api/rosdomofon/faces" in kwargs["config"]["url"]
    assert "sig=" in kwargs["config"]["url"]


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


def _multipart_request(hass, form):
    from multidict import MultiDict

    post = MultiDict()
    for key, value in form.items():
        if key == "photo":
            for field in (value if isinstance(value, list) else [value]):
                post.add("photo", field)
        else:
            post.add(key, value)

    signed = sign_proxy_path(hass, "/api/rosdomofon/faces")
    sig = signed.split("sig=", 1)[1]
    req = MagicMock(
        path="/api/rosdomofon/faces",
        query={"sig": sig},
        content_type="multipart/form-data",
    )
    req.get.return_value = False
    req.post = AsyncMock(return_value=post)
    return req


def _file(data: bytes):
    field = MagicMock()
    field.file = io.BytesIO(data)
    return field


def _with_deepface_entry(hass):
    entry = MockConfigEntry(
        domain=DOMAIN,
        options={CONF_DEEPFACE_URL: "http://df", CONF_MODEL: "Facenet512", CONF_DETECTOR: "opencv"},
    )
    entry.add_to_hass(hass)


@pytest.mark.asyncio
async def test_faces_add_photo(hass: HomeAssistant):
    store = await _store_with_people(hass)
    _with_deepface_entry(hass)
    field = MagicMock()
    field.file = io.BytesIO(b"img")
    req = _multipart_request(hass, {"name": "Пётр", "photo": field})

    view = RosdomofonFacesView(hass)
    with patch(
        "custom_components.rosdomofon.deepface_client.represent_faces",
        return_value=[{"embedding": [1.0, 0.0], "facial_area": None, "confidence": 0.9}],
    ):
        resp = await view.post(req)

    data = json.loads(resp.body)
    assert data["status"] == "ok"
    assert store.photo_count("Пётр") == 1


@pytest.mark.asyncio
async def test_faces_add_multiple_photos(hass: HomeAssistant):
    store = await _store_with_people(hass)
    _with_deepface_entry(hass)
    req = _multipart_request(
        hass, {"name": "Пётр", "photo": [_file(b"a"), _file(b"b"), _file(b"c")]}
    )
    view = RosdomofonFacesView(hass)
    # Первый и третий — с лицом, второй — без
    seq = [
        [{"embedding": [1.0, 0.0], "facial_area": None, "confidence": 0.9}],
        [],
        [{"embedding": [0.0, 1.0], "facial_area": None, "confidence": 0.9}],
    ]
    with patch("custom_components.rosdomofon.deepface_client.represent_faces", side_effect=seq):
        resp = await view.post(req)

    data = json.loads(resp.body)
    assert data["status"] == "ok"
    assert "добавлено 2" in data["message"]
    assert "без лица 1" in data["message"]
    assert store.photo_count("Пётр") == 2


@pytest.mark.asyncio
async def test_faces_add_photo_no_face(hass: HomeAssistant):
    store = await _store_with_people(hass)
    _with_deepface_entry(hass)
    field = MagicMock()
    field.file = io.BytesIO(b"img")
    req = _multipart_request(hass, {"name": "Пётр", "photo": field})

    view = RosdomofonFacesView(hass)
    with patch("custom_components.rosdomofon.deepface_client.represent_faces", return_value=[]):
        resp = await view.post(req)

    data = json.loads(resp.body)
    assert data["status"] == "error"
    assert "Пётр" not in store.people


@pytest.mark.asyncio
async def test_faces_add_photo_requires_name(hass: HomeAssistant):
    await _store_with_people(hass)
    _with_deepface_entry(hass)
    field = MagicMock()
    field.file = io.BytesIO(b"img")
    req = _multipart_request(hass, {"name": "  ", "photo": field})

    view = RosdomofonFacesView(hass)
    resp = await view.post(req)
    data = json.loads(resp.body)
    assert data["status"] == "error"


@pytest.mark.asyncio
async def test_faces_create_person_without_photo(hass: HomeAssistant):
    store = await _store_with_people(hass)
    view = RosdomofonFacesView(hass)
    resp = await view.post(
        _signed_request(hass, "POST", {"action": "create_person", "name": "Пётр"})
    )
    data = json.loads(resp.body)
    assert data["status"] == "ok"
    assert "Пётр" in store.people
    assert store.photo_count("Пётр") == 0


@pytest.mark.asyncio
async def test_faces_create_person_requires_name(hass: HomeAssistant):
    await _store_with_people(hass)
    view = RosdomofonFacesView(hass)
    resp = await view.post(
        _signed_request(hass, "POST", {"action": "create_person", "name": "   "})
    )
    assert json.loads(resp.body)["status"] == "error"


@pytest.mark.asyncio
async def test_faces_enroll_link(hass: HomeAssistant):
    from custom_components.rosdomofon.enroll import EnrollLinkManager

    store = await _store_with_people(hass)
    mgr = EnrollLinkManager(
        hass, store, {"url": "http://df", "model": "Facenet512", "detector": "opencv"}
    )
    hass.data[DOMAIN]["entry_x"] = {"enroll_manager": mgr}

    view = RosdomofonFacesView(hass)
    with patch("custom_components.rosdomofon.enroll.network.get_url", return_value="https://ha.example"), \
         patch("custom_components.rosdomofon.enroll.webhook.async_register"), \
         patch("custom_components.rosdomofon.enroll.async_call_later", return_value=MagicMock()):
        resp = await view.post(
            _signed_request(hass, "POST", {"action": "enroll_link", "name": "Иван"})
        )

    data = json.loads(resp.body)
    assert data["status"] == "ok"
    assert data["link"].startswith("https://ha.example/api/webhook/")
    assert data["link_person"] == "Иван"
