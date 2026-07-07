"""Тесты самостоятельного добавления лица по ссылке (Enroll Link) и связанного кода."""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.rosdomofon import deepface_client, face_crop
from custom_components.rosdomofon.const import DOMAIN
from custom_components.rosdomofon.face_store import FaceStore
from custom_components.rosdomofon.enroll import EnrollLinkManager


def _response(status, payload):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = str(payload)
    return resp


def _file(data: bytes):
    field = MagicMock()
    field.file = io.BytesIO(data)
    return field


def _upload_request(datas):
    """POST-запрос загрузки одного/нескольких фото (multipart)."""
    from multidict import MultiDict

    post = MultiDict()
    for data in datas:
        post.add("photo", _file(data))
    req = MagicMock(method="POST", content_type="multipart/form-data")
    req.post = AsyncMock(return_value=post)
    return req


def _jpeg(size=(300, 300), color=(120, 120, 120)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


# --- deepface_client.represent_faces -----------------------------------------


def test_represent_faces_parses_area():
    payload = {"results": [{"embedding": [1.0, 2.0], "facial_area": {"x": 1, "y": 2, "w": 3, "h": 4}, "face_confidence": 0.9}]}
    with patch("requests.post", return_value=_response(200, payload)):
        faces = deepface_client.represent_faces("http://df", b"img", "Facenet512", "opencv", False)
    assert faces == [{"embedding": [1.0, 2.0], "facial_area": {"x": 1, "y": 2, "w": 3, "h": 4}, "confidence": 0.9}]


def test_represent_wraps_faces():
    payload = {"results": [{"embedding": [1.0, 2.0]}, {"embedding": [3.0]}]}
    with patch("requests.post", return_value=_response(200, payload)):
        embs = deepface_client.represent("http://df", b"img", "Facenet512", "opencv", False)
    assert embs == [[1.0, 2.0], [3.0]]


# --- face_crop ----------------------------------------------------------------


def test_crop_face_fail_open_on_garbage():
    assert face_crop.crop_face(b"not-image", {"x": 0, "y": 0, "w": 10, "h": 10}) == b"not-image"


def test_crop_face_valid():
    pytest.importorskip("PIL")
    src = _jpeg((400, 400))
    out = face_crop.crop_face(src, {"x": 100, "y": 100, "w": 120, "h": 120})
    assert isinstance(out, bytes) and len(out) > 0
    from PIL import Image

    img = Image.open(io.BytesIO(out))
    assert max(img.size) <= 400  # уменьшено до max_side


# --- FaceStore: миграция и поштучное удаление ---------------------------------


@pytest.mark.asyncio
async def test_face_store_migrates_old_format(hass: HomeAssistant):
    store = FaceStore(hass)
    # Старый формат: список «голых» эмбеддингов
    with patch.object(store._store, "async_load", AsyncMock(return_value={
        "model": "Facenet512",
        "people": {"Bob": [[1.0, 0.0], [0.0, 1.0]]},
    })):
        await store.async_load()

    photos = store.photos("Bob")
    assert len(photos) == 2
    assert all(p["id"] for p in photos)
    assert all(p["photo"] is None for p in photos)  # старых фото нет
    # match работает по мигрированным эмбеддингам
    assert store.match([1.0, 0.0], threshold=0.3)[0] == "Bob"


@pytest.mark.asyncio
async def test_face_store_remove_photo(hass: HomeAssistant):
    store = FaceStore(hass)
    await store.async_load()
    with patch(
        "custom_components.rosdomofon.deepface_client.represent_faces",
        return_value=[{"embedding": [1.0, 0.0], "facial_area": None, "confidence": 0.9}],
    ):
        await store.async_add_person("Ann", b"i1", "http://df", "Facenet512", "opencv")
        await store.async_add_person("Ann", b"i2", "http://df", "Facenet512", "opencv")

    assert store.photo_count("Ann") == 2
    first_id = store.photos("Ann")[0]["id"]
    await store.async_remove_photo("Ann", first_id)
    assert store.photo_count("Ann") == 1

    # Удаление последнего фото убирает человека
    last_id = store.photos("Ann")[0]["id"]
    await store.async_remove_photo("Ann", last_id)
    assert "Ann" not in store.people


# --- EnrollLinkManager --------------------------------------------------------


def _manager(hass, face_store):
    return EnrollLinkManager(
        hass, face_store, {"url": "http://df", "model": "Facenet512", "detector": "opencv"}
    )


def _make_link(hass, manager, person="Ivan", camera="camera.dvor"):
    """Регистрирует ссылку в обход внешних зависимостей, возвращает webhook_id."""
    with patch("custom_components.rosdomofon.enroll.network.get_url", return_value="https://ha.example"), \
         patch("custom_components.rosdomofon.enroll.webhook.async_register"), \
         patch("custom_components.rosdomofon.enroll.async_call_later", return_value=MagicMock()):
        url = manager.generate(camera, person)
    assert url.startswith("https://ha.example/api/webhook/")
    return url.rsplit("/", 1)[1]


@pytest.mark.asyncio
async def test_enroll_expired_link_returns_410(hass: HomeAssistant):
    manager = _manager(hass, FaceStore(hass))
    wh_id = _make_link(hass, manager)
    manager._links[wh_id].created_at = 0  # протухла

    request = MagicMock(method="GET", query={})
    resp = await manager._handle_webhook(hass, wh_id, request)
    assert resp.status == 410


@pytest.mark.asyncio
async def test_enroll_get_page(hass: HomeAssistant):
    store = FaceStore(hass)
    await store.async_load()
    manager = _manager(hass, store)
    wh_id = _make_link(hass, manager, person="Иван", camera="camera.dvor")
    hass.states.async_set("camera.dvor", "streaming", {"friendly_name": "Двор"})

    request = MagicMock(method="GET", query={})
    resp = await manager._handle_webhook(hass, wh_id, request)
    assert resp.status == 200
    assert "Иван" in resp.text


@pytest.mark.asyncio
async def test_enroll_capture_adds_face(hass: HomeAssistant):
    store = FaceStore(hass)
    await store.async_load()
    manager = _manager(hass, store)
    wh_id = _make_link(hass, manager, person="Ivan")

    request = MagicMock(method="POST", content_type="application/json")
    request.json = AsyncMock(return_value={"action": "capture", "camera": "camera.dvor"})
    image = MagicMock(content=b"frame", content_type="image/jpeg")

    with patch("custom_components.rosdomofon.enroll.async_get_image", AsyncMock(return_value=image)), \
         patch("custom_components.rosdomofon.deepface_client.represent_faces",
               return_value=[{"embedding": [1.0, 0.0], "facial_area": None, "confidence": 0.9}]):
        resp = await manager._handle_webhook(hass, wh_id, request)

    data = json.loads(resp.body)
    assert data["status"] == "ok"
    assert data["count"] == 1
    assert store.photo_count("Ivan") == 1


@pytest.mark.asyncio
async def test_enroll_capture_no_face(hass: HomeAssistant):
    store = FaceStore(hass)
    await store.async_load()
    manager = _manager(hass, store)
    wh_id = _make_link(hass, manager, person="Ivan")

    request = MagicMock(method="POST", content_type="application/json")
    request.json = AsyncMock(return_value={"action": "capture", "camera": "camera.dvor"})
    image = MagicMock(content=b"frame", content_type="image/jpeg")

    with patch("custom_components.rosdomofon.enroll.async_get_image", AsyncMock(return_value=image)), \
         patch("custom_components.rosdomofon.deepface_client.represent_faces", return_value=[]):
        resp = await manager._handle_webhook(hass, wh_id, request)

    data = json.loads(resp.body)
    assert data["status"] == "error"
    assert store.photo_count("Ivan") == 0


@pytest.mark.asyncio
async def test_enroll_upload_and_delete(hass: HomeAssistant):
    store = FaceStore(hass)
    await store.async_load()
    manager = _manager(hass, store)
    wh_id = _make_link(hass, manager, person="Ivan")

    # Загрузка фото (multipart)
    upload_req = _upload_request([b"uploaded"])
    with patch("custom_components.rosdomofon.deepface_client.represent_faces",
               return_value=[{"embedding": [1.0, 0.0], "facial_area": None, "confidence": 0.9}]):
        resp = await manager._handle_webhook(hass, wh_id, upload_req)
    assert json.loads(resp.body)["status"] == "ok"
    assert store.photo_count("Ivan") == 1

    photo_id = store.photos("Ivan")[0]["id"]

    # Удаление фото по id
    del_req = MagicMock(method="POST", content_type="application/json")
    del_req.json = AsyncMock(return_value={"action": "delete", "id": photo_id})
    resp = await manager._handle_webhook(hass, wh_id, del_req)
    data = json.loads(resp.body)
    assert data["count"] == 0
    assert store.photo_count("Ivan") == 0


@pytest.mark.asyncio
async def test_enroll_without_camera(hass: HomeAssistant):
    """Ссылка без камеры: страница без предпросмотра, съёмка недоступна, загрузка работает."""
    store = FaceStore(hass)
    await store.async_load()
    manager = _manager(hass, store)
    wh_id = _make_link(hass, manager, person="Ivan", camera=None)

    # Страница без блока камеры
    page_req = MagicMock(method="GET", query={})
    resp = await manager._handle_webhook(hass, wh_id, page_req)
    assert resp.status == 200
    assert 'id="preview"' not in resp.text
    assert "Загрузить фото" in resp.text

    # Съёмка недоступна без камеры
    cap_req = MagicMock(method="POST", content_type="application/json")
    cap_req.json = AsyncMock(return_value={"action": "capture"})
    resp2 = await manager._handle_webhook(hass, wh_id, cap_req)
    assert json.loads(resp2.body)["status"] == "error"

    # Загрузка фото работает
    up_req = _upload_request([b"uploaded"])
    with patch("custom_components.rosdomofon.deepface_client.represent_faces",
               return_value=[{"embedding": [1.0, 0.0], "facial_area": None, "confidence": 0.9}]):
        resp3 = await manager._handle_webhook(hass, wh_id, up_req)
    assert json.loads(resp3.body)["status"] == "ok"
    assert store.photo_count("Ivan") == 1


@pytest.mark.asyncio
async def test_enroll_upload_multiple(hass: HomeAssistant):
    """Загрузка нескольких фото за один запрос по ссылке."""
    store = FaceStore(hass)
    await store.async_load()
    manager = _manager(hass, store)
    wh_id = _make_link(hass, manager, person="Ivan", camera=None)

    up_req = _upload_request([b"a", b"b", b"c"])
    seq = [
        [{"embedding": [1.0, 0.0], "facial_area": None, "confidence": 0.9}],
        [],
        [{"embedding": [0.0, 1.0], "facial_area": None, "confidence": 0.9}],
    ]
    with patch("custom_components.rosdomofon.deepface_client.represent_faces", side_effect=seq):
        resp = await manager._handle_webhook(hass, wh_id, up_req)

    data = json.loads(resp.body)
    assert data["status"] == "ok"
    assert "добавлено 2" in data["message"]
    assert store.photo_count("Ivan") == 2


@pytest.mark.asyncio
async def test_enroll_unbound_uses_rosdomofon_cameras(hass: HomeAssistant):
    """Единый механизм: ссылка без камеры предлагает съёмку с камер Росдомофон."""
    from homeassistant.helpers import entity_registry as er

    reg = er.async_get(hass)
    reg.async_get_or_create("camera", DOMAIN, "cam1", suggested_object_id="dvor")

    store = FaceStore(hass)
    await store.async_load()
    manager = _manager(hass, store)
    wh_id = _make_link(hass, manager, person="Ivan", camera=None)

    # Страница показывает предпросмотр (камера доступна)
    page_req = MagicMock(method="GET", query={})
    page = await manager._handle_webhook(hass, wh_id, page_req)
    assert 'id="preview"' in page.text

    # Съёмка с найденной камеры работает
    cap_req = MagicMock(method="POST", content_type="application/json")
    cap_req.json = AsyncMock(return_value={"action": "capture", "camera": "camera.dvor"})
    image = MagicMock(content=b"frame", content_type="image/jpeg")
    with patch("custom_components.rosdomofon.enroll.async_get_image", AsyncMock(return_value=image)), \
         patch("custom_components.rosdomofon.deepface_client.represent_faces",
               return_value=[{"embedding": [1.0, 0.0], "facial_area": None, "confidence": 0.9}]):
        resp = await manager._handle_webhook(hass, wh_id, cap_req)
    assert json.loads(resp.body)["status"] == "ok"
    assert store.photo_count("Ivan") == 1

    # Чужая камера (не Росдомофон) отклоняется
    bad_req = MagicMock(method="POST", content_type="application/json")
    bad_req.json = AsyncMock(return_value={"action": "capture", "camera": "camera.other"})
    resp2 = await manager._handle_webhook(hass, wh_id, bad_req)
    assert json.loads(resp2.body)["status"] == "error"


@pytest.mark.asyncio
async def test_enroll_snapshot(hass: HomeAssistant):
    manager = _manager(hass, FaceStore(hass))
    wh_id = _make_link(hass, manager)

    request = MagicMock(method="GET", query={"snapshot": "1", "camera": "camera.dvor"})
    image = MagicMock(content=b"jpegbytes", content_type="image/jpeg")
    with patch("custom_components.rosdomofon.enroll.async_get_image", AsyncMock(return_value=image)):
        resp = await manager._handle_webhook(hass, wh_id, request)
    assert resp.status == 200
    assert resp.body == b"jpegbytes"
