"""
Самостоятельное добавление лица по ссылке (Enroll Link).

Генерирует временную ссылку (webhook) на страницу с предпросмотром камеры, где
человек может снять своё лицо с камеры или загрузить фото с телефона. Найденное
лицо автоматически обрезается и добавляется в эталоны выбранного человека
(несколько фото на человека поддерживается). Ссылка живёт ограниченное время (TTL).

ВНИМАНИЕ по безопасности: по этой ссылке любой может добавить лицо в список
«своих» — фактически выдать себе доступ на открытие двери. Ссылка защищена
только неугадываемым webhook_id и сроком жизни. Не пересылайте её посторонним.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiohttp import hdrs, web
from homeassistant.components import webhook
from homeassistant.components.camera import async_get_image
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import network
from homeassistant.helpers.event import async_call_later

from . import deepface_client
from .const import DOMAIN, ENROLL_LINK_DEFAULT_TTL_HOURS, ENROLL_LINK_WEBHOOK_PREFIX
from .face_store import FaceStore
from .share import ExternalURLNotAvailable

_LOGGER = logging.getLogger(__name__)


@dataclass
class EnrollLink:
    """Одна временная ссылка для добавления лица человеку."""

    webhook_id: str
    camera_entity_id: str
    person_name: str
    created_at: float = field(default_factory=time.time)
    ttl_hours: float = ENROLL_LINK_DEFAULT_TTL_HOURS
    cancel_expiry: Any = None

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl_hours * 3600

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class EnrollLinkManager:
    """Управляет временными ссылками для добавления лиц."""

    def __init__(
        self, hass: HomeAssistant, face_store: FaceStore, deepface_config: dict
    ) -> None:
        self.hass = hass
        self._face_store = face_store
        # {"url": str, "model": str, "detector": str}
        self._config = dict(deepface_config)
        self._links: dict[str, EnrollLink] = {}

    def update_config(self, deepface_config: dict) -> None:
        self._config = dict(deepface_config)

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    def generate(
        self,
        camera_entity_id: str,
        person_name: str,
        ttl_hours: float = ENROLL_LINK_DEFAULT_TTL_HOURS,
    ) -> str:
        """Создаёт ссылку для добавления лица и возвращает полный URL."""
        try:
            external_url = network.get_url(
                self.hass,
                allow_internal=False,
                allow_ip=True,
                prefer_external=True,
                prefer_cloud=True,
            )
        except network.NoURLAvailableError as exc:
            raise ExternalURLNotAvailable from exc

        webhook_id = f"{ENROLL_LINK_WEBHOOK_PREFIX}{uuid.uuid4().hex}"
        webhook.async_register(
            self.hass,
            domain=DOMAIN,
            name=f"Enroll link: {person_name}",
            webhook_id=webhook_id,
            handler=self._handle_webhook,
            local_only=False,
            allowed_methods=(hdrs.METH_GET, hdrs.METH_POST),
        )

        link = EnrollLink(
            webhook_id=webhook_id,
            camera_entity_id=camera_entity_id,
            person_name=person_name,
            ttl_hours=ttl_hours,
        )
        link.cancel_expiry = async_call_later(
            self.hass, ttl_hours * 3600, self._make_expiry_callback(webhook_id)
        )
        self._links[webhook_id] = link

        _LOGGER.info(
            "Сгенерирована ссылка добавления лица для «%s» (камера %s, TTL %s ч)",
            person_name,
            camera_entity_id,
            ttl_hours,
        )
        return f"{external_url}/api/webhook/{webhook_id}"

    def revoke_all(self) -> None:
        """Отзывает все активные ссылки (при выгрузке интеграции)."""
        for wh_id in list(self._links):
            link = self._links.pop(wh_id, None)
            if link and link.cancel_expiry:
                link.cancel_expiry()
            try:
                webhook.async_unregister(self.hass, wh_id)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Webhook handler
    # ------------------------------------------------------------------

    async def _handle_webhook(
        self, hass: HomeAssistant, webhook_id: str, request: web.Request
    ) -> web.Response:
        link = self._links.get(webhook_id)
        if link is None or link.is_expired:
            return web.Response(
                text=_simple_page(
                    "Ссылка недействительна",
                    "Срок действия ссылки истёк или она была отозвана.",
                ),
                content_type="text/html",
                status=410,
            )

        # GET: снимок для предпросмотра или сама страница
        if request.method == hdrs.METH_GET:
            if request.query.get("snapshot"):
                return await self._serve_snapshot(link.camera_entity_id)
            return self._serve_page(link)

        # POST: загрузка файла (multipart) или действие в JSON
        if (request.content_type or "").startswith("multipart/"):
            return await self._handle_upload(link, request)
        return await self._handle_action(link, request)

    async def _serve_snapshot(self, camera: str) -> web.Response:
        try:
            image = await async_get_image(self.hass, camera, timeout=10)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Не удалось получить кадр для предпросмотра %s: %s", camera, exc)
            return web.Response(status=503, text="camera unavailable")
        return web.Response(
            body=image.content,
            content_type=getattr(image, "content_type", None) or "image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    async def _handle_upload(
        self, link: EnrollLink, request: web.Request
    ) -> web.Response:
        try:
            post = await request.post()
        except Exception as exc:  # noqa: BLE001
            return web.json_response(
                {"status": "error", "message": f"Не удалось прочитать файл: {exc}"},
                status=400,
            )
        field_value = post.get("photo")
        image_bytes = None
        if field_value is not None and hasattr(field_value, "file"):
            image_bytes = field_value.file.read()
        if not image_bytes:
            return web.json_response(
                {"status": "error", "message": "Файл не получен."}, status=400
            )
        return web.json_response(await self._enroll(link.person_name, image_bytes))

    async def _handle_action(
        self, link: EnrollLink, request: web.Request
    ) -> web.Response:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        action = body.get("action")

        if action == "capture":
            try:
                image = await async_get_image(
                    self.hass, link.camera_entity_id, timeout=10
                )
            except Exception:  # noqa: BLE001
                return web.json_response(
                    {
                        "status": "error",
                        "message": "Камера недоступна. Попробуйте ещё раз.",
                    }
                )
            return web.json_response(
                await self._enroll(link.person_name, image.content)
            )

        if action == "delete":
            photo_id = body.get("id")
            if photo_id:
                await self._face_store.async_remove_photo(link.person_name, photo_id)
            return web.json_response(self._list_payload(link.person_name))

        if action == "list":
            return web.json_response(self._list_payload(link.person_name))

        return web.json_response(
            {"status": "error", "message": "Неизвестное действие."}, status=400
        )

    async def _enroll(self, person: str, image_bytes: bytes) -> dict:
        """Добавляет лицо из изображения человеку. Возвращает JSON-словарь ответа."""
        url = self._config.get("url")
        if not url:
            return {"status": "error", "message": "Сервис DeepFace не настроен."}
        try:
            thumb = await self._face_store.async_add_person(
                person,
                image_bytes,
                url,
                self._config.get("model"),
                self._config.get("detector"),
            )
        except deepface_client.NoFaceError:
            return {
                "status": "error",
                "message": "Лицо не найдено. Встаньте ровно к камере, хорошее освещение.",
            }
        except deepface_client.DeepFaceError as exc:
            _LOGGER.error("Ошибка добавления лица для «%s»: %s", person, exc)
            return {"status": "error", "message": "Ошибка распознавания. Попробуйте ещё раз."}

        count = self._face_store.photo_count(person)
        return {
            "status": "ok",
            "message": f"Фото добавлено. Всего у «{person}»: {count}.",
            "photo": base64.b64encode(thumb).decode("ascii"),
            "count": count,
        }

    def _list_payload(self, person: str) -> dict:
        return {
            "status": "ok",
            "photos": self._face_store.photos(person),
            "count": self._face_store.photo_count(person),
        }

    def _serve_page(self, link: EnrollLink) -> web.Response:
        state = self.hass.states.get(link.camera_entity_id)
        camera_name = state.name if state else link.camera_entity_id
        remaining = max(0, link.expires_at - time.time())
        return web.Response(
            text=_enroll_page(
                person=link.person_name,
                camera_name=camera_name,
                remaining_h=int(remaining // 3600),
                remaining_m=int((remaining % 3600) // 60),
                photos=self._face_store.photos(link.person_name),
            ),
            content_type="text/html",
        )

    def _make_expiry_callback(self, webhook_id: str):
        @callback
        def _expire(_now) -> None:
            _LOGGER.info("Ссылка добавления лица %s истекла, удаляем", webhook_id)
            self._links.pop(webhook_id, None)
            try:
                webhook.async_unregister(self.hass, webhook_id)
            except ValueError:
                pass

        return _expire


def _thumb_html(photo: dict) -> str:
    """HTML одного эскиза с кнопкой удаления (id — в data-атрибуте)."""
    pid = html.escape(photo["id"], quote=True)
    if photo.get("photo"):
        src = f"data:image/jpeg;base64,{photo['photo']}"
        img = f'<img src="{src}" alt="лицо">'
    else:
        img = '<div class="noimg">без фото</div>'
    return (
        f'<div class="thumb" data-id="{pid}">{img}'
        f'<button class="del">✕</button></div>'
    )


def _enroll_page(
    person: str,
    camera_name: str,
    remaining_h: int,
    remaining_m: int,
    photos: list[dict],
) -> str:
    """Страница добавления лица: предпросмотр камеры, съёмка, загрузка, эскизы."""
    person_e = html.escape(person, quote=True)
    camera_e = html.escape(camera_name, quote=True)
    thumbs = "".join(_thumb_html(p) for p in photos)
    # Безопасно прокидываем имя в JS-строку
    person_json = json.dumps(person)

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Добавление лица · {person_e}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif;
         background:linear-gradient(160deg,#8fb7ff,#c7a4ff); color:#fff; min-height:100vh; }}
  .wrap {{ max-width:480px; margin:0 auto; padding:20px 16px 40px; }}
  h1 {{ font-size:1.15rem; margin:6px 0; }}
  .sub {{ font-size:.85rem; opacity:.9; margin-bottom:16px; }}
  .preview {{ width:100%; aspect-ratio:4/3; background:#0004; border-radius:16px;
             object-fit:cover; display:block; }}
  .btns {{ display:flex; gap:10px; margin:14px 0; }}
  button.act {{ flex:1; border:none; border-radius:14px; padding:14px; font-size:1rem;
               font-weight:700; cursor:pointer; background:#fff; color:#7b5cff; }}
  button.act:disabled {{ opacity:.6; cursor:default; }}
  label.upload {{ flex:1; border:none; border-radius:14px; padding:14px; font-size:1rem;
                 font-weight:700; cursor:pointer; background:#ffffff22; color:#fff;
                 text-align:center; border:1px solid #ffffff55; }}
  #file {{ display:none; }}
  .status {{ min-height:1.3em; font-size:.9rem; margin:6px 0 14px; }}
  .ok {{ color:#d6ffe8; }} .err {{ color:#ffdede; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(84px,1fr)); gap:8px; }}
  .thumb {{ position:relative; aspect-ratio:1; border-radius:10px; overflow:hidden; background:#0003; }}
  .thumb img {{ width:100%; height:100%; object-fit:cover; }}
  .thumb .noimg {{ display:flex; align-items:center; justify-content:center; height:100%;
                  font-size:.7rem; opacity:.7; }}
  .del {{ position:absolute; top:3px; right:3px; width:22px; height:22px; border:none;
         border-radius:50%; background:#000a; color:#fff; cursor:pointer; font-size:.8rem; }}
  .count {{ font-size:.85rem; opacity:.9; margin:16px 0 8px; }}
  .warn {{ font-size:.75rem; opacity:.8; margin-top:20px; }}
</style></head>
<body><div class="wrap">
  <h1>Добавление лица: {person_e}</h1>
  <div class="sub">Камера: {camera_e} · ссылка активна ещё {remaining_h}ч {remaining_m}м</div>

  <img class="preview" id="preview" alt="предпросмотр камеры">
  <div class="btns">
    <button class="act" id="cap">📸 Снять с камеры</button>
    <label class="upload">⬆️ Загрузить фото<input type="file" id="file" accept="image/*"></label>
  </div>
  <div class="status" id="status">Встаньте ровно к камере и нажмите «Снять», либо загрузите фото.</div>

  <div class="count" id="count">Фото у «{person_e}»: {len(photos)}</div>
  <div class="grid" id="grid">{thumbs}</div>

  <div class="warn">⚠️ Добавленное лицо получает доступ на открытие двери. Не передавайте ссылку посторонним.</div>
</div>
<script>
  const base = window.location.pathname;
  const preview = document.getElementById('preview');
  const status = document.getElementById('status');
  const cap = document.getElementById('cap');
  const file = document.getElementById('file');
  const grid = document.getElementById('grid');
  const count = document.getElementById('count');
  const person = {person_json};

  function poll() {{
    const img = new Image();
    img.onload = () => {{ preview.src = img.src; setTimeout(poll, 2500); }};
    img.onerror = () => setTimeout(poll, 4000);
    img.src = base + '?snapshot=1&t=' + Date.now();
  }}
  poll();

  function setStatus(msg, ok) {{ status.textContent = msg; status.className = 'status ' + (ok ? 'ok' : 'err'); }}

  function esc(s) {{ const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }}

  function renderList(data) {{
    count.textContent = 'Фото у «' + person + '»: ' + data.count;
    grid.innerHTML = (data.photos || []).map(p => {{
      const inner = p.photo
        ? '<img src="data:image/jpeg;base64,' + p.photo + '">'
        : '<div class="noimg">без фото</div>';
      return '<div class="thumb" data-id="' + esc(p.id) + '">' + inner +
             '<button class="del">✕</button></div>';
    }}).join('');
  }}

  async function refresh() {{
    try {{
      const r = await fetch(base, {{ method:'POST', headers:{{'Content-Type':'application/json'}},
        body: JSON.stringify({{action:'list'}}) }});
      renderList(await r.json());
    }} catch (e) {{}}
  }}

  async function capture() {{
    cap.disabled = true; setStatus('Снимаю и распознаю…', true);
    try {{
      const r = await fetch(base, {{ method:'POST', headers:{{'Content-Type':'application/json'}},
        body: JSON.stringify({{action:'capture'}}) }});
      const d = await r.json();
      setStatus(d.message, d.status === 'ok');
      if (d.status === 'ok') await refresh();
    }} catch (e) {{ setStatus('Ошибка соединения.', false); }}
    cap.disabled = false;
  }}

  async function upload(f) {{
    setStatus('Загружаю и распознаю…', true);
    const fd = new FormData(); fd.append('photo', f);
    try {{
      const r = await fetch(base, {{ method:'POST', body: fd }});
      const d = await r.json();
      setStatus(d.message, d.status === 'ok');
      if (d.status === 'ok') await refresh();
    }} catch (e) {{ setStatus('Ошибка загрузки.', false); }}
  }}

  async function delPhoto(id) {{
    try {{
      const r = await fetch(base, {{ method:'POST', headers:{{'Content-Type':'application/json'}},
        body: JSON.stringify({{action:'delete', id:id}}) }});
      renderList(await r.json());
    }} catch (e) {{}}
  }}

  cap.addEventListener('click', capture);
  file.addEventListener('change', e => {{ if (e.target.files[0]) upload(e.target.files[0]); }});
  grid.addEventListener('click', e => {{
    if (e.target.classList.contains('del')) {{
      const thumb = e.target.closest('.thumb');
      if (thumb) delPhoto(thumb.dataset.id);
    }}
  }});
</script>
</body></html>"""


def _simple_page(title: str, message: str) -> str:
    title = html.escape(title, quote=True)
    message = html.escape(message, quote=True)
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0f172a;color:#fff}}
.card{{max-width:420px;padding:28px;border-radius:20px;background:#ffffff14;text-align:center}}</style>
</head><body><div class="card"><h2>{title}</h2><p>{message}</p></div></body></html>"""
