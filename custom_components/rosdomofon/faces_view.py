"""
Страница просмотра и управления эталонными лицами.

Показывает всех сохранённых людей с эскизами фото. Позволяет удалить отдельное
фото или человека целиком. Доступ защищён той же HMAC-подписью, что и прокси
потоков (см. stream_proxy) — ссылку с подписью пользователь открывает из меню
«Люди» в настройках интеграции.
"""

import html
import logging

from aiohttp import web
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant

from . import deepface_client
from .const import (
    CONF_DEEPFACE_URL,
    CONF_DETECTOR,
    CONF_MODEL,
    DATA_FACE_STORE,
    DEFAULT_DETECTOR,
    DEFAULT_MODEL,
    DOMAIN,
)
from .share import ExternalURLNotAvailable
from .stream_proxy import _validate_proxy_request, sign_proxy_path

_LOGGER = logging.getLogger(__name__)

_FACES_PATH = "/api/rosdomofon/faces"

# Путь боковой панели HA, в которую встроена галерея лиц
FACES_PANEL_URL_PATH = "rosdomofon-faces"


def async_register_faces_panel(hass: HomeAssistant) -> None:
    """Регистрирует (обновляя) боковую панель HA со страницей управления лицами.

    Панель — iframe того же origin с подписанным путём, поэтому открывается прямо
    в интерфейсе Home Assistant, а не отдельной ссылкой. Только для админов.
    """
    from homeassistant.components import frontend

    async_remove_faces_panel(hass)
    signed_path = sign_proxy_path(hass, _FACES_PATH)  # относительный путь с подписью
    frontend.async_register_built_in_panel(
        hass,
        component_name="iframe",
        sidebar_title="Лица (Росдомофон)",
        sidebar_icon="mdi:face-recognition",
        frontend_url_path=FACES_PANEL_URL_PATH,
        config={"url": signed_path},
        require_admin=True,
    )
    _LOGGER.info("Боковая панель управления лицами зарегистрирована")


def async_remove_faces_panel(hass: HomeAssistant) -> None:
    """Убирает боковую панель (если была)."""
    from homeassistant.components import frontend

    try:
        frontend.async_remove_panel(hass, FACES_PANEL_URL_PATH)
    except Exception:  # noqa: BLE001 — панели могло не быть
        pass


def _face_store(hass: HomeAssistant):
    return hass.data.get(DOMAIN, {}).get(DATA_FACE_STORE)


def _enroll_manager(hass: HomeAssistant):
    for data in hass.data.get(DOMAIN, {}).values():
        if isinstance(data, dict) and "enroll_manager" in data:
            return data["enroll_manager"]
    return None


def _upload_summary(name: str, added: int, no_face: int, errors: int) -> tuple[str, str]:
    """Строит статус и сообщение по результату загрузки нескольких фото."""
    parts = []
    if added:
        parts.append(f"добавлено {added}")
    if no_face:
        parts.append(f"без лица {no_face}")
    if errors:
        parts.append(f"ошибок {errors}")
    status = "ok" if added else "error"
    detail = ", ".join(parts) if parts else "нет подходящих файлов"
    return status, f"«{name}»: {detail}"


def _deepface_config(hass: HomeAssistant) -> dict | None:
    """Возвращает конфиг DeepFace из первой настроенной записи (или None)."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        url = entry.options.get(CONF_DEEPFACE_URL)
        if url:
            return {
                "url": url,
                "model": entry.options.get(CONF_MODEL, DEFAULT_MODEL),
                "detector": entry.options.get(CONF_DETECTOR, DEFAULT_DETECTOR),
            }
    return None


class RosdomofonFacesView(HomeAssistantView):
    """HTTP View: просмотр и управление эталонными лицами."""

    url = _FACES_PATH
    name = "api:rosdomofon:faces"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass

    async def get(self, request: web.Request) -> web.Response:
        if not _validate_proxy_request(self.hass, request):
            return web.Response(status=401, text="Invalid signature")
        return web.Response(
            text=_render_page(self._people_payload()["people"]),
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def post(self, request: web.Request) -> web.Response:
        if not _validate_proxy_request(self.hass, request):
            return web.json_response({"status": "error"}, status=401)

        store = _face_store(self.hass)
        if store is None:
            return web.json_response({"status": "error", "message": "нет хранилища"}, status=500)

        # Добавление фото приходит multipart-формой (имя + файл)
        if (request.content_type or "").startswith("multipart/"):
            return await self._handle_add(request)

        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        action = body.get("action")
        name = (body.get("name") or "").strip()

        if action == "delete_photo" and name and body.get("id"):
            await store.async_remove_photo(name, body["id"])
        elif action == "delete_person" and name:
            await store.async_remove_person(name)
        elif action == "create_person":
            if not name:
                return web.json_response(self._result("error", "Введите имя человека."))
            await store.async_create_person(name)
            return web.json_response(self._result("ok", f"Человек «{name}» создан."))
        elif action == "enroll_link":
            if not name:
                return web.json_response(self._result("error", "Введите имя человека."))
            return await self._handle_enroll_link(name)
        elif action != "list":
            return web.json_response({"status": "error", "message": "неизвестное действие"}, status=400)

        return web.json_response(self._people_payload())

    async def _handle_enroll_link(self, name: str) -> web.Response:
        """Создаёт временную ссылку самозагрузки фото для человека (без камеры)."""
        mgr = _enroll_manager(self.hass)
        if mgr is None:
            return web.json_response(self._result("error", "Интеграция не настроена."))
        try:
            url = mgr.generate(None, name)
        except ExternalURLNotAvailable:
            return web.json_response(
                self._result(
                    "error",
                    "Для ссылки нужен внешний доступ (External URL или Nabu Casa).",
                )
            )
        payload = self._result("ok", f"Ссылка для «{name}» создана.")
        payload["link"] = url
        payload["link_person"] = name
        return web.json_response(payload)

    def _result(self, status: str, message: str) -> dict:
        payload = self._people_payload()
        payload["status"] = status
        payload["message"] = message
        return payload

    async def _handle_add(self, request: web.Request) -> web.Response:
        """Добавляет человеку одно или несколько загруженных фото."""
        try:
            post = await request.post()
        except Exception as exc:  # noqa: BLE001
            return self._add_result("error", f"Не удалось прочитать файлы: {exc}")

        name = (post.get("name") or "").strip()
        if not name:
            return self._add_result("error", "Введите имя человека.")

        files = [f for f in post.getall("photo") if hasattr(f, "file")]
        if not files:
            return self._add_result("error", "Файлы не получены.")

        cfg = _deepface_config(self.hass)
        if not cfg:
            return self._add_result("error", "Сервис DeepFace не настроен.")

        store = _face_store(self.hass)
        added = no_face = errors = 0
        for field in files:
            data = field.file.read()
            if not data:
                continue
            try:
                await store.async_add_person(
                    name, data, cfg["url"], cfg["model"], cfg["detector"]
                )
                added += 1
            except deepface_client.NoFaceError:
                no_face += 1
            except deepface_client.DeepFaceError as exc:
                _LOGGER.error("Ошибка добавления фото для «%s»: %s", name, exc)
                errors += 1

        status, message = _upload_summary(name, added, no_face, errors)
        return self._add_result(status, message)

    def _add_result(self, status: str, message: str) -> web.Response:
        payload = self._people_payload()
        payload["status"] = status
        payload["message"] = message
        return web.json_response(payload)

    def _people_payload(self) -> dict:
        store = _face_store(self.hass)
        if store is None:
            return {"status": "ok", "people": []}
        people = [
            {"name": name, "photos": store.photos(name), "count": store.photo_count(name)}
            for name in store.people
        ]
        return {"status": "ok", "people": people}


def _thumb_html(photo: dict) -> str:
    pid = html.escape(photo["id"], quote=True)
    if photo.get("photo"):
        inner = f'<img src="data:image/jpeg;base64,{photo["photo"]}" alt="лицо">'
    else:
        inner = '<div class="noimg">без фото</div>'
    return f'<div class="thumb" data-id="{pid}">{inner}<button class="del" title="Удалить фото">✕</button></div>'


def _person_card(person: dict) -> str:
    name_e = html.escape(person["name"], quote=True)
    thumbs = "".join(_thumb_html(p) for p in person["photos"])
    grid_inner = thumbs or '<div class="empty-grid">Фото пока нет</div>'
    return (
        f'<div class="card person" data-name="{name_e}">'
        f'<div class="phead"><span class="pname">{name_e}</span>'
        f'<span class="chip">{person["count"]} фото</span></div>'
        f'<div class="grid">{grid_inner}</div>'
        f'<div class="actions">'
        f'<button class="btn primary addphoto">Добавить фото</button>'
        f'<button class="btn enroll">Ссылка для загрузки</button>'
        f'<button class="btn danger delperson">Удалить</button>'
        f'<input type="file" class="pfile" accept="image/*" multiple hidden></div>'
        f"</div>"
    )


def _render_page(people: list[dict]) -> str:
    cards = "".join(_person_card(p) for p in people) or (
        '<div class="card empty">Пока нет добавленных лиц. Создайте человека выше.</div>'
    )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Росдомофон · Лица</title>
<style>
  :root {{
    --bg:#f2f3f5; --card:#fff; --text:#212121; --secondary:#5f6368; --primary:#03a9f4;
    --on-primary:#fff; --divider:#e3e5e8; --danger:#db4437; --chip-bg:#eceff1;
    --shadow:0 2px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.06);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:#111214; --card:#1c1e22; --text:#e3e3e3; --secondary:#9aa0a6; --divider:#2a2d31;
      --chip-bg:#2a2d31; --shadow:0 2px 6px rgba(0,0,0,.4);
    }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font-family:Roboto,"Helvetica Neue",-apple-system,Segoe UI,sans-serif; }}
  .appbar {{ background:var(--primary); color:var(--on-primary); height:56px; padding:0 16px;
    display:flex; align-items:center; position:sticky; top:0; z-index:2; box-shadow:var(--shadow); }}
  .appbar h1 {{ font-size:20px; font-weight:400; margin:0; }}
  .wrap {{ max-width:720px; margin:0 auto; padding:16px; }}
  .card {{ background:var(--card); border-radius:12px; box-shadow:var(--shadow);
    padding:16px; margin-bottom:16px; }}
  .formrow {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
  input[type=text] {{ flex:1; min-width:180px; padding:10px 12px; border-radius:8px;
    border:1px solid var(--divider); background:var(--bg); color:var(--text); font-size:14px; }}
  .btn {{ border:none; border-radius:8px; padding:9px 14px; font-size:14px; font-weight:500;
    cursor:pointer; background:var(--chip-bg); color:var(--primary); }}
  .btn.primary {{ background:var(--primary); color:var(--on-primary); }}
  .btn.danger {{ background:transparent; color:var(--danger); }}
  .btn:active {{ opacity:.85; }}
  .hint {{ color:var(--secondary); font-size:12px; margin-top:8px; }}
  .status {{ min-height:1.2em; font-size:13px; margin:0 2px 14px; }}
  .status.ok {{ color:#2e7d32; }} .status.err {{ color:var(--danger); }}
  @media (prefers-color-scheme: dark) {{ .status.ok {{ color:#81c995; }} }}
  .phead {{ display:flex; align-items:center; gap:10px; margin-bottom:12px; }}
  .pname {{ font-weight:500; font-size:16px; }}
  .chip {{ font-size:12px; color:var(--secondary); background:var(--chip-bg);
    border-radius:12px; padding:2px 10px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(88px,1fr));
    gap:8px; margin-bottom:12px; }}
  .thumb {{ position:relative; aspect-ratio:1; border-radius:8px; overflow:hidden; background:var(--chip-bg); }}
  .thumb img {{ width:100%; height:100%; object-fit:cover; }}
  .thumb .noimg {{ display:flex; align-items:center; justify-content:center; height:100%;
    font-size:11px; color:var(--secondary); }}
  .del {{ position:absolute; top:4px; right:4px; width:22px; height:22px; border:none;
    border-radius:50%; background:rgba(0,0,0,.55); color:#fff; cursor:pointer; font-size:12px; line-height:1; }}
  .empty-grid {{ color:var(--secondary); font-size:13px; padding:8px 0 4px; }}
  .actions {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .empty {{ color:var(--secondary); text-align:center; }}
  .linktitle {{ font-weight:500; margin-bottom:6px; }}
  .linkrow {{ display:flex; gap:8px; }}
  .linkhint {{ color:var(--secondary); font-size:12px; margin-top:8px; }}
</style></head>
<body>
<div class="appbar"><h1>Росдомофон · Лица</h1></div>
<div class="wrap">
  <div class="card">
    <div class="formrow">
      <input type="text" id="newname" placeholder="Имя человека">
      <button class="btn" id="createnew">Создать без фото</button>
      <button class="btn primary" id="uploadnew">Загрузить фото</button>
      <input type="file" id="newfile" accept="image/*" multiple hidden>
    </div>
    <div class="hint">Создайте человека без фото или сразу загрузите одно/несколько фото. Фото автоматически обрезается по лицу.</div>
  </div>

  <div class="status" id="status"></div>

  <div class="card" id="linkbox" hidden>
    <div class="linktitle" id="linktitle"></div>
    <div class="linkrow">
      <input type="text" id="linkurl" readonly>
      <button class="btn primary" id="copylink">Копировать</button>
    </div>
    <div class="linkhint">Откройте ссылку на телефоне нужного человека — он сам загрузит свои фото. ⚠️ Добавленное лицо получает доступ к двери; ссылка временная.</div>
  </div>

  <div id="root">{cards}</div>
</div>
<script>
  const base = window.location.pathname + window.location.search;
  const root = document.getElementById('root');
  const status = document.getElementById('status');
  const newname = document.getElementById('newname');
  const newfile = document.getElementById('newfile');
  const linkbox = document.getElementById('linkbox');
  const linktitle = document.getElementById('linktitle');
  const linkurl = document.getElementById('linkurl');

  function esc(s) {{ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }}
  function setStatus(msg, ok) {{ status.textContent = msg || ''; status.className = 'status ' + (ok ? 'ok' : 'err'); }}

  function personCard(p) {{
    const thumbs = (p.photos||[]).map(ph => {{
      const inner = ph.photo
        ? '<img src="data:image/jpeg;base64,' + ph.photo + '" alt="лицо">'
        : '<div class="noimg">без фото</div>';
      return '<div class="thumb" data-id="' + esc(ph.id) + '">' + inner +
             '<button class="del" title="Удалить фото">✕</button></div>';
    }}).join('') || '<div class="empty-grid">Фото пока нет</div>';
    return '<div class="card person" data-name="' + esc(p.name) + '"><div class="phead">' +
      '<span class="pname">' + esc(p.name) + '</span>' +
      '<span class="chip">' + p.count + ' фото</span></div>' +
      '<div class="grid">' + thumbs + '</div>' +
      '<div class="actions">' +
      '<button class="btn primary addphoto">Добавить фото</button>' +
      '<button class="btn enroll">Ссылка для загрузки</button>' +
      '<button class="btn danger delperson">Удалить</button>' +
      '<input type="file" class="pfile" accept="image/*" multiple hidden></div></div>';
  }}

  function render(people) {{
    root.innerHTML = people.length
      ? people.map(personCard).join('')
      : '<div class="card empty">Пока нет добавленных лиц. Создайте человека выше.</div>';
  }}

  async function send(payload) {{
    const r = await fetch(base, {{ method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify(payload) }});
    const d = await r.json();
    if (d.people) render(d.people);
    if (d.message !== undefined) setStatus(d.message, d.status === 'ok');
    return d;
  }}

  async function uploadPhoto(name, files) {{
    if (!name) {{ setStatus('Введите имя человека.', false); return; }}
    if (!files || !files.length) return;
    setStatus('Загружаю ' + files.length + ' фото…', true);
    const fd = new FormData(); fd.append('name', name);
    for (const f of files) fd.append('photo', f);
    try {{
      const r = await fetch(base, {{ method:'POST', body: fd }});
      const d = await r.json();
      setStatus(d.message, d.status === 'ok');
      if (d.people) render(d.people);
    }} catch (e) {{ setStatus('Ошибка загрузки.', false); }}
  }}

  function showLink(url, person) {{
    linktitle.textContent = 'Ссылка для «' + person + '»';
    linkurl.value = url;
    linkbox.hidden = false;
    if (navigator.clipboard) navigator.clipboard.writeText(url).catch(() => {{}});
    linkbox.scrollIntoView({{ behavior:'smooth', block:'nearest' }});
  }}

  document.getElementById('createnew').addEventListener('click', async () => {{
    const name = newname.value.trim();
    if (!name) {{ setStatus('Введите имя человека.', false); return; }}
    const d = await send({{action:'create_person', name:name}});
    if (d.status === 'ok') newname.value = '';
  }});
  document.getElementById('uploadnew').addEventListener('click', () => {{
    if (!newname.value.trim()) {{ setStatus('Введите имя человека.', false); return; }}
    newfile.click();
  }});
  newfile.addEventListener('change', e => {{
    if (e.target.files.length) uploadPhoto(newname.value.trim(), e.target.files);
    e.target.value = '';
  }});
  document.getElementById('copylink').addEventListener('click', () => {{
    linkurl.select();
    if (navigator.clipboard) navigator.clipboard.writeText(linkurl.value).catch(() => {{}});
    setStatus('Ссылка скопирована.', true);
  }});

  root.addEventListener('click', async e => {{
    const person = e.target.closest('.person');
    if (!person) return;
    const name = person.dataset.name;
    if (e.target.classList.contains('del')) {{
      const thumb = e.target.closest('.thumb');
      if (thumb) send({{action:'delete_photo', name:name, id:thumb.dataset.id}});
    }} else if (e.target.classList.contains('delperson')) {{
      if (confirm('Удалить «' + name + '» и все его фото?'))
        send({{action:'delete_person', name:name}});
    }} else if (e.target.classList.contains('addphoto')) {{
      const f = person.querySelector('.pfile');
      if (f) f.click();
    }} else if (e.target.classList.contains('enroll')) {{
      const d = await send({{action:'enroll_link', name:name}});
      if (d.link) showLink(d.link, d.link_person || name);
    }}
  }});
  root.addEventListener('change', e => {{
    if (e.target.classList.contains('pfile')) {{
      const person = e.target.closest('.person');
      if (person && e.target.files.length) uploadPhoto(person.dataset.name, e.target.files);
      e.target.value = '';
    }}
  }});
</script>
</body></html>"""


def setup_faces_view(hass: HomeAssistant) -> None:
    """Регистрирует HTTP-view галереи лиц (один раз на домен)."""
    hass.http.register_view(RosdomofonFacesView(hass))
    _LOGGER.info("Галерея эталонных лиц зарегистрирована: %s", _FACES_PATH)
