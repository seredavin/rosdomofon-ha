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
        name = body.get("name")

        if action == "delete_photo" and name and body.get("id"):
            await store.async_remove_photo(name, body["id"])
        elif action == "delete_person" and name:
            await store.async_remove_person(name)
        elif action != "list":
            return web.json_response({"status": "error", "message": "неизвестное действие"}, status=400)

        return web.json_response(self._people_payload())

    async def _handle_add(self, request: web.Request) -> web.Response:
        """Добавляет фото человеку из загруженного файла."""
        try:
            post = await request.post()
        except Exception as exc:  # noqa: BLE001
            return self._add_result("error", f"Не удалось прочитать файл: {exc}")

        name = (post.get("name") or "").strip()
        if not name:
            return self._add_result("error", "Введите имя человека.")

        field = post.get("photo")
        image_bytes = field.file.read() if field is not None and hasattr(field, "file") else None
        if not image_bytes:
            return self._add_result("error", "Файл не получен.")

        cfg = _deepface_config(self.hass)
        if not cfg:
            return self._add_result("error", "Сервис DeepFace не настроен.")

        store = _face_store(self.hass)
        try:
            await store.async_add_person(
                name, image_bytes, cfg["url"], cfg["model"], cfg["detector"]
            )
        except deepface_client.NoFaceError:
            return self._add_result("error", "Лицо на фото не найдено. Возьмите фото анфас.")
        except deepface_client.DeepFaceError as exc:
            _LOGGER.error("Ошибка добавления фото для «%s»: %s", name, exc)
            return self._add_result("error", "Ошибка распознавания. Попробуйте другое фото.")

        return self._add_result("ok", f"Фото добавлено к «{name}».")

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


def _person_card(person: dict) -> str:
    name_e = html.escape(person["name"], quote=True)
    thumbs = []
    for photo in person["photos"]:
        pid = html.escape(photo["id"], quote=True)
        if photo.get("photo"):
            inner = f'<img src="data:image/jpeg;base64,{photo["photo"]}">'
        else:
            inner = '<div class="noimg">без фото</div>'
        thumbs.append(
            f'<div class="thumb" data-id="{pid}">{inner}<button class="del">✕</button></div>'
        )
    grid = "".join(thumbs)
    return (
        f'<div class="person" data-name="{name_e}">'
        f'<div class="phead"><span class="pname">{name_e}</span>'
        f'<span class="pcount">{person["count"]} фото</span>'
        f'<button class="addphoto">＋ фото</button>'
        f'<button class="delperson">Удалить человека</button>'
        f'<input type="file" class="pfile" accept="image/*" hidden></div>'
        f'<div class="grid">{grid}</div></div>'
    )


def _render_page(people: list[dict]) -> str:
    cards = (
        "".join(_person_card(p) for p in people)
        if people
        else '<p class="empty">Пока нет добавленных лиц. Добавьте человека в настройках '
        "или по ссылке добавления лица.</p>"
    )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Росдомофон · лица</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif; background:#111; color:#eee; }}
  header {{ padding:14px 16px; background:#1c1c1c; position:sticky; top:0; }}
  header h1 {{ font-size:16px; margin:0; }}
  .wrap {{ padding:16px; max-width:760px; margin:0 auto; }}
  .person {{ background:#1c1c1c; border-radius:12px; padding:12px 14px; margin-bottom:14px; }}
  .phead {{ display:flex; align-items:center; gap:10px; margin-bottom:10px; }}
  .pname {{ font-weight:700; font-size:15px; }}
  .pcount {{ font-size:12px; color:#9ab; }}
  .addphoto {{ margin-left:auto; background:#123; color:#9cf; border:1px solid #46a;
              border-radius:8px; padding:6px 10px; cursor:pointer; font-size:12px; }}
  .delperson {{ background:#402; color:#f99; border:1px solid #a44;
               border-radius:8px; padding:6px 10px; cursor:pointer; font-size:12px; }}
  .newrow {{ display:flex; gap:8px; margin-bottom:14px; flex-wrap:wrap; }}
  .newrow input[type=text] {{ flex:1; min-width:160px; padding:10px; border-radius:8px;
    border:1px solid #444; background:#0d0d0d; color:#eee; }}
  .newrow button {{ background:#fff; color:#7b5cff; border:none; border-radius:8px;
    padding:10px 14px; font-weight:700; cursor:pointer; }}
  .status {{ min-height:1.2em; font-size:13px; margin:-4px 0 14px; }}
  .status.ok {{ color:#8fe6b0; }} .status.err {{ color:#ff9a9a; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(84px,1fr)); gap:8px; }}
  .thumb {{ position:relative; aspect-ratio:1; border-radius:10px; overflow:hidden; background:#0003; }}
  .thumb img {{ width:100%; height:100%; object-fit:cover; }}
  .thumb .noimg {{ display:flex; align-items:center; justify-content:center; height:100%;
                  font-size:.7rem; opacity:.6; }}
  .del {{ position:absolute; top:3px; right:3px; width:22px; height:22px; border:none;
         border-radius:50%; background:#000a; color:#fff; cursor:pointer; font-size:.8rem; }}
  .empty {{ color:#999; padding:24px; }}
</style></head>
<body>
<header><h1>Эталонные лица</h1></header>
<div class="wrap">
  <div class="newrow">
    <input type="text" id="newname" placeholder="Имя нового человека">
    <button id="addnew">＋ Добавить фото</button>
    <input type="file" id="newfile" accept="image/*" hidden>
  </div>
  <div class="status" id="status"></div>
  <div id="root">{cards}</div>
</div>
<script>
  const base = window.location.pathname + window.location.search;
  const root = document.getElementById('root');
  const status = document.getElementById('status');
  const newname = document.getElementById('newname');
  const newfile = document.getElementById('newfile');
  const addnew = document.getElementById('addnew');

  function esc(s) {{ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }}
  function setStatus(msg, ok) {{ status.textContent = msg || ''; status.className = 'status ' + (ok ? 'ok' : 'err'); }}

  function render(people) {{
    if (!people.length) {{ root.innerHTML = '<p class="empty">Пока нет добавленных лиц.</p>'; return; }}
    root.innerHTML = people.map(p => {{
      const thumbs = (p.photos||[]).map(ph => {{
        const inner = ph.photo
          ? '<img src="data:image/jpeg;base64,' + ph.photo + '">'
          : '<div class="noimg">без фото</div>';
        return '<div class="thumb" data-id="' + esc(ph.id) + '">' + inner + '<button class="del">✕</button></div>';
      }}).join('');
      return '<div class="person" data-name="' + esc(p.name) + '"><div class="phead">' +
        '<span class="pname">' + esc(p.name) + '</span>' +
        '<span class="pcount">' + p.count + ' фото</span>' +
        '<button class="addphoto">＋ фото</button>' +
        '<button class="delperson">Удалить человека</button>' +
        '<input type="file" class="pfile" accept="image/*" hidden></div>' +
        '<div class="grid">' + thumbs + '</div></div>';
    }}).join('');
  }}

  async function act(payload) {{
    const r = await fetch(base, {{ method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify(payload) }});
    const d = await r.json();
    if (d.people) render(d.people);
  }}

  async function uploadPhoto(name, file) {{
    if (!name) {{ setStatus('Введите имя человека.', false); return; }}
    setStatus('Загружаю и распознаю…', true);
    const fd = new FormData(); fd.append('name', name); fd.append('photo', file);
    try {{
      const r = await fetch(base, {{ method:'POST', body: fd }});
      const d = await r.json();
      setStatus(d.message, d.status === 'ok');
      if (d.people) render(d.people);
      if (d.status === 'ok') {{ newname.value = ''; }}
    }} catch (e) {{ setStatus('Ошибка загрузки.', false); }}
  }}

  addnew.addEventListener('click', () => {{
    if (!newname.value.trim()) {{ setStatus('Введите имя нового человека.', false); return; }}
    newfile.click();
  }});
  newfile.addEventListener('change', e => {{
    if (e.target.files[0]) uploadPhoto(newname.value.trim(), e.target.files[0]);
    e.target.value = '';
  }});

  root.addEventListener('click', e => {{
    const person = e.target.closest('.person');
    if (!person) return;
    const name = person.dataset.name;
    if (e.target.classList.contains('del')) {{
      const thumb = e.target.closest('.thumb');
      if (thumb) act({{action:'delete_photo', name:name, id:thumb.dataset.id}});
    }} else if (e.target.classList.contains('delperson')) {{
      if (confirm('Удалить «' + name + '» и все его фото?'))
        act({{action:'delete_person', name:name}});
    }} else if (e.target.classList.contains('addphoto')) {{
      const f = person.querySelector('.pfile');
      if (f) f.click();
    }}
  }});
  root.addEventListener('change', e => {{
    if (e.target.classList.contains('pfile')) {{
      const person = e.target.closest('.person');
      if (person && e.target.files[0]) uploadPhoto(person.dataset.name, e.target.files[0]);
      e.target.value = '';
    }}
  }});
</script>
</body></html>"""


def setup_faces_view(hass: HomeAssistant) -> None:
    """Регистрирует HTTP-view галереи лиц (один раз на домен)."""
    hass.http.register_view(RosdomofonFacesView(hass))
    _LOGGER.info("Галерея эталонных лиц зарегистрирована: %s", _FACES_PATH)
