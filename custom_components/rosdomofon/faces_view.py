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
from homeassistant.helpers.network import get_url

from .const import DATA_FACE_STORE, DOMAIN
from .stream_proxy import _validate_proxy_request, sign_proxy_path

_LOGGER = logging.getLogger(__name__)

_FACES_PATH = "/api/rosdomofon/faces"


def faces_gallery_url(hass: HomeAssistant) -> str | None:
    """Строит абсолютную подписанную ссылку на галерею лиц (или None)."""
    try:
        base_url = get_url(hass, prefer_external=False)
    except Exception:  # noqa: BLE001
        try:
            base_url = get_url(hass)
        except Exception:  # noqa: BLE001
            return None
    return f"{base_url}{sign_proxy_path(hass, _FACES_PATH)}"


def _face_store(hass: HomeAssistant):
    return hass.data.get(DOMAIN, {}).get(DATA_FACE_STORE)


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
        f'<button class="delperson">Удалить человека</button></div>'
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
  .delperson {{ margin-left:auto; background:#402; color:#f99; border:1px solid #a44;
               border-radius:8px; padding:6px 10px; cursor:pointer; font-size:12px; }}
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
<div class="wrap" id="root">{cards}</div>
<script>
  const base = window.location.pathname + window.location.search;
  const root = document.getElementById('root');

  function esc(s) {{ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }}

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
        '<button class="delperson">Удалить человека</button></div>' +
        '<div class="grid">' + thumbs + '</div></div>';
    }}).join('');
  }}

  async function act(payload) {{
    const r = await fetch(base, {{ method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify(payload) }});
    const d = await r.json();
    if (d.people) render(d.people);
  }}

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
    }}
  }});
</script>
</body></html>"""


def setup_faces_view(hass: HomeAssistant) -> None:
    """Регистрирует HTTP-view галереи лиц (один раз на домен)."""
    hass.http.register_view(RosdomofonFacesView(hass))
    _LOGGER.info("Галерея эталонных лиц зарегистрирована: %s", _FACES_PATH)
