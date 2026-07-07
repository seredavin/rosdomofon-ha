"""
Отладочная галерея кадров, отправленных в DeepFace.

Держит в памяти кольцевой буфер последних кадров (когда включена отладка) с
результатом по каждому: сколько лиц, совпадение/расстояние, подделка, ошибка.
Отдаёт HTML-галерею по адресу /api/rosdomofon/debug. Доступ защищён той же
HMAC-подписью, что и прокси потоков (см. stream_proxy) — ссылку с подписью
пользователь получает в уведомлении при включении отладки.
"""

import html
import logging
from collections import deque

from aiohttp import web
from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant
from homeassistant.helpers.network import get_url

from .const import DEBUG_LOG_MAXLEN, DOMAIN
from .stream_proxy import _validate_proxy_request, sign_proxy_path

_LOGGER = logging.getLogger(__name__)

# Ключ, под которым в hass.data лежит отладочный лог
DATA_DEBUG_LOG = "_debug_log"
_DEBUG_PATH = "/api/rosdomofon/debug"


class DebugLog:
    """Кольцевой буфер последних кадров, отправленных в DeepFace."""

    def __init__(self, maxlen: int = DEBUG_LOG_MAXLEN) -> None:
        self._entries: deque = deque(maxlen=maxlen)
        self._seq = 0

    def add(
        self,
        camera: str,
        image: bytes,
        summary: str,
        elapsed_ms: int,
        when: str,
    ) -> None:
        """Добавляет запись (последняя — в начале)."""
        self._seq += 1
        self._entries.appendleft(
            {
                "id": self._seq,
                "camera": camera,
                "image": image,
                "summary": summary,
                "elapsed_ms": elapsed_ms,
                "when": when,
            }
        )

    def entries(self) -> list[dict]:
        """Записи от новых к старым."""
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()


def get_debug_log(hass: HomeAssistant) -> DebugLog | None:
    """Возвращает отладочный лог из hass.data (если создан)."""
    return hass.data.get(DOMAIN, {}).get(DATA_DEBUG_LOG)


def debug_gallery_url(hass: HomeAssistant) -> str | None:
    """Строит абсолютную подписанную ссылку на галерею (или None)."""
    try:
        base_url = get_url(hass, prefer_external=False)
    except Exception:  # noqa: BLE001 — внешний/внутренний URL может быть не настроен
        try:
            base_url = get_url(hass)
        except Exception:  # noqa: BLE001
            return None
    return f"{base_url}{sign_proxy_path(hass, _DEBUG_PATH)}"


class RosdomofonDebugView(HomeAssistantView):
    """HTTP View: HTML-галерея отладочных кадров DeepFace."""

    url = _DEBUG_PATH
    name = "api:rosdomofon:debug"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass

    async def get(self, request: web.Request) -> web.Response:
        """Отдаёт HTML-галерею последних кадров с результатами."""
        if not _validate_proxy_request(self.hass, request):
            return web.Response(status=401, text="Invalid signature")

        debug_log = get_debug_log(self.hass)
        entries = debug_log.entries() if debug_log else []
        return web.Response(
            text=_render_gallery(entries),
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )


def _render_gallery(entries: list[dict]) -> str:
    """Собирает HTML-страницу галереи из записей."""
    import base64

    cards = []
    for entry in entries:
        b64 = base64.b64encode(entry["image"]).decode("ascii")
        summary = html.escape(entry["summary"])
        camera = html.escape(entry["camera"])
        when = html.escape(entry["when"])
        cards.append(
            f'<div class="card">'
            f'<img src="data:image/jpeg;base64,{b64}" loading="lazy">'
            f'<div class="meta"><div class="head">{when} · {camera} '
            f'· {entry["elapsed_ms"]} мс</div>'
            f'<div class="sum">{summary}</div></div>'
            f"</div>"
        )

    body = (
        "".join(cards)
        if cards
        else '<p class="empty">Пока нет отправленных кадров. '
        "Дождитесь движения/лица перед камерой при включённой отладке.</p>"
    )

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Росдомофон · отладка DeepFace</title>
<style>
  body {{ margin: 0; font-family: system-ui, sans-serif; background: #111; color: #eee; }}
  header {{ padding: 12px 16px; background: #1c1c1c; position: sticky; top: 0; }}
  header h1 {{ font-size: 16px; margin: 0; }}
  header .hint {{ font-size: 12px; color: #999; margin-top: 4px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
           gap: 12px; padding: 16px; }}
  .card {{ background: #1c1c1c; border-radius: 10px; overflow: hidden; }}
  .card img {{ width: 100%; display: block; background: #000; }}
  .meta {{ padding: 8px 10px; }}
  .head {{ font-size: 12px; color: #9ab; }}
  .sum {{ font-size: 13px; margin-top: 3px; word-break: break-word; }}
  .empty {{ padding: 24px; color: #999; }}
</style></head>
<body>
<header><h1>Отладка распознавания · кадры, отправленные в DeepFace</h1>
<div class="hint">Последние {len(entries)} кадров (новые сверху). Обновите страницу, чтобы увидеть свежие.</div>
</header>
<div class="grid">{body}</div>
</body></html>"""


def setup_debug_view(hass: HomeAssistant) -> None:
    """Регистрирует отладочный лог и HTTP-view (один раз на домен)."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    if DATA_DEBUG_LOG not in domain_data:
        domain_data[DATA_DEBUG_LOG] = DebugLog()
    hass.http.register_view(RosdomofonDebugView(hass))
    _LOGGER.info("Отладочная галерея DeepFace зарегистрирована: %s", _DEBUG_PATH)
