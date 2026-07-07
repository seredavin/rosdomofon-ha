"""
Клиент к сервису DeepFace (REST API) для интеграции Росдомофон.

DeepFace используется для получения эмбеддингов лиц и проверки на подделку
(anti-spoofing). Запросы синхронные (requests) — вызывать из executor.
"""

import base64
import logging

import requests

_LOGGER = logging.getLogger(__name__)

# Таймаут запроса к DeepFace (инференс на CPU может быть небыстрым)
_REQUEST_TIMEOUT = 30


class DeepFaceError(Exception):
    """Базовая ошибка обращения к DeepFace."""


class SpoofDetected(DeepFaceError):
    """DeepFace определил подделку (фото/экран вместо живого лица)."""


def _image_to_data_uri(image: bytes) -> str:
    """Преобразует байты изображения в data URI (DeepFace ждёт такой формат)."""
    b64 = base64.b64encode(image).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def represent(
    base_url: str,
    image: bytes,
    model_name: str,
    detector_backend: str,
    anti_spoofing: bool,
) -> list[list[float]]:
    """Возвращает эмбеддинги всех лиц на изображении.

    При включённом anti_spoofing DeepFace бросает ошибку, если лицо признано
    подделкой — тогда поднимаем SpoofDetected. Если лицо не найдено, возвращаем
    пустой список (enforce_detection=false). Прочие сбои — DeepFaceError.
    """
    url = f"{base_url.rstrip('/')}/represent"
    payload = {
        "img": _image_to_data_uri(image),
        "model_name": model_name,
        "detector_backend": detector_backend,
        "anti_spoofing": anti_spoofing,
        # Не падать, если лицо не обнаружено — просто вернуть пусто.
        "enforce_detection": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise DeepFaceError(f"Сетевая ошибка обращения к DeepFace: {exc}") from exc

    if response.status_code == 200:
        try:
            results = response.json().get("results", [])
        except ValueError as exc:
            raise DeepFaceError(f"Некорректный ответ DeepFace: {exc}") from exc
        embeddings: list[list[float]] = []
        for item in results:
            embedding = item.get("embedding")
            if embedding:
                embeddings.append(embedding)
        return embeddings

    # Ненулевой статус — разбираем сообщение об ошибке.
    message = _error_message(response)
    lowered = message.lower()
    if "spoof" in lowered:
        raise SpoofDetected(message)
    if "could not be detected" in lowered or "face could not" in lowered:
        # Лицо не найдено — не ошибка для нашего сценария.
        return []
    raise DeepFaceError(f"DeepFace вернул {response.status_code}: {message}")


def _error_message(response: requests.Response) -> str:
    """Извлекает текст ошибки из ответа DeepFace."""
    try:
        data = response.json()
    except ValueError:
        return response.text
    if isinstance(data, dict):
        return str(data.get("error") or data.get("message") or data)
    return str(data)


def check_available(base_url: str) -> bool:
    """Проверяет доступность сервиса DeepFace (для config flow)."""
    url = f"{base_url.rstrip('/')}/"
    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException:
        return False
    return response.status_code == 200
