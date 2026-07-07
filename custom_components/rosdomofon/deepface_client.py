"""
Клиент к сервису DeepFace (REST API) для интеграции Росдомофон.

DeepFace используется для получения эмбеддингов лиц и проверки на подделку
(anti-spoofing). Запросы синхронные (requests) — вызывать из executor.
"""

import base64
import logging

import requests

_LOGGER = logging.getLogger(__name__)

# Таймаут запроса к DeepFace. Первый вызов может скачивать модели (~200 МБ)
# и грузить их в память, поэтому таймаут щедрый.
_REQUEST_TIMEOUT = 120


class DeepFaceError(Exception):
    """Базовая ошибка обращения к DeepFace."""


class SpoofDetected(DeepFaceError):
    """DeepFace определил подделку (фото/экран вместо живого лица)."""


class NoFaceError(DeepFaceError):
    """На изображении не найдено лицо."""


class AntiSpoofUnavailable(DeepFaceError):
    """Антиспуфинг недоступен в сервисе (в образе DeepFace не установлен torch)."""


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
    enforce_detection: bool = True,
) -> list[list[float]]:
    """Возвращает эмбеддинги всех лиц на изображении.

    При включённом anti_spoofing DeepFace бросает ошибку, если лицо признано
    подделкой — тогда поднимаем SpoofDetected. Прочие сбои — DeepFaceError.

    enforce_detection=True (по умолчанию): DeepFace сам детектирует лицо своим
    детектором (opencv/retinaface/…). Если лица нет — возвращаем пустой список
    (так «проверка на лицо в кадре» выполняется на стороне DeepFace и не тратит
    эмбеддинг на пустые кадры). Так как opencv в свежих сборках Home Assistant
    (Python 3.14) не устанавливается, локальный детектор лиц недоступен, и это —
    основной способ отсеять кадры без лиц.
    """
    url = f"{base_url.rstrip('/')}/represent"
    payload = {
        "img": _image_to_data_uri(image),
        "model_name": model_name,
        "detector_backend": detector_backend,
        "anti_spoofing": anti_spoofing,
        "enforce_detection": enforce_detection,
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
    # Именно обнаружение подделки ("Spoof detected in the given image."),
    # а не любые упоминания слова spoofing (например, ошибка про отсутствие torch).
    if "spoof detected" in lowered:
        raise SpoofDetected(message)
    if "install torch" in lowered or "anti spoofing" in lowered:
        raise AntiSpoofUnavailable(message)
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
