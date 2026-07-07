"""
Поток настройки (Config Flow) интеграции Росдомофон.

Шаг 1 - пользователь вводит номер телефона РФ, сервис отправляет SMS.
Шаг 2 - пользователь вводит код из SMS, интеграция получает OAuth-токен.
"""

import logging
import re
import time

import aiohttp
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.components.file_upload import process_uploaded_file
from homeassistant.core import callback
from homeassistant.helpers import aiohttp_client, entity_registry as er, selector

from . import deepface_client
from .const import (
    CLIENT_ID,
    COMPANY_NAME,
    CONF_ANTISPOOF,
    CONF_CAMERAS,
    CONF_COOLDOWN,
    CONF_DEEPFACE_URL,
    CONF_INTERVAL,
    CONF_MODEL,
    CONF_THRESHOLD,
    DATA_FACE_STORE,
    DEFAULT_ANTISPOOF,
    DEFAULT_COOLDOWN,
    DEFAULT_DETECTOR,
    DEFAULT_INTERVAL,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    DOMAIN,
    GRANT_TYPE_MOBILE,
    PHONE_LENGTH,
    PHONE_PREFIX,
    SMS_REQUEST_URL,
    TOKEN_REQUEST_URL,
)

_LOGGER = logging.getLogger(__name__)

# Таймаут для HTTP-запросов к API
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)


def _normalize_phone(raw_phone: str) -> str:
    """Приводит номер телефона к строгому формату (11 цифр, начиная с 7).

    Удаляет пробелы, тире, скобки, плюс.
    Заменяет ведущую 8 на 7.
    """
    digits = re.sub(r"\D", "", raw_phone)
    if digits.startswith("8") and len(digits) == PHONE_LENGTH:
        digits = PHONE_PREFIX + digits[1:]
    return digits


def _validate_phone(phone: str) -> str | None:
    """Возвращает код ошибки или None если номер корректен."""
    if len(phone) != PHONE_LENGTH:
        return "invalid_phone_length"
    if not phone.startswith(PHONE_PREFIX):
        return "invalid_phone_prefix"
    return None


class RosdomofonConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Поток настройки интеграции Росдомофон."""

    VERSION = 1

    def __init__(self):
        self._phone: str | None = None
        self._tok: dict | None = None

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Возвращает поток настроек (распознавание лиц)."""
        return RosdomofonOptionsFlow()

    # --- Шаг 1: Ввод номера телефона ---

    async def async_step_user(self, user_input=None):
        """Запрос номера телефона и отправка SMS."""
        errors: dict[str, str] = {}

        if user_input is not None:
            phone = _normalize_phone(user_input["phone"])
            error = _validate_phone(phone)

            if error:
                errors["phone"] = error
            elif await self._request_sms(phone):
                self._phone = phone
                return await self.async_step_sms()
            else:
                errors["base"] = "sms_failed"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required("phone"): str,
            }),
            description_placeholders={
                "note": "+7 (XXX) XXX-XX-XX, можно вводить в свободном формате — пробелы и символы будут удалены автоматически",
            },
            errors=errors,
        )

    # --- Шаг 2: Ввод SMS-кода ---

    async def async_step_sms(self, user_input=None):
        """Запрос кода из SMS и получение токена."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._tok = await self._get_token(
                self._phone, user_input["sms_code"]
            )
            if self._tok:
                self._tok["timestamp"] = int(time.time())
                return self._create_entry()
            errors["base"] = "invalid_code"

        return self.async_show_form(
            step_id="sms",
            data_schema=vol.Schema({
                vol.Required("sms_code"): str,
            }),
            description_placeholders={"phone": self._phone},
            errors=errors,
        )

    # --- Создание config entry ---

    def _create_entry(self):
        """Создаёт config entry с данными авторизации."""
        return self.async_create_entry(
            title=f"Росдомофон ({self._phone})",
            data={
                "phone": self._phone,
                "token_data": self._tok,
            },
        )

    # --- HTTP-запросы к API ---

    async def _request_sms(self, phone: str) -> bool:
        """Отправляет запрос на SMS-код для указанного номера."""
        try:
            session = aiohttp_client.async_get_clientsession(self.hass)
            async with session.post(
                SMS_REQUEST_URL.format(phone=phone),
                headers={"Content-Type": "application/json"},
                timeout=_REQUEST_TIMEOUT,
            ) as resp:
                if resp.status == 200:
                    _LOGGER.debug("SMS отправлено успешно")
                    return True
                _LOGGER.error("Ошибка отправки SMS: %d", resp.status)
        except (aiohttp.ClientError, TimeoutError) as exc:
            _LOGGER.error("Ошибка запроса SMS: %s", exc)
        return False

    async def _get_token(self, phone: str, sms_code: str) -> dict | None:
        """Получает OAuth-токен по номеру телефона и SMS-коду."""
        try:
            session = aiohttp_client.async_get_clientsession(self.hass)
            payload = {
                "grant_type": GRANT_TYPE_MOBILE,
                "client_id": CLIENT_ID,
                "phone": phone,
                "sms_code": sms_code,
                "company": COMPANY_NAME,
            }
            async with session.post(
                TOKEN_REQUEST_URL,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=_REQUEST_TIMEOUT,
            ) as resp:
                if resp.status == 200:
                    _LOGGER.debug("Токен получен успешно")
                    return await resp.json()
                _LOGGER.error(
                    "Ошибка получения токена: %d %s",
                    resp.status,
                    await resp.text(),
                )
        except (aiohttp.ClientError, TimeoutError) as exc:
            _LOGGER.error("Ошибка запроса токена: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Options Flow — распознавание лиц (авто-открытие двери)
# ---------------------------------------------------------------------------


class RosdomofonOptionsFlow(config_entries.OptionsFlow):
    """Настройка авто-открытия по лицу: сервис, люди, камеры."""

    async def async_step_init(self, user_input=None):
        """Главное меню настроек."""
        return self.async_show_menu(
            step_id="init",
            menu_options=["settings", "people", "cameras"],
        )

    # --- Настройки сервиса DeepFace ---

    async def async_step_settings(self, user_input=None):
        """Параметры сервиса распознавания."""
        errors: dict[str, str] = {}
        opts = dict(self.config_entry.options)

        if user_input is not None:
            url = user_input[CONF_DEEPFACE_URL].strip()
            reachable = await self.hass.async_add_executor_job(
                deepface_client.check_available, url
            )
            if not reachable:
                errors["base"] = "deepface_unreachable"
            else:
                opts.update(user_input)
                opts[CONF_DEEPFACE_URL] = url
                return self.async_create_entry(title="", data=opts)

        schema = vol.Schema({
            vol.Required(
                CONF_DEEPFACE_URL,
                default=opts.get(CONF_DEEPFACE_URL, ""),
            ): str,
            vol.Optional(
                CONF_MODEL,
                default=opts.get(CONF_MODEL, DEFAULT_MODEL),
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=["Facenet512", "Facenet", "ArcFace", "VGG-Face", "SFace"],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_THRESHOLD,
                default=opts.get(CONF_THRESHOLD, DEFAULT_THRESHOLD),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.1, max=0.6, step=0.01, mode=selector.NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_INTERVAL,
                default=opts.get(CONF_INTERVAL, DEFAULT_INTERVAL),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=1, max=30, step=1)
            ),
            vol.Optional(
                CONF_COOLDOWN,
                default=opts.get(CONF_COOLDOWN, DEFAULT_COOLDOWN),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(min=5, max=600, step=5)
            ),
            vol.Optional(
                CONF_ANTISPOOF,
                default=opts.get(CONF_ANTISPOOF, DEFAULT_ANTISPOOF),
            ): bool,
        })
        return self.async_show_form(
            step_id="settings", data_schema=schema, errors=errors
        )

    # --- Управление людьми ---

    async def async_step_people(self, user_input=None):
        """Меню управления людьми."""
        return self.async_show_menu(
            step_id="people",
            menu_options=["add_person", "remove_person"],
        )

    async def async_step_add_person(self, user_input=None):
        """Добавление человека: имя + фото (загрузка через UI)."""
        errors: dict[str, str] = {}
        opts = self.config_entry.options
        store = self.hass.data.get(DOMAIN, {}).get(DATA_FACE_STORE)

        if not opts.get(CONF_DEEPFACE_URL):
            return self.async_abort(reason="configure_deepface_first")

        if user_input is not None:
            name = user_input["name"].strip()
            file_id = user_input["photo"]
            if not name:
                errors["name"] = "empty_name"
            else:
                try:
                    image = await self.hass.async_add_executor_job(
                        self._read_uploaded_file, file_id
                    )
                    await store.async_add_person(
                        name,
                        image,
                        opts[CONF_DEEPFACE_URL],
                        opts.get(CONF_MODEL, DEFAULT_MODEL),
                        DEFAULT_DETECTOR,
                    )
                    return await self.async_step_people()
                except deepface_client.SpoofDetected:
                    errors["base"] = "spoof_on_photo"
                except deepface_client.DeepFaceError as exc:
                    _LOGGER.error("Ошибка добавления лица: %s", exc)
                    errors["base"] = "no_face"

        schema = vol.Schema({
            vol.Required("name"): str,
            vol.Required("photo"): selector.FileSelector(
                selector.FileSelectorConfig(accept="image/*")
            ),
        })
        return self.async_show_form(
            step_id="add_person", data_schema=schema, errors=errors
        )

    async def async_step_remove_person(self, user_input=None):
        """Удаление человека."""
        store = self.hass.data.get(DOMAIN, {}).get(DATA_FACE_STORE)
        people = store.people if store else []

        if not people:
            return self.async_abort(reason="no_people")

        if user_input is not None:
            await store.async_remove_person(user_input["name"])
            return await self.async_step_people()

        schema = vol.Schema({
            vol.Required("name"): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=people, mode=selector.SelectSelectorMode.DROPDOWN
                )
            ),
        })
        return self.async_show_form(step_id="remove_person", data_schema=schema)

    # --- Привязка камер к замкам ---

    async def async_step_cameras(self, user_input=None):
        """Выбор камер и замков, которые они открывают."""
        opts = dict(self.config_entry.options)
        current = opts.get(CONF_CAMERAS, {})

        registry = er.async_get(self.hass)
        entries = er.async_entries_for_config_entry(
            registry, self.config_entry.entry_id
        )
        cameras = [e for e in entries if e.domain == "camera"]

        if not cameras:
            return self.async_abort(reason="no_cameras")

        if user_input is not None:
            mapping = {
                cam: lock for cam, lock in user_input.items() if lock
            }
            opts[CONF_CAMERAS] = mapping
            return self.async_create_entry(title="", data=opts)

        schema_dict = {}
        lock_selector = selector.EntitySelector(
            selector.EntitySelectorConfig(domain="lock", integration=DOMAIN)
        )
        for cam in cameras:
            schema_dict[
                vol.Optional(
                    cam.entity_id,
                    description={"suggested_value": current.get(cam.entity_id)},
                )
            ] = lock_selector

        return self.async_show_form(
            step_id="cameras", data_schema=vol.Schema(schema_dict)
        )

    # --- Вспомогательное ---

    def _read_uploaded_file(self, file_id: str) -> bytes:
        """Читает загруженный через UI файл (в executor)."""
        with process_uploaded_file(self.hass, file_id) as path:
            return path.read_bytes()
