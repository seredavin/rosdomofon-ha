"""Константы интеграции Росдомофон."""

DOMAIN = "rosdomofon"

# Базовый URL API Росдомофон
BASE_URL = "https://rdba.rosdomofon.com"

# Эндпоинты авторизации
# noinspection SpellCheckingInspection
SMS_REQUEST_URL = f"{BASE_URL}/abonents-service/api/v1/abonents/{{phone}}/sms"
# noinspection SpellCheckingInspection
TOKEN_REQUEST_URL = f"{BASE_URL}/authserver-service/oauth/token"

# Эндпоинты замков
# noinspection SpellCheckingInspection
LOCKS_LIST_URL = f"{BASE_URL}/abonents-service/api/v2/abonents/keys"
# noinspection SpellCheckingInspection
LOCK_UNLOCK_URL = f"{BASE_URL}/rdas-service/api/v1/rdas/{{adapter_id}}/activate_key"

# Параметры OAuth
GRANT_TYPE_MOBILE = "mobile"
GRANT_TYPE_REFRESH = "refresh_token"
# noinspection SpellCheckingInspection
CLIENT_ID = "abonent"
COMPANY_NAME = ""

# Валидация номера телефона РФ (11 цифр, начинается с 7)
PHONE_LENGTH = 11
PHONE_PREFIX = "7"

# Эндпоинты камер
# noinspection SpellCheckingInspection
CAMERAS_LIST_URL = f"{BASE_URL}/abonents-service/api/v2/abonents/cameras"
CAMERA_DETAILS_URL = f"{BASE_URL}/cameras-service/api/v1/cameras/{{camera_id}}"

# Ссылки для гостевого доступа (Share Link)
SHARE_LINK_DEFAULT_TTL_HOURS = 12
SHARE_LINK_WEBHOOK_PREFIX = "rosdomofon_share_"

# ---------------------------------------------------------------------------
# Распознавание лиц (авто-открытие двери по лицу через DeepFace)
# ---------------------------------------------------------------------------

# Ключи опций config entry (options flow)
CONF_DEEPFACE_URL = "deepface_url"
CONF_MODEL = "model"
CONF_THRESHOLD = "threshold"
CONF_INTERVAL = "interval"
CONF_COOLDOWN = "cooldown"
CONF_ANTISPOOF = "anti_spoofing"
CONF_DETECTOR = "detector"
CONF_CAMERAS = "cameras"  # dict: camera_id -> {"enabled": bool, "lock": entity_id}

# Значения по умолчанию
DEFAULT_MODEL = "Facenet512"
# opencv встроен в образ DeepFace и не требует загрузки моделей (в отличие от
# retinaface/mtcnn) — быстрый и надёжный выбор по умолчанию для домофона.
DEFAULT_DETECTOR = "opencv"
# Порог косинусного расстояния: чем меньше — тем строже. 0.30 — базовый для
# Facenet512; для двери берём строгое значение по умолчанию.
DEFAULT_THRESHOLD = 0.28
DEFAULT_INTERVAL = 3  # секунды между кадрами
DEFAULT_COOLDOWN = 30  # секунды тишины после открытия
DEFAULT_ANTISPOOF = True

# Хранилище эталонных лиц (эмбеддинги)
FACE_STORE_KEY = f"{DOMAIN}_faces"
FACE_STORE_VERSION = 1

# Данные распознавания в hass.data[DOMAIN]
DATA_FACE_STORE = "_face_store"
DATA_FACE_COORDINATOR = "_face_coordinator"

# ---------------------------------------------------------------------------
# Лента активности лиц (события Логбука + image-сущности с кадрами)
# ---------------------------------------------------------------------------

# События на шине HA — попадают в Логбук (ленту активности)
EVENT_FACE_RECOGNIZED = f"{DOMAIN}_face_recognized"
EVENT_FACE_UNKNOWN = f"{DOMAIN}_face_unknown"

# Стабильные entity_id image-сущностей с последними кадрами.
# Задаём явно, чтобы события Логбука ссылались на конкретную сущность
# (иначе HA сгенерировал бы id из кириллического имени).
IMAGE_RECOGNIZED_OBJECT_ID = "rosdomofon_last_recognized_face"
IMAGE_UNKNOWN_OBJECT_ID = "rosdomofon_last_unknown_face"
IMAGE_RECOGNIZED_ENTITY_ID = f"image.{IMAGE_RECOGNIZED_OBJECT_ID}"
IMAGE_UNKNOWN_ENTITY_ID = f"image.{IMAGE_UNKNOWN_OBJECT_ID}"
