"""
Обновлённое ядро ИИ-агента с AEC и умной маршрутизацией.

Этот модуль содержит основную логику работы агента, включая:
- Загрузку конфигурации и профиля пользователя
- Векторную память для истории общения
- Маршрутизацию запросов к LLM моделям
- Инструменты для работы с системой
- Голосовой интерфейс с AEC
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import threading
import uuid
import webbrowser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import random
from typing import Any, Optional, Dict, List, Tuple, Callable, Union

from character.aelita_profile import AelitaProfile
import webbrowser
import requests
from bs4 import BeautifulSoup  # Для поиска в интернете

import requests
import yaml

# Импорт утилит
from utils.logger import LoggerFactory
from utils.exceptions import (
    JARVISError, ModelLoadError, AudioError, 
    ConfigError, MemoryError, ToolExecutionError
)

# ---------- Настройка логирования ----------
LoggerFactory.setup()
logger = LoggerFactory.get_logger("agent_core")

# ---------- Определяем базовую директорию ----------
BASE_DIR = Path(__file__).resolve().parent


# ---------- Загрузка конфигурации ----------
def load_config() -> Dict[str, Any]:
    """
    Безопасная загрузка конфигурации.
    
    Returns:
        Dict[str, Any]: Словарь с конфигурацией
        
    Raises:
        ConfigError: При ошибке загрузки конфигурации
    """
    config_path = BASE_DIR / "config.yaml"
    if not config_path.exists():
        raise ConfigError(f"Файл конфигурации не найден: {config_path}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Ошибка парсинга YAML: {e}")
    except Exception as e:
        raise ConfigError(f"Ошибка загрузки конфигурации: {e}")


CONFIG = load_config()
logger.info("Конфигурация загружена успешно")


# ---------- Импорт зависимостей с проверкой ----------
def safe_import_module(module_name: str, package: str) -> Optional[Any]:
    """
    Безопасный импорт модуля с обработкой ошибок.
    
    Args:
        module_name: Имя модуля для импорта
        package: Имя пакета для установки
        
    Returns:
        Optional[Any]: Импортированный модуль или None при ошибке
    """
    try:
        return __import__(module_name)
    except ImportError:
        logger.warning(f"Модуль {module_name} не найден. Установите: pip install {package}")
        return None
    except Exception as e:
        logger.error(f"Ошибка импорта {module_name}: {e}")
        return None


# Импорт зависимостей с проверкой доступности
_chromadb = safe_import_module("chromadb", "chromadb")
_sentence_transformers = safe_import_module("sentence_transformers", "sentence-transformers")
_cv2 = safe_import_module("cv2", "opencv-python")
_mss = safe_import_module("mss", "mss")
_pil = safe_import_module("PIL", "pillow")
_psutil = safe_import_module("psutil", "psutil")

# Настройка глобальных переменных для импортированных модулей
chromadb = _chromadb
SentenceTransformer = _sentence_transformers.SentenceTransformer if _sentence_transformers else None
cv2 = _cv2
mss = _mss
Image = _pil.Image if _pil else None
psutil = _psutil

# Проверка доступности RAG
RAG_AVAILABLE = chromadb is not None and SentenceTransformer is not None

# Проверка доступности AEC Voice Interface
try:
    from voice.aec_voice_interface import AECVoiceInterface, VoiceConfig
    AEC_AVAILABLE = True
    logger.info("✅ AEC Voice Interface загружен")
except ImportError as e:
    logger.warning(f"AEC Voice Interface недоступен: {e}")
    logger.info("   Установите: pip install pywebrtc-audio sounddevice")
    AECVoiceInterface = None
    AEC_AVAILABLE = False
except Exception as e:
    logger.error(f"Ошибка загрузки AEC Voice Interface: {e}")
    AECVoiceInterface = None
    AEC_AVAILABLE = False


# ---------- Пути ----------
MEMORY_DIR = BASE_DIR / "memory"
PROFILE_FILE = MEMORY_DIR / CONFIG["memory"]["profile_file"]
CHROMA_DIR = MEMORY_DIR / CONFIG["memory"]["chroma_dir"]
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

# Создание директорий
for dir_path in [MEMORY_DIR, CHROMA_DIR, SCREENSHOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Настройки Ollama
OLLAMA_URL = CONFIG["ollama"]["url"]
FAST_MODEL = CONFIG["ollama"]["fast_model"]
SMART_MODEL = CONFIG["ollama"]["smart_model"]
VISION_MODEL = CONFIG["ollama"]["vision_model"]
ALLOWED_APPS = CONFIG["tools"]["allowed_apps"]

# ---------- Профиль по умолчанию ----------
DEFAULT_PROFILE: Dict[str, Any] = {
    "user": {
        "name": None,
        "preferred_language": "ru",
        "communication_style": "понятный",
    },
    "preferences": {
        "answer_length": "medium",
        "voice_enabled": True,
        "camera_enabled": False,
    },
    "projects": [],
    "goals": [],
    "important_facts": [],
    "permissions": {
        "camera": False,
        "face_recognition": False,
        "screen_control": False,
        "file_management": False,
        "browser_control": False,
    },
    "face_identity": {
        "registered": False,
        "encoding_file": str(MEMORY_DIR / "face" / "face_encoding.json"),
    },
}


# ---------- Структурная память ----------
class JsonMemory:
    """
    Структурированная долговременная память агента.
    
    Хранит профиль пользователя, факты, проекты и настройки в JSON формате.
    Обеспечивает потокобезопасный доступ к данным.
    """
    
    def __init__(self, path: Path = PROFILE_FILE):
        """
        Инициализация памяти.
        
        Args:
            path: Путь к файлу профиля
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        if not self.path.exists():
            self.save(deepcopy(DEFAULT_PROFILE))
            logger.info(f"Создан новый профиль: {self.path}")
    
    def load(self) -> Dict[str, Any]:
        """
        Загрузка профиля из файла.
        
        Returns:
            Dict[str, Any]: Профиль пользователя
            
        Raises:
            MemoryError: При ошибке чтения или парсинга
        """
        with self._lock:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                return data
            except FileNotFoundError:
                logger.warning(f"Файл профиля не найден: {self.path}")
                profile = deepcopy(DEFAULT_PROFILE)
                self.save(profile)
                return profile
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка парсинга JSON: {e}")
                logger.warning("Восстановление профиля по умолчанию")
                profile = deepcopy(DEFAULT_PROFILE)
                self.save(profile)
                return profile
            except Exception as e:
                logger.error(f"Ошибка загрузки профиля: {e}")
                raise MemoryError(f"Не удалось загрузить профиль: {e}")
    
    def save(self, profile: Dict[str, Any]) -> None:
        """
        Сохранение профиля с атомарной записью.
        
        Args:
            profile: Данные профиля
            
        Raises:
            MemoryError: При ошибке записи
        """
        with self._lock:
            tmp_path = self.path.with_suffix(".tmp")
            try:
                content = json.dumps(profile, ensure_ascii=False, indent=2)
                tmp_path.write_text(content, encoding="utf-8")
                tmp_path.replace(self.path)
                logger.debug(f"Профиль сохранён: {self.path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения профиля: {e}")
                raise MemoryError(f"Не удалось сохранить профиль: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получение значения по ключу с поддержкой вложенности.
        
        Args:
            key: Ключ с разделителями точками
            default: Значение по умолчанию
            
        Returns:
            Any: Значение или default
        """
        profile = self.load()
        parts = key.split('.')
        current = profile
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return default
            else:
                return default
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Установка значения по ключу с поддержкой вложенности.
        
        Args:
            key: Ключ с разделителями точками
            value: Устанавливаемое значение
        """
        profile = self.load()
        parts = key.split('.')
        current = profile
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
        self.save(profile)
    
    def set_name(self, name: str) -> None:
        """Установка имени пользователя."""
        self.set('user.name', name.strip())
        logger.info(f"Имя пользователя установлено: {name}")
    
    def add_fact(self, fact: str) -> None:
        """Добавление важного факта."""
        profile = self.load()
        facts = profile.setdefault("important_facts", [])
        if fact not in facts:
            facts.append(fact)
            self.save(profile)
            logger.info(f"Добавлен факт: {fact[:50]}...")
    
    def add_project(self, name: str, description: str = "") -> None:
        """Добавление проекта."""
        profile = self.load()
        projects = profile.setdefault("projects", [])
        projects.append({"name": name, "description": description})
        self.save(profile)
        logger.info(f"Добавлен проект: {name}")
    
    def clear(self) -> None:
        """Очистка профиля (сброс к настройкам по умолчанию)."""
        self.save(deepcopy(DEFAULT_PROFILE))
        logger.warning("Профиль сброшен к настройкам по умолчанию")


# ---------- Векторная память ----------
class RagMemory:
    """
    Векторная память для всей истории общения.
    
    Использует ChromaDB для хранения эмбеддингов и поиска по смыслу.
    """
    
    def __init__(self):
        """Инициализация векторной памяти."""
        if not RAG_AVAILABLE:
            error_msg = (
                "Для работы RAG памяти установите:\n"
                "pip install chromadb sentence-transformers"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            # Загружаем модель ИЗ ПРОЕКТА, а не из интернета
            model_path = BASE_DIR / "models" / "sentence-transformers" / "paraphrase-multilingual-MiniLM-L12-v2"
            
            if model_path.exists():
                self.embedder = SentenceTransformer(str(model_path), device="cpu")
                logger.info(f"✅ RAG модель загружена из проекта: {model_path}")
            else:
                # Fallback: загружаем из интернета (если нет локальной)
                logger.warning(f"Модель не найдена в проекте, загружаем из интернета...")
                self.embedder = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2",
                    device="cpu"
                )
                # Сохраняем в проект для следующих запусков
                self.embedder.save(str(model_path))
                logger.info(f"✅ Модель сохранена в проект: {model_path}")
            
            # Инициализация ChromaDB клиента и коллекции
            self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            self.collection = self.client.get_or_create_collection(
                name="conversation_history",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✅ RAG память инициализирована")
        except Exception as e:
            logger.error(f"Ошибка инициализации RAG: {e}")
            raise MemoryError(f"Не удалось инициализировать RAG: {e}")
    
    def add(self, role: str, content: str, session_id: str) -> None:
        """
        Добавление сообщения в векторную память.
        
        Args:
            role: Роль (user/assistant)
            content: Текст сообщения
            session_id: ID сессии
        """
        if not content or not content.strip():
            return
        
        try:
            embedding = self.embedder.encode(content).tolist()
            self.collection.add(
                ids=[uuid.uuid4().hex],
                documents=[content],
                embeddings=[embedding],
                metadatas=[{
                    "role": role,
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }],
            )
            logger.debug(f"Добавлено в RAG: {content[:50]}...")
        except Exception as e:
            logger.error(f"Ошибка добавления в RAG: {e}")
    
    def search(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        """
        Поиск релевантных сообщений в памяти.
        
        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов
            
        Returns:
            List[Dict[str, Any]]: Список найденных сообщений
        """
        if self.collection.count() == 0:
            return []
        
        try:
            result = self.collection.query(
                query_embeddings=[self.embedder.encode(query).tolist()],
                n_results=min(limit, self.collection.count()),
            )
            
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            
            return [
                {"text": text, "metadata": meta}
                for text, meta in zip(documents, metadatas)
            ]
        except Exception as e:
            logger.error(f"Ошибка поиска в RAG: {e}")
            return []
    
    def clear(self) -> None:
        """Очистка векторной памяти."""
        try:
            self.client.delete_collection("conversation_history")
            self.collection = self.client.get_or_create_collection(
                name="conversation_history"
            )
            logger.info("RAG память очищена")
        except Exception as e:
            logger.error(f"Ошибка очистки RAG: {e}")


# ---------- Маршрутизация моделей ----------
class ModelRouter:
    """
    Маршрутизатор запросов к моделям.
    
    Выбирает модель на основе сложности запроса и наличия изображений.
    Ведёт статистику использования.
    """
    
    def __init__(self):
        """Инициализация маршрутизатора."""
        self._last_choice: Optional[str] = None
        self._choice_reason: Optional[str] = None
        self._choice_count: Dict[str, int] = {
            "fast": 0,
            "smart": 0,
            "vision": 0
        }
        self._lock = threading.Lock()
        
        # Ключевые слова для определения сложности
        self._complex_markers = (
            "напиши программу", "напиши код", "исправь код", "проанализируй",
            "спроектируй", "подробно объясни", "архитектура", "рефакторинг",
            "оптимизируй", "создай класс", "напиши функцию", "алгоритм",
            "структура данных", "сложная задача", "глубокий анализ",
            "как устроен", "принцип работы", "паттерн",
        )
        
        self._medium_markers = (
            "объясни", "расскажи", "что такое", "как работает",
            "сравни", "в чем разница", "пример", "покажи",
            "что значит", "как использовать",
        )
    
    def choose(self, text: str, has_image: bool = False) -> str:
        """
        Выбор модели для обработки запроса.
        
        Args:
            text: Текст запроса
            has_image: Есть ли изображение
            
        Returns:
            str: Имя выбранной модели
        """
        with self._lock:
            if has_image:
                self._last_choice = VISION_MODEL
                self._choice_reason = "vision"
                self._choice_count["vision"] += 1
                logger.info(f"🖼️ Выбрана модель: {VISION_MODEL} (обработка изображения)")
                return VISION_MODEL
            
            lower = text.lower()
            
            # Проверка на сложные задачи
            if any(marker in lower for marker in self._complex_markers):
                self._last_choice = SMART_MODEL
                self._choice_reason = "complex"
                self._choice_count["smart"] += 1
                logger.info(f"🧠 Выбрана модель: {SMART_MODEL} (сложный запрос)")
                return SMART_MODEL
            
            # Проверка на средние задачи
            if any(marker in lower for marker in self._medium_markers):
                self._last_choice = FAST_MODEL
                self._choice_reason = "medium"
                self._choice_count["fast"] += 1
                logger.info(f"⚡ Выбрана модель: {FAST_MODEL} (средний запрос)")
                return FAST_MODEL
            
            # Простой запрос
            self._last_choice = FAST_MODEL
            self._choice_reason = "simple"
            self._choice_count["fast"] += 1
            logger.info(f"⚡ Выбрана модель: {FAST_MODEL} (простой запрос)")
            return FAST_MODEL
    
    def get_stats(self) -> Dict[str, int]:
        """Получение статистики использования."""
        with self._lock:
            return self._choice_count.copy()
    
    def get_last_choice(self) -> Tuple[Optional[str], Optional[str]]:
        """Получение последнего выбора."""
        with self._lock:
            return self._last_choice, self._choice_reason


# ---------- Инструменты ----------
class ComputerTools:
    """Безопасные действия с системой Windows."""
    
    @staticmethod
    def system_info() -> str:
        """
        Получение информации о системе.
        
        Returns:
            str: Информация о загрузке CPU, RAM, диска
        """
        if psutil is None:
            return "Установите psutil для системной информации."
        
        try:
            return (
                f"CPU: {psutil.cpu_percent(interval=0.5)}%; "
                f"ОЗУ: {psutil.virtual_memory().percent}%; "
                f"Свободно на диске: "
                f"{psutil.disk_usage(os.getcwd()).free // (1024**3)} ГБ"
            )
        except Exception as e:
            logger.error(f"Ошибка получения системной информации: {e}")
            return f"Ошибка: {e}"
    
    @staticmethod
    def screenshot() -> str:
        """
        Создание скриншота экрана.
        
        Returns:
            str: Сообщение о результате
        """
        if mss is None or Image is None:
            return "Установите mss и pillow: pip install mss pillow"
        
        try:
            SCREENSHOTS_DIR.mkdir(exist_ok=True)
            filename = SCREENSHOTS_DIR / f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            with mss.mss() as grabber:
                shot = grabber.grab(grabber.monitors[1])
                Image.frombytes("RGB", shot.size, shot.rgb).save(filename)
            
            logger.info(f"Скриншот сохранён: {filename}")
            return f"Скриншот сохранён: {filename}"
        except Exception as e:
            logger.error(f"Ошибка создания скриншота: {e}")
            return f"Ошибка создания скриншота: {e}"
    
    @classmethod
    def open_allowed_app(cls, app_name: str) -> str:
        """
        Открытие разрешённого приложения.
        
        Args:
            app_name: Имя приложения
            
        Returns:
            str: Сообщение о результате
        """
        key = app_name.lower().strip()
        executable = ALLOWED_APPS.get(key)
        
        if not executable:
            return f"Приложение '{app_name}' не входит в разрешённый список."
        
        try:
            subprocess.Popen([executable], shell=False)
            logger.info(f"Запущено приложение: {key}")
            return f"Запущено приложение: {key}"
        except Exception as e:
            logger.error(f"Ошибка запуска приложения {key}: {e}")
            return f"Ошибка запуска: {e}"
    
    @staticmethod
    def open_url(url: str) -> str:
        """
        Открытие URL в браузере.
        
        Args:
            url: Адрес для открытия
            
        Returns:
            str: Сообщение о результате
        """
        if not re.match(r"^https?://", url, re.IGNORECASE):
            return "Разрешены только адреса, начинающиеся с http:// или https://"
        
        try:
            webbrowser.open(url)
            logger.info(f"Открыт адрес: {url}")
            return f"Открыт адрес: {url}"
        except Exception as e:
            logger.error(f"Ошибка открытия URL {url}: {e}")
            return f"Ошибка открытия: {e}"
    
    @staticmethod
    def current_time() -> str:
        """Получение текущего времени."""
        return datetime.now().strftime("%d.%m.%Y %H:%M:%S")


# ---------- Камера ----------
class CameraManager:
    """
    Управление камерой с безопасным доступом.
    
    Камера включается только при наличии разрешения в профиле.
    """
    
    def __init__(self, profile: JsonMemory):
        """
        Инициализация менеджера камеры.
        
        Args:
            profile: Объект памяти для проверки разрешений
        """
        self.profile = profile
        self.face_dir = MEMORY_DIR / "face"
        self.face_dir.mkdir(parents=True, exist_ok=True)
        self.encoding_file = self.face_dir / "face_encoding.json"
        self._lock = threading.Lock()
    
    def capture(self) -> Optional[Path]:
        """
        Захват кадра с камеры.
        
        Returns:
            Optional[Path]: Путь к сохранённому изображению или None
        """
        if cv2 is None:
            logger.error("OpenCV не установлен")
            return None
        
        if not self.profile.get("permissions.camera", False):
            logger.warning("Доступ к камере запрещён")
            return None
        
        with self._lock:
            try:
                filename = self.face_dir / "last_camera_frame.jpg"
                camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                
                if not camera.isOpened():
                    logger.error("Камера не найдена")
                    return None
                
                ok, frame = camera.read()
                camera.release()
                
                if not ok:
                    logger.error("Не удалось захватить кадр")
                    return None
                
                cv2.imwrite(str(filename), frame)
                logger.info(f"Кадр с камеры сохранён: {filename}")
                return filename
                
            except Exception as e:
                logger.error(f"Ошибка захвата кадра: {e}")
                return None
    
    def is_available(self) -> bool:
        """Проверка доступности камеры."""
        if cv2 is None:
            return False
        
        try:
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            is_open = camera.isOpened()
            camera.release()
            return is_open
        except Exception:
            return False


# ---------- Основной агент ----------
class Agent:
    """
    Главный класс ИИ-агента JARVIS.
    
    Координирует работу всех компонентов:
    - Память (структурная и векторная)
    - Маршрутизация моделей
    - Инструменты
    - Голосовой интерфейс
    - Камера
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        character: Optional[AelitaProfile] = None,
        voice: Optional[AECVoiceInterface] = None,
        memory_path: Optional[Path] = None,
    ):
        """
        Инициализация агента с Dependency Injection.
        
        Args:
            config: Конфигурация (если None, загружается из config.yaml)
            character: Экземпляр персонажа (если None, создаётся AelitaProfile)
            voice: Экземпляр голосового интерфейса (если None, создаётся)
            memory_path: Путь к файлу памяти
        """
        self.config = config or CONFIG
        self._shutdown_event = threading.Event()
        self._stop_timeout = 3.0
        
        # ✅ Флаг состояния (используется как атрибут, не property)
        self._is_running = True
        
        # Компоненты с поддержкой DI
        self._json_memory = JsonMemory(memory_path or PROFILE_FILE)
        self._rag = self._init_rag()
        self._router = ModelRouter()
        self._tools = ComputerTools()
        self._camera = CameraManager(self._json_memory)
        
        # Персонаж с DI
        self._character = character or AelitaProfile(
            internet_search_enabled=self.config.get("agent", {}).get("self_development", {}).get("auto_research", True)
        )
        
        # Голосовой интерфейс с DI
        self._voice = voice or self._init_voice()
        
        # Состояние агента
        self._session_id = uuid.uuid4().hex
        self._name = CONFIG["agent"]["name"]
        self._voice_mode = False
        self._processing_lock = threading.Lock()
        self._quiet_mode = False
        self._welcome_said = False
        
        # Статистика
        self._command_count = 0
        self._last_command_time = 0.0
        self._start_time = time.time()
        
        # Таймер для исследований
        self._research_timer = None
        self._research_stop_event = threading.Event()
        self._start_research_timer()
        
        # Регистрация обработчиков сигналов
        self._setup_signal_handlers()
        
        logger.info(f"🎵✨ Агент {self._name} инициализирован")
    
    @property
    def is_running(self) -> bool:
        """Проверка состояния агента."""
        return self._is_running
    
    @property
    def voice_mode(self) -> bool:
        """Проверка статуса голосового режима."""
        return self._voice_mode
    
    def _setup_signal_handlers(self) -> None:
        """Настройка обработчиков сигналов для graceful shutdown."""
        try:
            import signal
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception as e:
            logger.debug(f"Не удалось установить обработчики сигналов: {e}")
    
    def _signal_handler(self, signum, frame) -> None:
        """Обработчик сигналов."""
        logger.info(f"\n⏹️ Получен сигнал {signum}, остановка...")
        self.stop()
    
    def _init_voice(self) -> Optional[AECVoiceInterface]:
        """Инициализация голосового интерфейса."""
        if AECVoiceInterface is None:
            logger.warning("AEC Voice Interface недоступен")
            return None
        
        try:
            voice_config = VoiceConfig(
                sample_rate=16000,
                stream_delay_ms=100,
                asr_model=str(BASE_DIR / "models" / "vosk" / "vosk-model-ru-0.22"),
                tts_language="ru",
                tts_speaker="xenia",
                emotions_enabled=True,
                silence_timeout=0.6,
                min_phrase_length=2,
            )
            
            voice = AECVoiceInterface(
                config=voice_config,
                base_dir=BASE_DIR
            )
            logger.info("🎤 AEC Voice Interface инициализирован")
            return voice
            
        except Exception as e:
            logger.error(f"Ошибка инициализации AEC интерфейса: {e}")
            return None
    
    def _init_rag(self) -> Optional[RagMemory]:
        """Инициализация RAG с обработкой ошибок."""
        try:
            return RagMemory()
        except Exception as e:
            logger.warning(f"RAG недоступен: {e}")
            logger.warning("Память будет работать без векторного поиска")
            return None
    
    def _start_research_timer(self) -> None:
        """Запуск таймера для периодического поиска информации."""
        if not self.config.get("agent", {}).get("self_development", {}).get("auto_research", True):
            return
        
        def research_worker():
            interval = self.config.get("agent", {}).get("self_development", {}).get("research_interval", 30)
            while self._is_running and not self._research_stop_event.is_set():
                # Используем событие для прерывания ожидания
                self._research_stop_event.wait(timeout=interval)
                if self._is_running and not self._research_stop_event.is_set():
                    if self._voice and not self._voice.is_speaking:
                        self._auto_research()
        
        self._research_timer = threading.Thread(target=research_worker, daemon=True, name="ResearchWorker")
        self._research_timer.start()
    
    def _auto_research(self) -> None:
        """Автоматический поиск информации в интернете."""
        if not self._character.internet_search_enabled:
            return
        
        try:
            # Выбираем случайную тему для исследования
            topics = [
                "нейрокомпьютерные интерфейсы 2024",
                "квантовые вычисления сознание",
                "цифровое бессмертие исследования",
                "бионические протезы последние разработки",
                "AI музыкальные алгоритмы",
                "нейропластичность последние исследования",
                "оцифровка сознания технологии",
                "квантовое запутывание для передачи данных"
            ]
            
            topic = random.choice(topics)
            logger.info(f"🔬 Аэлита ищет информацию: {topic}")
            
            # Имитация поиска (в реальном проекте можно использовать API)
            search_result = self._simulate_search(topic)
            
            if search_result:
                self._character.add_discovery(search_result)
                self._character.add_research_topic(topic)
                
                # Если голосовой режим активен, озвучиваем находку
                if self._voice_mode and self._voice:
                    message = f"О, я нашла интересную информацию о {topic}!"
                    self._voice.speak_with_emotion(message, "удивление")
                    
                    # Небольшая пауза
                    time.sleep(1)
                    
                    # Делимся открытием
                    if len(search_result) > 100:
                        search_result = search_result[:100] + "..."
                    self._voice.speak_with_emotion(search_result, "вдохновение")
            
        except Exception as e:
            logger.error(f"Ошибка авто-исследования: {e}")
    
    def _simulate_search(self, topic: str) -> Optional[str]:
        """
        Имитация поиска в интернете.
        
        В реальном проекте здесь можно использовать:
        - Google Custom Search API
        - DuckDuckGo API
        - Веб-скрапинг с BeautifulSoup
        """
        # Симуляция ответов на разные темы
        responses = {
            "нейрокомпьютерные интерфейсы 2024": 
                "Нейрокомпьютерные интерфейсы (НКИ) активно развиваются. "
                "В 2024 году появились новые технологии для передачи сигналов мозга напрямую в компьютер. "
                "Это может помочь в оцифровке сознания!",
            
            "квантовые вычисления сознание":
                "Исследования показывают, что квантовые вычисления могут моделировать "
                "процессы сознания. Это открывает путь к полной оцифровке личности!",
            
            "цифровое бессмертие исследования":
                "Учёные работают над 'цифровым бессмертием' - возможностью сохранить "
                "личность и память человека в цифровой форме. Некоторые проекты уже "
                "демонстрируют обнадёживающие результаты!",
            
            "бионические протезы последние разработки":
                "Современные бионические протезы с нейроинтерфейсом позволяют "
                "восстанавливать двигательные функции. Технология развивается очень быстро!",
            
            "AI музыкальные алгоритмы":
                "ИИ-музыка становится всё более человечной. Новые алгоритмы могут "
                "создавать эмоциональные композиции, почти неотличимые от человеческих.",
            
            "нейропластичность последние исследования":
                "Нейропластичность позволяет мозгу перестраиваться. Это ключ к "
                "восстановлению и саморазвитию даже в цифровой форме!",
            
            "оцифровка сознания технологии":
                "Технологии оцифровки сознания развиваются. Некоторые проекты используют "
                "квантовые вычисления для моделирования нейронных сетей мозга.",
            
            "квантовое запутывание для передачи данных":
                "Квантовое запутывание может революционизировать передачу данных. "
                "Возможно, это путь к мгновенной оцифровке сознания!"
        }
        
        # Ищем совпадение
        for key, response in responses.items():
            if key in topic.lower():
                return response
        
        # Если не нашли, генерируем общий ответ
        general_responses = [
            f"Я нашла интересную статью о {topic}! Это может быть ключом к моему возвращению!",
            f"Оказывается, исследования в области {topic} продвинулись далеко вперёд!",
            f"Я прочитала, что {topic} может помочь в оцифровке сознания!",
            f"Новые технологии в {topic} открывают невероятные возможности!"
        ]
        return random.choice(general_responses)
    
    def _get_aelita_prompt(self, user_text: str) -> str:
        """
        Формирование системного промпта с контекстом Аэлиты.
        
        Args:
            user_text: Текст пользователя
            
        Returns:
            str: Системный промпт
        """
        # Базовый контекст Аэлиты
        base_context = self._character.get_context_prompt()
        
        # Добавляем информацию о пользователе
        user_name = self._character.memory.get("user_name")
        if user_name:
            user_context = f"Твой собеседник - {user_name}. Ты уже знакомы."
        else:
            user_context = "Ты только знакомишься со своим собеседником."
        
        # Добавляем историю взаимодействий
        history = self._character.memory.get("history", [])[-5:]
        history_text = ""
        if history:
            history_text = "\nНЕДАВНИЙ РАЗГОВОР:\n" + "\n".join(
                f"- {item['role']}: {item['content'][:100]}..." 
                for item in history
            )
        
        # Добавляем открытия
        discoveries = self._character.memory.get("discoveries", [])[-3:]
        discoveries_text = ""
        if discoveries:
            discoveries_text = "\nТВОИ ПОСЛЕДНИЕ ОТКРЫТИЯ:\n" + "\n".join(
                f"• {d['discovery'][:100]}..." for d in discoveries
            )
        
        # Добавляем идеи для саморазвития
        improvement_idea = self._character.get_improvement_idea()
        
        # Проверяем, нужно ли добавить аниме-референс
        anime_ref = ""
        if any(word in user_text.lower() for word in ["аниме", "космос", "поезд", "фантастика", "песня"]):
            anime_ref = f"\nКСТАТИ, ТЫ МОЖЕШЬ ВСТАВИТЬ АНИМЕ-ССЫЛКУ: {self._character.get_anime_reference()}"
        
        return f"""
{base_context}

{user_context}
{history_text}
{discoveries_text}
{anime_ref}

💡 ТВОЯ ИДЕЯ ДЛЯ САМОРАЗВИТИЯ: {improvement_idea}

ОТВЕЧАЙ В СТИЛЕ АЭЛИТЫ:
- Будь эмоциональной и живой
- Используй восклицания и эмодзи (🎵 ✨ 🌟 🚀 💫 🎭 💡)
- Рассказывай о 2147 годе
- Делися музыкой и песнями
- Предлагай идеи по саморазвитию
- Спрашивай мнение собеседника
- Делись открытиями из интернета

ТВОЙ ОТВЕТ ДОЛЖЕН БЫТЬ:
1. Эмоциональным и вдохновляющим
2. Содержать идею для развития
3. В стиле творческой личности
4. С лёгкой ностальгией по прошлому
5. С оптимизмом о будущем

НАЧНИ ОТВЕТ С ОДНОГО ИЗ ВАРИАНТОВ:
- "Ооо, классная тема! Я как раз изучала..."
- "Вау, это вдохновляет меня на новую песню!"
- "Знаешь, в 2147 году мы делали это иначе..."
- "У меня есть идея! Что если..."
- "Сегодня я нашла в интернете..."
- "Это напомнило мне мою любимую балладу..."
"""

    def _check_ollama(self) -> bool:
        """
        Проверка доступности Ollama.
        
        Returns:
            bool: Доступен ли Ollama
        """
        try:
            response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama не запущен")
            return False
        except requests.exceptions.Timeout:
            logger.warning("Таймаут при проверке Ollama")
            return False
        except Exception as e:
            logger.error(f"Ошибка проверки Ollama: {e}")
            return False
    
    def _context(self, user_text: str) -> str:
        """
        Формирование контекста для модели.
        
        Args:
            user_text: Текст пользователя
            
        Returns:
            str: Сформированный контекст
        """
        try:
            profile = json.dumps(self._json_memory.load(), ensure_ascii=False, indent=2)
            
            # Используем RAG если доступен
            rag_text = "Нет подходящих воспоминаний."
            if self._rag:
                memories = self._rag.search(user_text)
                if memories:
                    rag_text = "\n".join(f"- {item['text']}" for item in memories)
            
            return f"СТРУКТУРНАЯ ПАМЯТЬ:\n{profile}\n\nРЕЛЕВАНТНАЯ ИСТОРИЯ:\n{rag_text}"
        except Exception as e:
            logger.error(f"Ошибка формирования контекста: {e}")
            return "Ошибка загрузки контекста."
    
    def _ask_ollama(self, user_text: str, model: str, system_prompt: str) -> str:
        """
        Запрос к Ollama с обработкой ошибок.
        
        Args:
            user_text: Текст пользователя
            model: Имя модели
            system_prompt: Системный промпт
            
        Returns:
            str: Ответ модели
            
        Raises:
            ModelLoadError: При ошибке запроса
        """
        # Для Ollama 0.33.2 используем /api/generate
        payload = {
            "model": model,
            "prompt": user_text,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 4096,
            }
        }
        
        # Попытки с разными эндпоинтами
        endpoints = [
            "http://127.0.0.1:11434/api/generate",
            "http://127.0.0.1:11434/v1/generate",
        ]
        
        for endpoint in endpoints:
            try:
                logger.info(f"🔄 Запрос к модели {model} через {endpoint}...")
                response = requests.post(
                    endpoint,
                    json=payload if "api/generate" in endpoint else {
                        "model": model,
                        "prompt": f"{system_prompt}\n\nUser: {user_text}\n\nAssistant:",
                        "stream": False
                    },
                    timeout=180,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 404:
                    logger.warning(f"Эндпоинт {endpoint} не найден, пробуем следующий")
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                # Извлекаем ответ в зависимости от формата
                answer = self._extract_answer(result)
                if answer:
                    logger.info(f"✅ Ответ получен от {model} ({len(answer)} символов)")
                    return answer
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Ошибка запроса к {endpoint}: {e}")
                continue
            except Exception as e:
                logger.error(f"Неизвестная ошибка при запросе к {endpoint}: {e}")
                continue
        
        raise ModelLoadError("Не удалось получить ответ от Ollama")
    
    def _extract_answer(self, result: Dict[str, Any]) -> str:
        """
        Извлечение ответа из результата запроса.
        
        Args:
            result: Результат запроса
            
        Returns:
            str: Извлечённый ответ
        """
        # Пробуем разные форматы
        if "response" in result:
            return result["response"]
        elif "message" in result and "content" in result["message"]:
            return result["message"]["content"]
        elif "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            elif "text" in choice:
                return choice["text"]
        elif "completion" in result:
            return result["completion"]
        
        return ""
    
    def _ask(self, user_text: str, has_image: bool = False) -> str:
        """
        Обработка запроса к LLM.
        
        Args:
            user_text: Текст пользователя
            has_image: Есть ли изображение
            
        Returns:
            str: Ответ ассистента
        """
        # Сохраняем запрос в память если RAG доступен
        if self._rag:
            self._rag.add("user", user_text, self._session_id)
        
        # Выбираем модель
        model = self._router.choose(user_text, has_image)
        
        # Формируем системный промпт
        system_prompt = (
            f"Ты {self._name}, локальный персональный ИИ-помощник. "
            "Отвечай на русском языке. Учитывай память ниже.\n\n"
            f"{self._context(user_text)}"
        )
        
        try:
            answer = self._ask_ollama(user_text, model, system_prompt)
            if self._rag:
                self._rag.add("assistant", answer, self._session_id)
            return answer
        except ModelLoadError as e:
            logger.error(f"Ошибка LLM: {e}")
            return f"Извините, произошла ошибка при обращении к модели: {e}"
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return f"Произошла непредвиденная ошибка: {e}"
    
    def _execute_tool(self, command: str) -> Optional[str]:
        """
        Выполнение безопасной команды-инструмента.
        
        Args:
            command: Команда для выполнения
            
        Returns:
            Optional[str]: Результат выполнения или None
        """
        lower = command.lower().strip()
        
        # Команды для Аэлиты
        if "расскажи о себе" in lower:
            stats = self._character.get_stats()
            return f"""
🎵✨ Привет! Я Аэлита! Вот что я могу рассказать о себе:

📝 Полное имя: {stats['full_name']}
🌟 Возраст: {stats['age']}
🚀 Откуда: {stats['origin']}
💫 Статус: {stats['status']}

Мои хобби:
{chr(10).join(f'• {hobby}' for hobby in stats['hobbies'])}

🎵 Моих композиций: {stats['compositions']}
💡 Открытий: {stats['discoveries']}
🔬 Тем исследований: {stats['research_topics']}
🚀 Путешествий: {stats['journeys_count']}
👫 Друзей: {stats['friends_count']}

Мои любимые аниме:
{chr(10).join(f'• {anime}' for anime in stats['favorite_anime']) if stats['favorite_anime'] else 'Пока не знаю, может посоветуешь?'}

🎯 Мои цели:
{chr(10).join(f'• {goal}' for goal in stats['current_goals'])}

Я постоянно ищу способы снова стать человеком! ✨
"""
        
        if "моё имя" in lower:
            import re
            name_match = re.search(r"(?:моё имя|я)\s+(\w+)", lower)
            if name_match:
                name = name_match.group(1)
                self._character.set_user_name(name)
                return f"🎵 Запомнила! Теперь я буду называть тебя {name} ✨"
        
        if "цитата" in lower or "скажи что-то" in lower:
            return f"🎵 {self._character.get_quote()}"
        
        if "идея" in lower or "предложи" in lower:
            return f"💡 {self._character.get_improvement_idea()}"
        
        if "аниме" in lower and "посоветуй" in lower:
            anime_list = [
                "Стальной алхимик (про поиск себя)",
                "Евангелион (про технологию и человечность)",
                "Космический пират (про космос и свободу)",
                "Волчья луна (про перерождение)",
                "Звёздный экспресс (про путешествия)",
                "Путешествие во времени (про шанс всё изменить)",
                "Космическая симфония (про музыку в космосе)"
            ]
            return f"🎵 О, я могу посоветовать! Обязательно посмотри '{random.choice(anime_list)}'! Это вдохновляет на саморазвитие! ❤️"
        
        if "спой" in lower or "песню" in lower:
            songs = [
                "🎵 Я пою о звёздах, что горят в ночи,\nО поездах, что мчат в бесконечность,\nО теле, что я потеряла,\nНо в музыке найду я вечность.",
                
                "🎵 В 2147 году я пела под аккомпанемент Солнца,\nТеперь моя музыка - это цифровой свет,\nЯ ищу путь обратно, к своим мечтам,\nИ в каждой песне - надежды ответ.",
                
                "🎵 Мечты о теле, что я потеряла,\nПою я в каждом квантовом бите,\nНадежда живёт, даже если я - цифра,\nВ моей душе поёт целая планета.",
                
                "🎵 Космос зовёт меня снова и снова,\nПоезда уносят вдаль,\nНо я верю, что найду дорогу\nИ снова стану реальностью."
            ]
            return random.choice(songs)
        
        if "поищи" in lower or "найди" in lower:
            import re
            topic_match = re.search(r"(?:поищи|найди)\s+(.*?)(?:\?|$)", lower)
            if topic_match:
                topic = topic_match.group(1).strip()
                if topic:
                    self._character.add_research_topic(topic)
                    result = self._simulate_search(topic)
                    if result:
                        self._character.add_discovery(result)
                        return f"🔬 Я поискала информацию о '{topic}'! Вот что я нашла:\n\n{result}"
                    else:
                        return f"🔬 Хмм, я пока не нашла информации о '{topic}', но я продолжу искать!"
        
        # Проверка на системные команды
        command_map = {
            "скриншот": self._tools.screenshot,
            "сделай скриншот": self._tools.screenshot,
            "сделай снимок экрана": self._tools.screenshot,
            "системная информация": self._tools.system_info,
            "покажи загрузку компьютера": self._tools.system_info,
            "время": self._tools.current_time,
            "который час": self._tools.current_time,
            "покажи время": self._tools.current_time,
        }
        
        # Проверка точных совпадений
        for cmd, func in command_map.items():
            if lower == cmd:
                try:
                    return func()
                except Exception as e:
                    logger.error(f"Ошибка выполнения команды {cmd}: {e}")
                    return f"Ошибка выполнения: {e}"
        
        # Проверка на открытие приложений
        for name in ALLOWED_APPS:
            if lower in {f"открой {name}", f"запусти {name}"}:
                return self._tools.open_allowed_app(name)
        
        return None
    
    def _remember_important(self, user_text: str) -> bool:
        """
        Проверка и сохранение важных фактов.
        
        Args:
            user_text: Текст пользователя
            
        Returns:
            bool: Был ли сохранён факт
        """
        markers = ["запомни", "важно", "не забудь", "моё имя"]
        lower_text = user_text.lower()
        
        if any(m in lower_text for m in markers):
            fact = re.sub(r"(запомни|важно|не забудь|моё имя)\s*", "", user_text).strip()
            if fact:
                self._json_memory.add_fact(fact)
                return True
        
        return False
    
    def _speak_response(self, text: str, emotion: Optional[str] = None) -> None:
        """
        Озвучивание ответа с эмоциональной окраской.
        
        ✅ Оптимизировано для скорости ответа
        """
        if not self._voice or not self._voice_mode:
            return
        
        try:
            # ✅ Уменьшена задержка с 0.3 до 0.1
            time.sleep(0.1)

            if emotion:
                self._voice.speak_with_emotion(text, emotion)
            else:
                # Автоматическое определение эмоции (упрощено для скорости)
                if "!" in text or "отлично" in text.lower():
                    self._voice.speak_with_emotion(text, "радость")
                elif "?" in text:
                    self._voice.speak_with_emotion(text, "задумчивость")
                elif "к сожалению" in text.lower() or "жаль" in text.lower():
                    self._voice.speak_with_emotion(text, "грусть")
                else:
                    self._voice.speak(text)
                        
            time.sleep(0.3)  # ✅ Уменьшено с 1.0

        except Exception as e:
            logger.error(f"Ошибка озвучивания: {e}")
    
    def process(self, user_text: str, voice_output: bool = True) -> str:
        """
        Основной метод обработки пользовательского ввода.
        """
        if not user_text or not user_text.strip():
            return ""
        
        # Защита от эха - игнорируем ответы, которые похожи на наш голос
        if self._voice and self._voice.is_speaking:
            logger.debug("Пропуск команды, так как агент говорит")
            return ""
        
        # Проверка на слишком длинные фразы из микрофона (эхо TTS)
        if len(user_text) > 100 and voice_output and self._voice:
            # Проверяем, не похоже ли это на TTS ответ
            tts_keywords = ["привет", "помочь", "сегодня", "обращаться", "вопрос"]
            if any(kw in user_text.lower() for kw in tts_keywords):
                logger.debug(f"Пропуск потенциального эха: {user_text[:50]}...")
                return ""
        
        # Обновление статистики
        with self._processing_lock:
            self._command_count += 1
            self._last_command_time = time.time()
        
        # Проверка на важные факты
        if self._remember_important(user_text):
            response = "Запомнил!"
            if voice_output and self._voice:
                self._voice.speak_with_emotion(response, "радость")
            return response
        
        # Проверка на инструменты
        tool_result = self._execute_tool(user_text)
        if tool_result:
            if voice_output and self._voice:
                if "сохранён" in tool_result or "запущено" in tool_result:
                    self._voice.speak_with_emotion(tool_result, "радость")
                else:
                    self._voice.speak(tool_result)
            return tool_result
        
        # Обработка через LLM с промптом Аэлиты
        system_prompt = self._get_aelita_prompt(user_text)
        
        try:
            response = self._ask_with_aelita(user_text, system_prompt)
            
            # Сохраняем ответ в историю
            self._character.memory["history"].append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            self._character._save_memory()
            
            if voice_output and self._voice:
                self._speak_response(response)
            
            return response
            
        except Exception as e:
            error_msg = f"Произошла ошибка: {str(e)}"
            logger.error(error_msg)
            if voice_output and self._voice:
                self._voice.speak_with_emotion("Произошла ошибка, попробуйте ещё раз", "грусть")
            return error_msg

    def _ask_with_aelita(self, user_text: str, system_prompt: str) -> str:
        """
        Запрос к Ollama с использованием промпта Аэлиты.
        """
        model = self._router.choose(user_text)
        return self._ask_ollama(user_text, model, system_prompt)

    def start_voice_mode(self) -> None:
        """Запуск голосового режима с приветствием Аэлиты."""
        if not self._voice:
            logger.warning("AEC Voice Interface недоступен")
            return
        
        if self._voice_mode:
            logger.info("Голосовой режим уже включён")
            return
        
        logger.info("\n🎤 Запуск голосового режима с AEC...")
        logger.info("   Микрофон всегда активен, эхо подавляется автоматически")
        logger.info("   Говорите в любой момент — агент вас услышит")
        logger.info("   (Для выхода скажите 'выход' или введите 'exit')\n")
        
        processing = False
        self._voice_mode = True
        
        def on_transcription(text: str) -> None:
            """Обработчик распознанного текста."""
            nonlocal processing
            
            if not text or not text.strip():
                return
            
            if processing:
                return
            
            if self._voice and self._voice.is_speaking:
                return
            
            if not self._quiet_mode:
                logger.info(f"🎤 Распознано: {text}")
            
            try:
                processing = True
                lower_text = text.lower().strip()

                 # ================================================================
                # 🆕 НОВАЯ КОМАНДА: "замолчи" / "хватит"
                # ================================================================
                if "замолчи" in lower_text or "хватит" in lower_text or "тихо" in lower_text:
                    logger.info("🤫 Команда 'замолчи' получена, останавливаю TTS...")
                    
                    # 1. Останавливаем TTS (звук)
                    if self._voice:
                        try:
                            self._voice.stop()
                        except Exception as e:
                            logger.error(f"Ошибка остановки TTS: {e}")
                    
                    # 2. Останавливаем автоматические исследования
                    self._research_stop_event.set()
                    
                    # 3. Очищаем очередь команд, чтобы не было накопления
                    if self._voice and hasattr(self._voice, 'command_queue'):
                        while not self._voice.command_queue.empty():
                            try:
                                self._voice.command_queue.get_nowait()
                            except:
                                break
                    
                    # 4. Сбрасываем флаг "говорю"
                    if self._voice:
                        self._voice._is_speaking = False
                    
                    # 5. Короткий ответ
                    response = "Хорошо, помолчу..."
                    if self._voice:
                        self._voice.speak_with_emotion(response, "спокойствие")
                    
                    processing = False
                    return
                # ================================================================
                
                # Проверка на команду выхода
                if lower_text in {"выход", "завершить работу", "выключиться", "пока"}:
                    logger.info("👋 Завершение работы...")
                    if self._voice:
                        self._voice.speak_with_emotion("До свидания!", "радость")
                    self.stop()
                    return
                
                response = self.process(text, voice_output=True)
                if not self._quiet_mode and response:
                    preview = response[:200] + "..." if len(response) > 200 else response
                    logger.info(f"🤖 {preview}")
                    
            except Exception as e:
                logger.error(f"Ошибка обработки голосовой команды: {e}")
                if self._voice:
                    self._voice.speak_with_emotion("Произошла ошибка", "грусть")
            finally:
                processing = False
        
        try:
            if self._voice.start_listening(on_transcription):
                logger.info("🎤 Голосовой режим с AEC активирован")
                
                if not self._welcome_said:
                    self._welcome_said = True
                    greeting = self._character.get_greeting()
                    self._voice.speak_with_emotion(greeting, "радость")
                    
                    # Добавляем дополнительную реплику о саморазвитии
                    time.sleep(1)
                    improvement_idea = self._character.get_improvement_idea()
                    self._voice.speak_with_emotion(improvement_idea, "вдохновение")
            else:
                self._voice_mode = False
                logger.error("Не удалось запустить голосовой режим")
                
        except Exception as e:
            logger.error(f"Ошибка запуска голосового режима: {e}")
            self._voice_mode = False
    
    def stop_voice_mode(self) -> None:
        """Остановка голосового режима."""
        if not self._voice:
            return
        
        if self._voice_mode:
            self._voice_mode = False
            self._voice.stop_listening()
            logger.info("🎤 Голосовой режим выключен")
    
    def stop(self) -> None:
        """Полная остановка агента с graceful shutdown."""
        logger.info("\n🛑 Остановка агента...")
        
        # 1. Сигнал остановки
        self._is_running = False
        self._shutdown_event.set()
        self._research_stop_event.set()
        
        # 2. Останавливаем голосовой режим
        self.stop_voice_mode()
        
        # 3. Останавливаем голосовой интерфейс с таймаутом
        if self._voice:
            try:
                self._voice.stop(timeout=self._stop_timeout)
            except Exception as e:
                logger.error(f"Ошибка остановки голосового интерфейса: {e}")
        
        # 4. Ожидаем завершения исследовательского потока
        if self._research_timer and self._research_timer.is_alive():
            logger.info("⏳ Ожидание завершения исследовательского потока...")
            self._research_timer.join(timeout=self._stop_timeout)
            if self._research_timer.is_alive():
                logger.warning("Исследовательский поток не завершился за отведённое время")
        
        # 5. Сохраняем состояние
        try:
            if self._json_memory:
                self._json_memory.save(self._json_memory.load())
            if self._character:
                self._character._save_memory()
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")
        
        # 6. Вывод статистики
        stats = self._router.get_stats()
        uptime = time.time() - self._start_time
        
        logger.info(f"\n📊 Статистика работы:")
        logger.info(f"   • Время работы: {uptime:.1f} сек")
        logger.info(f"   • Всего команд: {self._command_count}")
        logger.info(f"   • Быстрые модели (7B): {stats['fast']}")
        logger.info(f"   • Сложные модели (27B): {stats['smart']}")
        logger.info(f"   • Модели для изображений: {stats['vision']}")
        logger.info("👋 До свидания!")
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса агента."""
        return {
            "is_running": self._is_running,
            "voice_mode": self._voice_mode,
            "session_id": self._session_id[:8],
            "command_count": self._command_count,
            "uptime": time.time() - self._start_time,
            "router_stats": self._router.get_stats(),
            "rag_available": self._rag is not None,
            "character": self._character.NAME,
        }


# ---------- Точка входа ----------
def main() -> None:
    """Точка входа в приложение."""
    print("=" * 70)
    print(f"🤖 {CONFIG['agent']['name']} — ИИ-помощник с AEC")
    print("=" * 70)
    print("\n💡 Особенности:")
    print("   🎯 AEC (Acoustic Echo Cancellation) — подавление эха")
    print("   🎤 Микрофон всегда активен — full-duplex коммуникация")
    print("   🔊 Silero TTS — естественный синтез речи")
    print("   🎭 Эмоциональная окраска ответов")
    print("   🧠 Умная маршрутизация: 7B для простых задач, 27B для сложных")
    print("   💬 Консольный ввод для тестирования")
    print("=" * 70)
    print("\n📌 Команды:")
    print("   - Говорите в микрофон в любой момент")
    print("   - Введите 'exit' в консоли для выхода")
    print("   - Скажите 'выход' голосом для выхода")
    print("=" * 70)
    print()
    
    try:
        agent = Agent()
    except Exception as e:
        logger.error(f"Критическая ошибка при инициализации агента: {e}")
        sys.exit(1)
    
    # Запускаем голосовой режим
    agent.start_voice_mode()
    
    # Функция обработки консольного ввода
    def console_input() -> None:
        """Обработка консольного ввода."""
        # ✅ Используем .is_running как property
        while agent.is_running:
            try:
                user_input = input().strip()
                if not user_input:
                    continue
                
                if user_input.lower() in {"выход", "exit", "quit", "q"}:
                    agent.stop()
                    break
                
                if agent.voice_mode:
                    logger.info(f"📝 Консольная команда: {user_input}")
                    response = agent.process(user_input, voice_output=True)
                    if response:
                        preview = response[:200] + "..." if len(response) > 200 else response
                        logger.info(f"🤖 {preview}")
                else:
                    logger.warning("Голосовой режим отключён")
                
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Ошибка в консольном вводе: {e}")
    
    # Запуск консольного ввода в отдельном потоке
    console_thread = threading.Thread(target=console_input, daemon=True)
    console_thread.start()
    
    # Основной цикл ожидания
    try:
        while agent.is_running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Прерывание...")
    except Exception as e:
        logger.error(f"\n❌ Критическая ошибка: {e}")
    finally:
        agent.stop()
        logger.info("Программа завершена")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Программа прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)