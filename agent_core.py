"""
agent_core.py
Ядро локального ИИ-агента JARVIS для Windows.
Работает в фоновом режиме с постоянным прослушиванием.
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
from typing import Any, Optional

import requests
import yaml

# ---------- Определяем базовую директорию ----------
BASE_DIR = Path(__file__).resolve().parent

# ---------- Загрузка конфигурации ----------
with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# ---------- Импорт зависимостей с проверкой ----------
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    chromadb = None
    SentenceTransformer = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mss
    from PIL import Image
except ImportError:
    mss = None
    Image = None

try:
    import psutil
except ImportError:
    psutil = None

# ---------- Импорт голосового интерфейса ----------
try:
    from voice_interface import VoiceInterface
    print("✅ Голосовой интерфейс загружен")
except ImportError as e:
    print(f"⚠️ Ошибка загрузки голосового интерфейса: {e}")
    VoiceInterface = None

# ---------- Пути ----------
MEMORY_DIR = BASE_DIR / "memory"
PROFILE_FILE = MEMORY_DIR / CONFIG["memory"]["profile_file"]
CHROMA_DIR = MEMORY_DIR / CONFIG["memory"]["chroma_dir"]
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

OLLAMA_URL = CONFIG["ollama"]["url"]
FAST_MODEL = CONFIG["ollama"]["fast_model"]
SMART_MODEL = CONFIG["ollama"]["smart_model"]
VISION_MODEL = CONFIG["ollama"]["vision_model"]
ALLOWED_APPS = CONFIG["tools"]["allowed_apps"]

# ---------- Профиль по умолчанию ----------
DEFAULT_PROFILE: dict[str, Any] = {
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
    """Структурированная долговременная память агента."""

    def __init__(self, path: Path = PROFILE_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.save(deepcopy(DEFAULT_PROFILE))

    def load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            profile = deepcopy(DEFAULT_PROFILE)
            self.save(profile)
            return profile

    def save(self, profile: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def set_name(self, name: str) -> None:
        profile = self.load()
        profile["user"]["name"] = name.strip()
        self.save(profile)

    def add_fact(self, fact: str) -> None:
        profile = self.load()
        facts = profile.setdefault("important_facts", [])
        if fact not in facts:
            facts.append(fact)
        self.save(profile)

    def add_project(self, name: str, description: str = "") -> None:
        profile = self.load()
        projects = profile.setdefault("projects", [])
        projects.append({"name": name, "description": description})
        self.save(profile)

    def clear(self) -> None:
        self.save(deepcopy(DEFAULT_PROFILE))


# ---------- Векторная память ----------
class RagMemory:
    """Векторная память для всей истории общения."""

    def __init__(self):
        if chromadb is None or SentenceTransformer is None:
            raise RuntimeError(
                "Установите: pip install chromadb sentence-transformers"
            )
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name="conversation_history"
        )
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def add(self, role: str, content: str, session_id: str) -> None:
        if not content.strip():
            return
        self.collection.add(
            ids=[uuid.uuid4().hex],
            documents=[content],
            embeddings=[self.embedder.encode(content).tolist()],
            metadatas=[{
                "role": role,
                "session_id": session_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }],
        )

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        if self.collection.count() == 0:
            return []
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

    def clear(self) -> None:
        try:
            self.client.delete_collection("conversation_history")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="conversation_history"
        )


# ---------- Маршрутизация моделей ----------
class ModelRouter:
    """Выбирает модель по типу задачи."""

    def choose(self, text: str, has_image: bool = False) -> str:
        if has_image:
            return VISION_MODEL
        lower = text.lower()
        complex_markers = (
            "напиши программу", "исправь код", "проанализируй",
            "спроектируй", "подробно объясни", "архитектура",
        )
        if any(marker in lower for marker in complex_markers):
            return SMART_MODEL
        return FAST_MODEL


# ---------- Инструменты ----------
class ComputerTools:
    """Безопасные действия Windows."""

    @staticmethod
    def system_info() -> str:
        if psutil is None:
            return "Установите psutil для системной информации."
        return (
            f"CPU: {psutil.cpu_percent(interval=0.5)}%; "
            f"ОЗУ: {psutil.virtual_memory().percent}%; "
            f"Свободно на диске: "
            f"{psutil.disk_usage(os.getcwd()).free // (1024**3)} ГБ"
        )

    @staticmethod
    def screenshot() -> str:
        if mss is None or Image is None:
            return "Установите mss и pillow: pip install mss pillow"
        SCREENSHOTS_DIR.mkdir(exist_ok=True)
        filename = SCREENSHOTS_DIR / f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with mss.mss() as grabber:
            shot = grabber.grab(grabber.monitors[1])
            Image.frombytes("RGB", shot.size, shot.rgb).save(filename)
        return f"Скриншот сохранён: {filename}"

    @classmethod
    def open_allowed_app(cls, app_name: str) -> str:
        key = app_name.lower().strip()
        executable = ALLOWED_APPS.get(key)
        if not executable:
            return "Это приложение не входит в разрешённый список."
        subprocess.Popen([executable], shell=False)
        return f"Запущено приложение: {key}"

    @staticmethod
    def open_url(url: str) -> str:
        if not re.match(r"^https?://", url, re.IGNORECASE):
            return "Разрешены только адреса, начинающиеся с http:// или https://"
        webbrowser.open(url)
        return f"Открыт адрес: {url}"

    @staticmethod
    def current_time() -> str:
        return datetime.now().strftime("%d.%m.%Y %H:%M:%S")


# ---------- Камера ----------
class CameraManager:
    """Камера включается только вызовом методов пользователем."""

    def __init__(self, profile: JsonMemory):
        self.profile = profile
        self.face_dir = MEMORY_DIR / "face"
        self.face_dir.mkdir(parents=True, exist_ok=True)
        self.encoding_file = self.face_dir / "face_encoding.json"

    def capture(self) -> Optional[Path]:
        if cv2 is None:
            print("Установите opencv-python для работы с камерой.")
            return None
        if not self.profile.load()["permissions"].get("camera", False):
            print("Камера запрещена. Разрешите её в profile.json.")
            return None

        filename = self.face_dir / "last_camera_frame.jpg"
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            print("Камера не найдена.")
            return None
        ok, frame = camera.read()
        camera.release()
        if not ok:
            return None
        cv2.imwrite(str(filename), frame)
        return filename


# ---------- Основной агент ----------
class Agent:
    def __init__(self):
        self.config = CONFIG
        self.json_memory = JsonMemory()
        self.rag = RagMemory()
        self.router = ModelRouter()
        self.tools = ComputerTools()
        self.camera = CameraManager(self.json_memory)
        
        # Инициализация голосового интерфейса
        if VoiceInterface is not None:
            try:
                self.voice = VoiceInterface(wake_words=["джарвис", "jarvis"])
                print("🎤 Голосовой интерфейс: Vosk + Piper")
            except Exception as e:
                print(f"⚠️ Ошибка инициализации голоса: {e}")
                self.voice = None
        else:
            self.voice = None
            print("⚠️ Голосовой интерфейс недоступен")
        
        self.session_id = uuid.uuid4().hex
        self.name = CONFIG["agent"]["name"]
        self.voice_mode = False
        self.is_running = True
        self.processing_lock = threading.Lock()
        
        # Флаг для вывода в консоль
        self.quiet_mode = False
        
        # Статистика
        self.command_count = 0
        self.last_command_time = 0

    def context(self, user_text: str) -> str:
        profile = json.dumps(self.json_memory.load(), ensure_ascii=False, indent=2)
        memories = self.rag.search(user_text)
        rag_text = "\n".join(f"- {item['text']}" for item in memories) or "Нет подходящих воспоминаний."
        return f"СТРУКТУРНАЯ ПАМЯТЬ:\n{profile}\n\nРЕЛЕВАНТНАЯ ИСТОРИЯ:\n{rag_text}"

    def ask(self, user_text: str, has_image: bool = False) -> str:
        self.rag.add("user", user_text, self.session_id)
        model = self.router.choose(user_text, has_image)
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Ты {self.name}, локальный персональный ИИ-помощник. "
                        "Отвечай на русском языке. Учитывай память ниже.\n\n"
                        f"{self.context(user_text)}"
                    ),
                },
                {"role": "user", "content": user_text},
            ],
            "stream": False,
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        answer = response.json()["message"]["content"]
        self.rag.add("assistant", answer, self.session_id)
        return answer

    def execute_safe_command(self, command: str) -> Optional[str]:
        lower = command.lower().strip()
        if lower in {"скриншот", "сделай скриншот", "сделай снимок экрана"}:
            return self.tools.screenshot()
        if lower in {"системная информация", "покажи загрузку компьютера"}:
            return self.tools.system_info()
        if lower in {"время", "который час", "покажи время"}:
            return self.tools.current_time()
        for name in ALLOWED_APPS:
            if lower in {f"открой {name}", f"запусти {name}"}:
                return self.tools.open_allowed_app(name)
        return None

    def remember_important(self, user_text: str) -> bool:
        markers = ["запомни", "важно", "не забудь", "моё имя"]
        if any(m in user_text.lower() for m in markers):
            fact = re.sub(r"(запомни|важно|не забудь|моё имя)\s*", "", user_text).strip()
            if fact:
                self.json_memory.add_fact(fact)
                return True
        return False

    def process(self, user_text: str, voice_output: bool = True) -> str:
        if not user_text.strip():
            return ""

        with self.processing_lock:
            self.command_count += 1
            self.last_command_time = time.time()

        # Проверка на запоминание
        if self.remember_important(user_text):
            response = "Запомнил!"
            if voice_output and self.voice:
                self.voice.speak(response, async_mode=True)
            return response

        # Проверка на инструменты
        tool_result = self.execute_safe_command(user_text)
        if tool_result:
            if voice_output and self.voice:
                self.voice.speak(tool_result, async_mode=True)
            return tool_result

        # Обычный запрос к ИИ
        response = self.ask(user_text)
        if voice_output and self.voice:
            self.voice.speak(response, async_mode=True)
        return response

    def start_voice_mode(self):
        """Запускает голосовой режим с постоянным прослушиванием."""
        if not self.voice:
            print("⚠️ Голосовой интерфейс недоступен")
            return

        if self.voice_mode:
            print("🎤 Голосовой режим уже включён")
            return

        print("\n🎤 Запуск голосового режима с постоянным прослушиванием...")
        print("   Говорите в микрофон, я буду отвечать на команды")
        print("   (Для выхода введите 'exit' в консоли)\n")
        
        def on_wake():
            """Вызывается при обнаружении речи."""
            # В режиме постоянного прослушивания просто показываем индикатор
            if not self.quiet_mode:
                print("🎤 Слышу вас...")
        
        def on_transcription(text):
            """Вызывается при распознавании команды."""
            if not text or not text.strip():
                return
                
            if not self.quiet_mode:
                print(f"🎤 Распознано: {text}")
            
            try:
                # Проверяем специальные команды выхода
                lower_text = text.lower().strip()
                if lower_text in {"выход", "завершить работу", "выключиться"}:
                    print("👋 Завершение работы по голосовой команде...")
                    self.stop()
                    return
                
                # Обрабатываем команду
                response = self.process(text, voice_output=True)
                if not self.quiet_mode:
                    print(f"🤖 {response}")
                    
            except Exception as e:
                error_msg = f"Произошла ошибка: {str(e)}"
                print(f"❌ {error_msg}")
                if self.voice:
                    self.voice.speak("Произошла ошибка", async_mode=True)

        # Запускаем голосовой режим с постоянным прослушиванием
        try:
            self.voice.start_voice_mode(
                on_wake=on_wake,
                on_transcription=on_transcription
            )
            self.voice_mode = True
            print("🎤 Голосовой режим активирован. Говорите в микрофон.")
            
        except Exception as e:
            print(f"❌ Ошибка в голосовом режиме: {e}")
            self.voice_mode = False

    def stop_voice_mode(self):
        """Останавливает голосовой режим."""
        if not self.voice:
            return

        if self.voice_mode:
            self.voice_mode = False
            self.voice.stop_voice_mode()
            print("🎤 Голосовой режим выключен")

    def stop(self):
        """Полная остановка агента."""
        print("\n🛑 Остановка агента...")
        self.is_running = False
        self.stop_voice_mode()
        
        if self.voice:
            self.voice.stop()
        
        print(f"📊 Статистика: обработано {self.command_count} команд")
        print("👋 До свидания!")


# ---------- Точка входа ----------
def main():
    print("=" * 50)
    print(f"🤖 {CONFIG['agent']['name']} — локальный ИИ-помощник")
    print("=" * 50)
    print("\n💡 Режим работы: постоянное прослушивание микрофона")
    print("   Говорите в микрофон, агент будет отвечать голосом")
    print("   В консоли введите 'exit' для завершения")
    print("   Скажите 'выход' голосом для завершения")
    print("=" * 50)
    print()

    agent = Agent()

    # Автоматически запускаем голосовой режим
    agent.start_voice_mode()

    # Поток для консольного ввода
    def console_input():
        while agent.is_running:
            try:
                user_input = input().strip()
                if not user_input:
                    continue

                if user_input.lower() in {"выход", "exit", "quit"}:
                    agent.stop()
                    break

                # Если введена команда в консоль, обрабатываем её
                if agent.voice_mode:
                    print(f"📝 Консольная команда: {user_input}")
                    response = agent.process(user_input, voice_output=True)
                    print(f"🤖 {response}")
                else:
                    print("⚠️ Голосовой режим отключён")

            except EOFError:
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")

    # Запускаем поток консольного ввода
    console_thread = threading.Thread(target=console_input, daemon=True)
    console_thread.start()

    try:
        # Ожидаем завершения
        while agent.is_running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n⏹️ Прерывание...")
        agent.stop()


if __name__ == "__main__":
    main()