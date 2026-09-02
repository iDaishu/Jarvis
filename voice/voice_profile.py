# voice/voice_profile.py
"""
Профиль голоса пользователя.

Хранит настройки и историю голосовых взаимодействий.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import threading

from utils.logger import LoggerFactory

logger = LoggerFactory.get_logger("voice_profile")


class VoiceProfile:
    """
    Профиль голоса пользователя.
    
    Хранит:
    - Имя пользователя
    - Настройки голоса
    - Историю взаимодействий
    - Статистику использования
    """
    
    def __init__(self, profile_path: Path):
        """
        Инициализация профиля голоса.
        
        Args:
            profile_path: Путь к файлу профиля
        """
        self.profile_path = profile_path
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self._profile = self._load()
        self._lock = threading.Lock()
        
        logger.info(f"Профиль голоса загружен: {profile_path}")
    
    def _load(self) -> Dict[str, Any]:
        """
        Загрузка профиля из файла.
        
        Returns:
            Dict[str, Any]: Данные профиля
        """
        if self.profile_path.exists():
            try:
                with open(self.profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка парсинга профиля: {e}")
            except Exception as e:
                logger.error(f"Ошибка загрузки профиля: {e}")
        
        # Профиль по умолчанию
        return {
            "name": None,
            "voice_encoding": None,
            "language": "ru",
            "speaking_rate": 1.0,
            "pitch_shift": 0.0,
            "sample_count": 0,
            "last_updated": None,
            "calibrated": False,
            "history": [],
            "preferences": {
                "tts_speaker": "xenia",
                "emotions_enabled": True,
                "noise_reduction": True,
            }
        }
    
    def save(self) -> None:
        """Сохранение профиля в файл."""
        with self._lock:
            self._profile["last_updated"] = datetime.now().isoformat()
            try:
                with open(self.profile_path, 'w', encoding='utf-8') as f:
                    json.dump(self._profile, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Ошибка сохранения профиля: {e}")
    
    def get_name(self) -> Optional[str]:
        """Получение имени пользователя."""
        return self._profile.get("name")
    
    def set_name(self, name: str) -> None:
        """Установка имени пользователя."""
        with self._lock:
            self._profile["name"] = name.strip()
            self.save()
        logger.info(f"Имя пользователя обновлено: {name}")
    
    def add_sample(self, text: str, audio: np.ndarray, 
                   emotion: str = "neutral") -> bool:
        """
        Добавление образца речи.
        
        Args:
            text: Распознанный текст
            audio: Аудио массив
            emotion: Эмоциональная окраска
            
        Returns:
            bool: Успешность добавления
        """
        try:
            with self._lock:
                self._profile["sample_count"] += 1
                self._profile["calibrated"] = True
                
                sample = {
                    "timestamp": datetime.now().isoformat(),
                    "text": text[:100],
                    "emotion": emotion,
                    "sample_number": self._profile["sample_count"],
                    "duration": len(audio) / 16000 if len(audio) > 0 else 0
                }
                
                self._profile["history"].append(sample)
                
                # Ограничение истории
                if len(self._profile["history"]) > 100:
                    self._profile["history"] = self._profile["history"][-100:]
                
                self.save()
                return True
                
        except Exception as e:
            logger.error(f"Ошибка добавления образца: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики профиля.
        
        Returns:
            Dict[str, Any]: Статистика
        """
        return {
            "sample_count": self._profile.get("sample_count", 0),
            "calibrated": self._profile.get("calibrated", False),
            "last_updated": self._profile.get("last_updated"),
            "history_length": len(self._profile.get("history", [])),
            "language": self._profile.get("language", "ru"),
            "name": self._profile.get("name"),
            "preferences": self._profile.get("preferences", {}),
        }
    
    def is_calibrated(self) -> bool:
        """Проверка калибровки профиля."""
        return self._profile.get("calibrated", False)
    
    def get_emotion_history(self) -> List[Dict[str, str]]:
        """Получение истории эмоций."""
        return [
            {"text": item.get("text", ""), "emotion": item.get("emotion", "neutral")}
            for item in self._profile.get("history", [])[-20:]
        ]
    
    def clear_history(self) -> None:
        """Очистка истории."""
        with self._lock:
            self._profile["history"] = []
            self.save()
        logger.info("История голосовых взаимодействий очищена")
    
    def reset(self) -> None:
        """Сброс профиля к настройкам по умолчанию."""
        with self._lock:
            self._profile = {
                "name": None,
                "voice_encoding": None,
                "language": "ru",
                "speaking_rate": 1.0,
                "pitch_shift": 0.0,
                "sample_count": 0,
                "last_updated": None,
                "calibrated": False,
                "history": [],
                "preferences": {
                    "tts_speaker": "xenia",
                    "emotions_enabled": True,
                    "noise_reduction": True,
                }
            }
            self.save()
        logger.info("Профиль голоса сброшен")
    
    @property
    def calibrated(self) -> bool:
        """Проверка калибровки."""
        return self.is_calibrated()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Получение настройки пользователя."""
        return self._profile.get("preferences", {}).get(key, default)
    
    def set_preference(self, key: str, value: Any) -> None:
        """Установка настройки пользователя."""
        with self._lock:
            if "preferences" not in self._profile:
                self._profile["preferences"] = {}
            self._profile["preferences"][key] = value
            self.save()