# voice/voice_profile.py
"""Профиль голоса пользователя."""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import threading

class VoiceProfile:
    """Профиль голоса пользователя."""
    
    def __init__(self, profile_path: Path):
        self.profile_path = profile_path
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile = self._load()
        self._lock = threading.Lock()
    
    def _load(self) -> Dict[str, Any]:
        """Загружает профиль."""
        if self.profile_path.exists():
            try:
                data = json.loads(self.profile_path.read_text(encoding='utf-8'))
                return data
            except:
                pass
        
        return {
            "name": None,
            "voice_encoding": None,
            "language": "ru",
            "speaking_rate": 1.0,
            "pitch_shift": 0.0,
            "sample_count": 0,
            "last_updated": None,
            "calibrated": False,
            "history": []
        }
    
    def save(self):
        """Сохраняет профиль."""
        with self._lock:
            self.profile["last_updated"] = datetime.now().isoformat()
            try:
                self.profile_path.write_text(
                    json.dumps(self.profile, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )
            except Exception as e:
                print(f"⚠️ Ошибка сохранения профиля: {e}")
    
    def get_name(self) -> Optional[str]:
        return self.profile.get("name")
    
    def set_name(self, name: str):
        self.profile["name"] = name.strip()
        self.save()
    
    def add_sample(self, text: str, audio: np.ndarray, emotion: str = "neutral") -> bool:
        try:
            with self._lock:
                self.profile["sample_count"] += 1
                self.profile["calibrated"] = True
                
                sample = {
                    "timestamp": datetime.now().isoformat(),
                    "text": text[:100],
                    "emotion": emotion,
                    "sample_number": self.profile["sample_count"]
                }
                
                self.profile["history"].append(sample)
                
                if len(self.profile["history"]) > 100:
                    self.profile["history"] = self.profile["history"][-100:]
                
                self.save()
                return True
                
        except Exception as e:
            print(f"⚠️ Ошибка добавления образца: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "sample_count": self.profile.get("sample_count", 0),
            "calibrated": self.profile.get("calibrated", False),
            "last_updated": self.profile.get("last_updated"),
            "history_length": len(self.profile.get("history", []))
        }
    
    def is_calibrated(self) -> bool:
        return self.profile.get("calibrated", False)
    
    def clear_history(self):
        self.profile["history"] = []
        self.save()
    
    def reset(self):
        self.profile = {
            "name": None,
            "voice_encoding": None,
            "language": "ru",
            "speaking_rate": 1.0,
            "pitch_shift": 0.0,
            "sample_count": 0,
            "last_updated": None,
            "calibrated": False,
            "history": []
        }
        self.save()