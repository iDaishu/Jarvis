"""Silero TTS для синтеза речи."""

import torch
import sounddevice as sd
import numpy as np
import threading
import queue
import time
from pathlib import Path
from typing import Optional, List

class SileroTTS:
    """Синтез речи через Silero TTS."""
    
    def __init__(
        self,
        model_name: str = "v5_4_ru",
        speaker: str = "xenia",
        language: str = "ru",
        device: str = "cpu",
        sample_rate: int = 48000
    ):
        self.model_name = model_name
        self.speaker = speaker
        self.language = language
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.model = None
        
        # Пути
        self.base_dir = Path(__file__).resolve().parent.parent
        self.models_dir = self.base_dir / "models" / "silero"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Асинхронное озвучивание
        self.queue = queue.Queue()
        self.is_speaking = False
        self.running = True
        
        # Загрузка модели
        self._load_model()
        self._start_worker()
    
    def _load_model(self):
        """Загружает модель."""
        model_path = self.models_dir / f"{self.model_name}.pt"
        
        if not model_path.exists():
            print(f"📥 Скачивание модели {self.model_name}...")
            torch.hub.download_url_to_file(
                f'https://models.silero.ai/models/tts/ru/{self.model_name}.pt',
                str(model_path)
            )
            print("✅ Модель скачана")
        
        print(f"📂 Загрузка модели {model_path.name}...")
        try:
            self.model = torch.package.PackageImporter(str(model_path)).load_pickle("tts_models", "model")
            print(f"✅ Silero TTS готов (голос: {self.speaker})")
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            raise
    
    def synthesize(self, text: str, speaker: Optional[str] = None) -> np.ndarray:
        """Синтезирует речь и возвращает numpy массив."""
        if not text or not text.strip() or self.model is None:
            return np.array([])
        
        speaker = speaker or self.speaker
        
        try:
            audio_np = self.model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=self.sample_rate
            )
            
            if audio_np is None or len(audio_np) == 0:
                return np.array([])
            
            # Нормализация
            if np.abs(audio_np).max() > 0:
                audio_np = audio_np / np.abs(audio_np).max() * 0.95
            
            return audio_np
            
        except Exception as e:
            print(f"⚠️ Ошибка синтеза: {e}")
            return np.array([])
    
    def speak(self, text: str, async_mode: bool = True, speaker: Optional[str] = None) -> bool:
        """Озвучивает текст."""
        if not text or not text.strip() or self.model is None:
            return False
        
        if async_mode:
            self.queue.put((text, speaker))
            return True
        else:
            return self._speak_sync(text, speaker)
    
    def _speak_sync(self, text: str, speaker: Optional[str] = None) -> bool:
        """Синхронное озвучивание."""
        try:
            audio = self.synthesize(text, speaker)
            if len(audio) == 0:
                return False
            
            self.is_speaking = True
            sd.play(audio, self.sample_rate)
            sd.wait()
            self.is_speaking = False
            return True
            
        except Exception as e:
            print(f"⚠️ Ошибка воспроизведения: {e}")
            self.is_speaking = False
            return False
    
    def _worker(self):
        """Поток для асинхронного озвучивания."""
        while self.running:
            try:
                text, speaker = self.queue.get(timeout=0.1)
                if text:
                    self._speak_sync(text, speaker)
                    self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Ошибка в TTS: {e}")
    
    def _start_worker(self):
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
    
    def get_speakers(self) -> List[str]:
        if self.model and hasattr(self.model, 'speakers'):
            return self.model.speakers
        return ['xenia', 'baya', 'natasha', 'ruslan', 'irina', 'kseniya']
    
    def stop(self):
        """Останавливает TTS."""
        self.running = False
        if hasattr(self, 'thread') and self.thread:
            self.thread.join(timeout=1)
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                break