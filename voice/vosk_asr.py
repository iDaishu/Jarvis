# voice/vosk_asr.py
"""Потоковое распознавание речи через Vosk."""

import json
import threading
import queue
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import time

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


class VoskASR:
    """Упрощенный ASR — только распознавание."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        sample_rate: int = 16000,
        partial_results: bool = True,
        max_alternatives: int = 1
    ):
        if not VOSK_AVAILABLE:
            raise ImportError("Установите Vosk: pip install vosk")
        
        self.sample_rate = sample_rate
        self.partial_results = partial_results
        self.max_alternatives = max_alternatives
        
        if model_path is None:
            model_path = Path("models/vosk/vosk-model-ru-0.22")
        self.model_path = Path(model_path)
        
        # Состояние
        self.is_listening = False
        self.model = None
        self.recognizer = None
        
        # Очередь для аудио
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Callbacks
        self.on_final = None
        self.on_partial = None
        self.on_error = None
        
        # Результаты
        self.partial_text = ""
        self.final_text = ""
        self._last_final_text = ""
        self._last_final_time = 0
        
        # Статистика
        self.stats = {
            "recognized_count": 0,
            "errors": 0,
            "silence_frames": 0,
            "speech_frames": 0
        }
        
        # Загрузка модели
        self._load_model()
    
    def _load_model(self):
        """Загружает модель Vosk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        
        print(f"📂 Загрузка Vosk из {self.model_path}...")
        try:
            self.model = Model(str(self.model_path))
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            self.recognizer.SetPartialWords(True)
            if self.max_alternatives > 1:
                self.recognizer.SetMaxAlternatives(self.max_alternatives)
            print("✅ Vosk ASR готов")
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            raise
    
    def feed_audio(self, audio_bytes: bytes):
        """
        Подает аудио в ASR.
        
        Args:
            audio_bytes: Аудио в формате 16-bit PCM
        """
        if not self.is_listening or not self.recognizer:
            return
        
        try:
            self.audio_queue.put_nowait(audio_bytes)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_bytes)
            except:
                pass
    
    def _process_queue(self):
        """Обрабатывает очередь аудио."""
        print("🔄 Vosk процессор запущен")
        
        silence_frames = 0
        silence_threshold = 40
        speech_detected = False
        
        while self.is_listening:
            try:
                data = self.audio_queue.get(timeout=0.1)
                if not data:
                    continue
                
                # Объединяем данные
                accumulated = data
                while not self.audio_queue.empty():
                    try:
                        accumulated += self.audio_queue.get_nowait()
                    except queue.Empty:
                        break
                
                if self.recognizer.AcceptWaveform(accumulated):
                    result_data = json.loads(self.recognizer.Result())
                    text = result_data.get("text", "").strip()
                    
                    if text:
                        current_time = time.time()
                        if text != self._last_final_text or current_time - self._last_final_time > 3.0:
                            self._last_final_text = text
                            self._last_final_time = current_time
                            self.final_text = text
                            self.stats["recognized_count"] += 1
                            if self.on_final:
                                self.on_final(text)
                    
                    speech_detected = False
                    silence_frames = 0
                    
                else:
                    if self.partial_results:
                        partial_data = json.loads(self.recognizer.PartialResult())
                        text = partial_data.get("partial", "").strip()
                        
                        if text and text != self.partial_text:
                            self.partial_text = text
                            if self.on_partial:
                                self.on_partial(text)
                            
                            if len(text) > 1:
                                speech_detected = True
                                silence_frames = 0
                                self.stats["speech_frames"] += 1
                            else:
                                silence_frames += 1
                        else:
                            silence_frames += 1
                    else:
                        silence_frames += 1
                
                # Таймаут
                if silence_frames > silence_threshold and speech_detected:
                    if self.on_final and self.partial_text and len(self.partial_text) > 2:
                        current_time = time.time()
                        if self.partial_text != self._last_final_text or current_time - self._last_final_time > 3.0:
                            if self.on_final:
                                self.on_final(self.partial_text)
                            self._last_final_text = self.partial_text
                            self._last_final_time = current_time
                    self.partial_text = ""
                    speech_detected = False
                    silence_frames = 0
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.on_error:
                    self.on_error(str(e))
                self.stats["errors"] += 1
    
    def start_listening(self, on_final, on_partial=None, on_error=None):
        """Запускает прослушивание."""
        if self.is_listening:
            return False
        
        self.on_final = on_final
        self.on_partial = on_partial
        self.on_error = on_error
        self.is_listening = True
        
        # Создаем новый recognizer
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)
        if self.max_alternatives > 1:
            self.recognizer.SetMaxAlternatives(self.max_alternatives)
        self.partial_text = ""
        self.final_text = ""
        self._last_final_text = ""
        self._last_final_time = 0
        
        # Запускаем поток обработки
        self._process_thread = threading.Thread(
            target=self._process_queue,
            daemon=True
        )
        self._process_thread.start()
        
        print("🎤 Vosk ASR запущен")
        return True
    
    def stop_listening(self):
        """Останавливает прослушивание."""
        self.is_listening = False
        
        if hasattr(self, '_process_thread') and self._process_thread:
            try:
                self._process_thread.join(timeout=1.0)
            except:
                pass
        
        # Очищаем очередь
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        print("⏹️ Vosk ASR остановлен")
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats["is_listening"] = self.is_listening
        stats["queue_size"] = self.audio_queue.qsize()
        return stats
    
    def reset(self):
        """Сбрасывает состояние."""
        self.partial_text = ""
        self.final_text = ""
        self._last_final_text = ""
        self._last_final_time = 0
        self.stats = {
            "recognized_count": 0,
            "errors": 0,
            "silence_frames": 0,
            "speech_frames": 0
        }
        if self.model:
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
    
    @staticmethod
    def is_available() -> bool:
        return VOSK_AVAILABLE