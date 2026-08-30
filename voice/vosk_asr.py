"""Потоковое распознавание речи через Vosk."""

import json
import threading
import queue
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import time

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


@dataclass
class RecognitionResult:
    """Результат распознавания."""
    text: str
    is_final: bool
    confidence: float
    timestamp: float
    duration: float


class VoskASR:
    """Потоковое распознавание речи через Vosk."""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        sample_rate: int = 16000,
        language: str = "ru",
        partial_results: bool = True,
        max_alternatives: int = 1
    ):
        """
        Args:
            model_path: Путь к модели Vosk
            sample_rate: Частота дискретизации (16000)
            language: Язык (ru, en, etc)
            partial_results: Показывать промежуточные результаты
            max_alternatives: Максимум альтернативных вариантов
        """
        if not VOSK_AVAILABLE:
            raise ImportError(
                "Установите Vosk: pip install vosk\n"
                "Скачайте модель: python download_vosk_model.py"
            )
        
        self.sample_rate = sample_rate
        self.language = language
        self.partial_results = partial_results
        self.max_alternatives = max_alternatives
        
        # Путь к модели
        if model_path is None:
            model_path = Path("models/vosk/vosk-model-ru-0.22")
        self.model_path = Path(model_path)
        
        # Состояние
        self.is_listening = False
        self.recognizer = None
        self.model = None
        
        # Результаты
        self.partial_text = ""
        self.final_text = ""
        self.last_result = None
        
        # Очередь для аудио
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Callbacks
        self.on_final = None
        self.on_partial = None
        self.on_error = None
        
        # Статистика
        self.stats = {
            "total_frames": 0,
            "speech_frames": 0,
            "recognized_count": 0,
            "errors": 0
        }
        
        # Загрузка модели
        self._load_model()
    
    def _load_model(self):
        """Загружает модель Vosk."""
        if not self.model_path.exists():
            error_msg = (
                f"❌ Модель не найдена: {self.model_path}\n"
                "📥 Скачайте модель командой:\n"
                "   python download_vosk_model.py\n"
                "Или вручную с https://alphacephei.com/vosk/models"
            )
            raise FileNotFoundError(error_msg)
        
        print(f"📂 Загрузка Vosk модели из {self.model_path}...")
        
        try:
            self.model = Model(str(self.model_path))
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            self.recognizer.SetPartialWords(True)
            
            # Настройка максимального количества альтернатив
            if self.max_alternatives > 1:
                self.recognizer.SetMaxAlternatives(self.max_alternatives)
            
            print(f"✅ Vosk ASR готов (язык: {self.language})")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback для sounddevice с игнорированием overflow."""
        if status:
            # Игнорируем input overflow - это нормально при работе с TTS
            if "input overflow" in str(status):
                return
            if self.on_error:
                self.on_error(f"Статус записи: {status}")
            return
        
        if self.is_listening and self.recognizer:
            try:
                audio = indata.flatten().astype(np.float32)
                # Конвертируем в 16-bit PCM для Vosk
                audio_int16 = (audio * 32767).astype(np.int16)
                self.audio_queue.put_nowait(audio_int16.tobytes())
                self.stats["total_frames"] += 1
            except queue.Full:
                # Если очередь переполнена, очищаем старые данные
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_int16.tobytes())
                except:
                    pass
            except Exception as e:
                if self.on_error:
                    self.on_error(f"Ошибка в callback: {e}")
    
    def _process_audio(self):
        """Поток обработки аудио."""
        print("🔄 Поток Vosk запущен")
        
        speech_detected = False
        silence_frames = 0
        silence_threshold = 30
        
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
                
                # Распознаём
                if self.recognizer.AcceptWaveform(accumulated):
                    result_data = json.loads(self.recognizer.Result())
                    text = result_data.get("text", "").strip()
                    
                    if text:
                        self.final_text = text
                        self.stats["recognized_count"] += 1
                        print(f"✅ Vosk: {text}")
                        # Проверяем, что on_final - это функция
                        if self.on_final is not None and callable(self.on_final):
                            self.on_final(text)
                    
                    speech_detected = False
                    silence_frames = 0
                    
                else:
                    if self.partial_results:
                        partial_data = json.loads(self.recognizer.PartialResult())
                        text = partial_data.get("partial", "").strip()
                        
                        if text and text != self.partial_text:
                            self.partial_text = text
                            # Проверяем, что on_partial - это функция
                            if self.on_partial is not None and callable(self.on_partial):
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
                    if self.on_final is not None and callable(self.on_final) and self.partial_text:
                        if len(self.partial_text) > 2:
                            print(f"✅ Vosk (таймаут): {self.partial_text}")
                            self.on_final(self.partial_text)
                    self.partial_text = ""
                    speech_detected = False
                    silence_frames = 0
                    
            except queue.Empty:
                continue
            except json.JSONDecodeError as e:
                if self.on_error is not None and callable(self.on_error):
                    self.on_error(f"Ошибка парсинга JSON: {e}")
                self.stats["errors"] += 1
            except Exception as e:
                if self.on_error is not None and callable(self.on_error):
                    self.on_error(f"Ошибка Vosk: {e}")
                self.stats["errors"] += 1
    
    def start_listening(
        self,
        on_final: Optional[Callable[[str], None]] = None,
        on_partial: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None
    ):
        """Запускает прослушивание."""
        if self.is_listening:
            print("⚠️ Уже слушаю")
            return False
        
        self.on_final = on_final
        self.on_partial = on_partial
        self.on_error = on_error
        self.is_listening = True
        
        # Очищаем очередь
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        # Создаём новый recognizer для чистой сессии
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)
        if self.max_alternatives > 1:
            self.recognizer.SetMaxAlternatives(self.max_alternatives)
        self.partial_text = ""
        self.final_text = ""
        
        # Запускаем поток обработки
        self.process_thread = threading.Thread(
            target=self._process_audio,
            daemon=True,
            name="VoskProcessor"
        )
        self.process_thread.start()
        
        # Запускаем захват аудио
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.03),  # 30ms
                latency='low'
            )
            self.stream.start()
            print("🎤 Vosk ASR запущен")
            return True
            
        except Exception as e:
            error_msg = f"Ошибка запуска: {e}"
            print(f"❌ {error_msg}")
            if self.on_error:
                self.on_error(error_msg)
            self.is_listening = False
            return False
    
    def stop(self):
        """Останавливает ASR."""
        if not self.is_listening:
            return
        
        print("⏹️ Остановка Vosk ASR...")
        self.is_listening = False
        
        # Останавливаем поток захвата
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        # Ждём завершения потока обработки
        if hasattr(self, 'process_thread') and self.process_thread:
            try:
                self.process_thread.join(timeout=1.0)
            except:
                pass
        
        # Очищаем очередь
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        print("✅ Vosk ASR остановлен")
    
    def get_final_result(self) -> Optional[str]:
        """Возвращает последний финальный результат."""
        return self.final_text if self.final_text else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику."""
        stats = self.stats.copy()
        stats["is_listening"] = self.is_listening
        stats["queue_size"] = self.audio_queue.qsize()
        return stats
    
    def reset(self):
        """Сбрасывает recognizer для новой сессии."""
        if self.recognizer:
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            if self.max_alternatives > 1:
                self.recognizer.SetMaxAlternatives(self.max_alternatives)
        self.partial_text = ""
        self.final_text = ""
        self.last_result = None
        self.stats = {
            "total_frames": 0,
            "speech_frames": 0,
            "recognized_count": 0,
            "errors": 0
        }
    
    @staticmethod
    def is_available() -> bool:
        """Проверяет доступность Vosk."""
        return VOSK_AVAILABLE

    def reset(self):
        """Сбрасывает recognizer для новой сессии."""
        if self.model:
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)
            if self.max_alternatives > 1:
                self.recognizer.SetMaxAlternatives(self.max_alternatives)
        self.partial_text = ""
        self.final_text = ""
        self.last_result = None
        self.stats = {
            "total_frames": 0,
            "speech_frames": 0,
            "recognized_count": 0,
            "errors": 0
        }