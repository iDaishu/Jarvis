# voice/vosk_asr.py
"""
Потоковое распознавание речи через Vosk.

Обеспечивает непрерывное распознавание с выдачей промежуточных и финальных результатов.
"""

import json
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from utils.logger import LoggerFactory
from utils.exceptions import VoiceError

logger = LoggerFactory.get_logger("vosk_asr")

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logger.warning("Vosk не установлен. ASR будет недоступен.")


class VoskASR:
    """
    ASR на основе Vosk с потоковым распознаванием.
    
    Поддерживает частичные результаты и определение тишины для автоматической финализации.
    Использует двойную буферизацию для потокобезопасности.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        sample_rate: int = 16000,
        partial_results: bool = True,
        max_alternatives: int = 1,
        silence_timeout: float = 0.6,
        min_phrase_length: int = 2,
        max_queue_size: int = 100
    ):
        """
        Инициализация Vosk ASR.
        
        Args:
            model_path: Путь к модели Vosk
            sample_rate: Частота дискретизации
            partial_results: Выдавать ли частичные результаты
            max_alternatives: Максимальное количество альтернатив
            silence_timeout: Таймаут тишины в секундах
            min_phrase_length: Минимальная длина фразы для финализации
            max_queue_size: Максимальный размер очереди аудио
            
        Raises:
            VoiceError: Если Vosk не установлен или модель не найдена
        """
        if not VOSK_AVAILABLE:
            raise VoiceError("Vosk не установлен. Установите: pip install vosk")
        
        self.sample_rate = sample_rate
        self.partial_results = partial_results
        self.max_alternatives = max_alternatives
        self.silence_timeout = silence_timeout
        self.min_phrase_length = min_phrase_length
        self.max_queue_size = max_queue_size
        
        # Путь к модели
        if model_path is None:
            model_path = Path("models/vosk/vosk-model-ru-0.22")
        self.model_path = Path(model_path)
        
        # Состояние
        self._is_listening = False
        self._model = None
        self._recognizer = None
        
        # ✅ Двойная буферизация для потокобезопасности
        self._write_lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._audio_buffer = bytearray()
        self._buffer_chunk_size = int(sample_rate * 0.1)  # 100ms чанки
        
        # Callbacks
        self._on_final = None
        self._on_partial = None
        self._on_error = None
        
        # Результаты
        self._partial_text = ""
        self._final_text = ""
        self._last_final_text = ""
        self._last_final_time = 0.0
        self._speech_detected = False
        self._silence_frames = 0
        
        # Поток обработки
        self._process_thread: Optional[threading.Thread] = None
        self._running = False
        self._processing_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        
        # Статистика
        self._stats = {
            "recognized_count": 0,
            "errors": 0,
            "partial_updates": 0,
            "silence_resets": 0,
            "audio_processed": 0,
            "buffer_overflows": 0,
        }
        
        # Загрузка модели
        self._load_model()
    
    def _load_model(self) -> None:
        """Загрузка модели Vosk."""
        if not self.model_path.exists():
            raise VoiceError(f"Модель не найдена: {self.model_path}")
        
        try:
            logger.info(f"Загрузка Vosk модели из {self.model_path}...")
            self._model = Model(str(self.model_path))
            self._create_recognizer()
            logger.info("Vosk ASR готов")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели Vosk: {e}")
            raise VoiceError(f"Не удалось загрузить модель Vosk: {e}")
    
    def _create_recognizer(self) -> None:
        """Создание нового распознавателя."""
        if self._model is None:
            raise VoiceError("Модель не загружена")
        
        self._recognizer = KaldiRecognizer(self._model, self.sample_rate)
        self._recognizer.SetWords(True)
        if self.max_alternatives > 1:
            self._recognizer.SetMaxAlternatives(self.max_alternatives)
        
        # Сброс состояния
        self._partial_text = ""
        self._final_text = ""
        self._speech_detected = False
        self._silence_frames = 0
        with self._write_lock:
            self._audio_buffer = bytearray()
    
    def feed_audio(self, audio_bytes: bytes) -> None:
        """
        Подача аудио в ASR с буферизацией.
        
        Использует двойную буферизацию:
        1. Запись в буфер с write_lock
        2. Обработка копии с read_lock
        
        Args:
            audio_bytes: Аудио в формате 16-bit PCM
        """
        if not self._is_listening or self._recognizer is None:
            return
        
        if not audio_bytes or len(audio_bytes) == 0:
            return
        
        # ✅ Запись в буфер с блокировкой
        with self._write_lock:
            # Защита от переполнения
            if len(self._audio_buffer) > self.max_queue_size * self._buffer_chunk_size:
                self._audio_buffer = bytearray()
                self._stats["buffer_overflows"] += 1
                logger.warning("Аудио-буфер переполнен, сброс")
            
            self._audio_buffer.extend(audio_bytes)
        
        # ✅ Обработка без блокировки записи
        self._process_pending()
    
    def _process_pending(self) -> None:
        """Обработка накопленных данных с копированием буфера."""
        # ✅ Блокируем только для чтения/копирования
        with self._read_lock:
            if len(self._audio_buffer) < self._buffer_chunk_size:
                return
            
            # Копируем данные для обработки
            buffer = self._audio_buffer
            self._audio_buffer = bytearray()
        
        # ✅ Обрабатываем без блокировки
        while len(buffer) >= self._buffer_chunk_size:
            chunk = bytes(buffer[:self._buffer_chunk_size])
            buffer = buffer[self._buffer_chunk_size:]
            self._process_chunk(chunk)
        
        # ✅ Возвращаем остаток в буфер
        if buffer:
            with self._write_lock:
                self._audio_buffer.extend(buffer)
    
    def _process_chunk(self, chunk: bytes) -> None:
        """
        Обработка одного чанка аудио.
        
        Args:
            chunk: Аудио чанк
        """
        if self._recognizer is None:
            return
        
        try:
            self._stats["audio_processed"] += 1
            
            if self._recognizer.AcceptWaveform(chunk):
                result_data = json.loads(self._recognizer.Result())
                text = result_data.get("text", "").strip()
                
                if text and len(text) >= self.min_phrase_length:
                    current_time = time.time()
                    # Защита от дублирования
                    if text != self._last_final_text or current_time - self._last_final_time > 2.0:
                        self._last_final_text = text
                        self._last_final_time = current_time
                        self._final_text = text
                        self._stats["recognized_count"] += 1
                        
                        if self._on_final:
                            self._on_final(text)
                
                self._speech_detected = False
                self._silence_frames = 0
                
            else:
                if self.partial_results:
                    partial_data = json.loads(self._recognizer.PartialResult())
                    text = partial_data.get("partial", "").strip()
                    
                    if text:
                        if text != self._partial_text:
                            self._partial_text = text
                            self._stats["partial_updates"] += 1
                            if self._on_partial:
                                self._on_partial(text)
                        
                        self._speech_detected = True
                        self._silence_frames = 0
                    else:
                        self._silence_frames += 1
                else:
                    self._silence_frames += 1
            
            # Проверка таймаута тишины
            silence_threshold = int(self.silence_timeout * 10)  # 10 чанков в секунду
            if self._silence_frames > silence_threshold and self._speech_detected:
                if self._on_final and self._partial_text:
                    text = self._partial_text
                    if len(text) >= self.min_phrase_length:
                        current_time = time.time()
                        if text != self._last_final_text or current_time - self._last_final_time > 2.0:
                            if self._on_final:
                                self._on_final(text)
                            self._last_final_text = text
                            self._last_final_time = current_time
                
                self._partial_text = ""
                self._speech_detected = False
                self._silence_frames = 0
                
        except Exception as e:
            logger.error(f"Ошибка обработки чанка: {e}")
            self._stats["errors"] += 1
            if self._on_error:
                self._on_error(str(e))
    
    def start_listening(
        self,
        on_final: Callable[[str], None],
        on_partial: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Запуск прослушивания.
        
        Args:
            on_final: Callback для финальных результатов
            on_partial: Callback для частичных результатов
            on_error: Callback для ошибок
            
        Returns:
            bool: Успешность запуска
        """
        if self._is_listening:
            logger.warning("Vosk уже слушает")
            return False
        
        if self._model is None:
            logger.error("Модель не загружена")
            return False
        
        self._on_final = on_final
        self._on_partial = on_partial
        self._on_error = on_error
        self._is_listening = True
        self._shutdown_event.clear()
        
        # Создание нового распознавателя
        self._create_recognizer()
        
        logger.info("Vosk ASR запущен")
        return True
    
    def stop_listening(self) -> None:
        """Остановка прослушивания."""
        self._is_listening = False
        self._shutdown_event.set()
        
        # Сброс состояния
        with self._write_lock:
            self._partial_text = ""
            self._final_text = ""
            self._speech_detected = False
            self._silence_frames = 0
            self._audio_buffer = bytearray()
        
        logger.info("Vosk ASR остановлен")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики."""
        stats = self._stats.copy()
        stats["is_listening"] = self._is_listening
        with self._read_lock:
            stats["buffer_size"] = len(self._audio_buffer)
        stats["partial_text"] = self._partial_text[:50] + "..." if self._partial_text else ""
        return stats
    
    def reset(self) -> None:
        """Сброс состояния распознавателя."""
        with self._write_lock:
            self._partial_text = ""
            self._final_text = ""
            self._last_final_text = ""
            self._last_final_time = 0.0
            self._speech_detected = False
            self._silence_frames = 0
            self._audio_buffer = bytearray()
            self._stats = {
                "recognized_count": 0,
                "errors": 0,
                "partial_updates": 0,
                "silence_resets": 0,
                "audio_processed": 0,
                "buffer_overflows": 0,
            }
            
            if self._model:
                self._create_recognizer()
                logger.debug("Vosk ASR сброшен")
    
    @staticmethod
    def is_available() -> bool:
        """Проверка доступности Vosk."""
        return VOSK_AVAILABLE