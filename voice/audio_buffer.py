"""Буферизация аудио с VAD и детекцией начала/конца речи."""

import numpy as np
import threading
import time
from typing import Optional, Callable, List
from collections import deque
from dataclasses import dataclass

from .vad import VAD


@dataclass
class BufferConfig:
    """Конфигурация буфера."""
    sample_rate: int = 16000
    min_speech_duration: float = 0.5  # Минимальная длительность речи (сек)
    max_speech_duration: float = 15.0  # Максимальная длительность речи (сек)
    silence_timeout: float = 0.8  # Таймаут тишины (сек)
    pre_speech_padding: float = 0.2  # Добавлять до начала речи (сек)
    post_speech_padding: float = 0.3  # Добавлять после речи (сек)
    min_audio_length: int = 16000  # Минимальная длина аудио для распознавания


class AudioBuffer:
    """Буфер аудио с детекцией начала и конца речи."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        vad: Optional[VAD] = None,
        min_speech_duration: float = 0.5,
        max_speech_duration: float = 15.0,
        silence_timeout: float = 0.8,
        pre_speech_padding: float = 0.2,
        post_speech_padding: float = 0.3
    ):
        """
        Args:
            sample_rate: Частота дискретизации
            vad: Экземпляр VAD
            min_speech_duration: Минимальная длительность речи
            max_speech_duration: Максимальная длительность речи
            silence_timeout: Таймаут тишины для завершения речи
            pre_speech_padding: Добавлять паузу перед речью
            post_speech_padding: Добавлять паузу после речи
        """
        self.config = BufferConfig(
            sample_rate=sample_rate,
            min_speech_duration=min_speech_duration,
            max_speech_duration=max_speech_duration,
            silence_timeout=silence_timeout,
            pre_speech_padding=pre_speech_padding,
            post_speech_padding=post_speech_padding
        )
        
        self.vad = vad or VAD(sample_rate=sample_rate)
        
        # Буферы
        self.ring_buffer = deque(maxlen=int(sample_rate * 5))  # 5 секунд
        self.speech_buffer: List[np.ndarray] = []
        self.pre_buffer: List[np.ndarray] = []
        
        # Состояние
        self.is_speaking = False
        self.speech_start_time = 0
        self.silence_start_time = 0
        self.has_speech = False
        
        # Параметры для padding
        self.pre_padding_samples = int(sample_rate * pre_speech_padding)
        self.post_padding_samples = int(sample_rate * post_speech_padding)
        
        self._lock = threading.Lock()
        self.on_speech_end: Optional[Callable[[np.ndarray], None]] = None
        self.on_speech_start: Optional[Callable[[], None]] = None
        
        # Статистика
        self.stats = {
            "speech_segments": 0,
            "total_audio_processed": 0,
            "total_speech_processed": 0
        }
    
    def add_audio(self, audio: np.ndarray) -> bool:
        """
        Добавляет аудио и проверяет, закончилась ли речь.
        
        Returns:
            bool: True если сегмент речи завершён
        """
        if len(audio) == 0:
            return False
        
        with self._lock:
            self.ring_buffer.extend(audio)
            self.stats["total_audio_processed"] += len(audio)
            
            # Определяем наличие речи
            is_speech = self.vad.is_speech(audio)
            
            if is_speech:
                self._handle_speech(audio)
            else:
                self._handle_silence(audio)
            
            return self.is_speaking and self._should_finish()
    
    def _handle_speech(self, audio: np.ndarray):
        """Обрабатывает голосовое аудио."""
        self.silence_start_time = 0
        
        if not self.is_speaking:
            # Начало речи
            self.is_speaking = True
            self.speech_start_time = time.time()
            self.speech_buffer = []
            self.has_speech = True
            
            # Добавляем pre-padding из кольцевого буфера
            ring_list = list(self.ring_buffer)
            if len(ring_list) > 0:
                pre_padding = ring_list[-self.pre_padding_samples:]
                if len(pre_padding) > 0:
                    self.speech_buffer.append(np.concatenate(pre_padding))
            
            if self.on_speech_start:
                self.on_speech_start()
            
            print("🎤 Начало речи")
        
        # Добавляем аудио в буфер
        self.speech_buffer.append(audio)
        self.stats["total_speech_processed"] += len(audio)
    
    def _handle_silence(self, audio: np.ndarray):
        """Обрабатывает тишину."""
        if self.is_speaking:
            if self.silence_start_time == 0:
                self.silence_start_time = time.time()
            
            # Добавляем тишину в буфер (для контекста)
            if len(self.speech_buffer) > 0:
                self.speech_buffer.append(audio)
    
    def _should_finish(self) -> bool:
        """Проверяет, нужно ли завершить сегмент речи."""
        if not self.is_speaking:
            return False
        
        # Проверяем максимальную длительность
        speech_duration = time.time() - self.speech_start_time
        if speech_duration > self.config.max_speech_duration:
            print(f"⏱️ Превышена максимальная длительность ({self.config.max_speech_duration}с)")
            self._finish_speech()
            return True
        
        # Проверяем таймаут тишины
        if self.silence_start_time > 0:
            silence_duration = time.time() - self.silence_start_time
            if silence_duration > self.config.silence_timeout:
                self._finish_speech()
                return True
        
        return False
    
    def _finish_speech(self):
        """Завершает сегмент речи."""
        if not self.speech_buffer:
            return
        
        try:
            # Объединяем аудио
            audio = np.concatenate(self.speech_buffer)
            
            # Добавляем post-padding (если есть в кольцевом буфере)
            # (не используем, так как ещё нет данных после)
            
            # Проверяем минимальную длительность
            duration = len(audio) / self.config.sample_rate
            if duration >= self.config.min_speech_duration:
                if self.on_speech_end:
                    self.on_speech_end(audio.copy())
                self.stats["speech_segments"] += 1
            else:
                print(f"🔇 Слишком короткая речь: {duration:.2f}с (мин: {self.config.min_speech_duration}с)")
                
        except Exception as e:
            print(f"⚠️ Ошибка завершения речи: {e}")
        finally:
            self.is_speaking = False
            self.speech_buffer = []
            self.silence_start_time = 0
            self.has_speech = False
    
    def get_audio(self) -> Optional[np.ndarray]:
        """Возвращает текущий буфер аудио."""
        if not self.speech_buffer:
            return None
        
        try:
            return np.concatenate(self.speech_buffer)
        except:
            return None
    
    def reset(self):
        """Сбрасывает буфер."""
        with self._lock:
            self.is_speaking = False
            self.speech_buffer = []
            self.silence_start_time = 0
            self.has_speech = False
            self.ring_buffer.clear()
            self.stats = {
                "speech_segments": 0,
                "total_audio_processed": 0,
                "total_speech_processed": 0
            }
    
    def get_stats(self) -> dict:
        """Возвращает статистику."""
        stats = self.stats.copy()
        stats["is_speaking"] = self.is_speaking
        stats["buffer_size"] = len(self.speech_buffer)
        stats["ring_buffer_size"] = len(self.ring_buffer)
        return stats