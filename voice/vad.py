"""Улучшенное обнаружение голосовой активности с WebRTC VAD."""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import threading

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


@dataclass
class VADConfig:
    """Конфигурация VAD."""
    sample_rate: int = 16000
    mode: int = 3  # 0-3, где 3 - самый агрессивный
    frame_duration_ms: int = 30  # 10, 20 или 30 мс
    speech_threshold: float = 0.5  # Доля голосовых фреймов для определения речи
    min_speech_frames: int = 3  # Минимальное число голосовых фреймов


class VAD:
    """Голосовая активность с WebRTC VAD."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        mode: int = 3,
        frame_duration_ms: int = 30,
        speech_threshold: float = 0.5,
        min_speech_frames: int = 3
    ):
        """
        Args:
            sample_rate: Частота дискретизации (8000, 16000, 32000, 48000)
            mode: Режим VAD (0-3)
            frame_duration_ms: Длительность фрейма (10, 20 или 30 мс)
            speech_threshold: Доля голосовых фреймов для определения речи (0-1)
            min_speech_frames: Минимальное число голосовых фреймов
        """
        self.config = VADConfig(
            sample_rate=sample_rate,
            mode=mode,
            frame_duration_ms=frame_duration_ms,
            speech_threshold=speech_threshold,
            min_speech_frames=min_speech_frames
        )
        
        self.vad = None
        self._lock = threading.Lock()
        
        if WEBRTC_AVAILABLE:
            self.vad = webrtcvad.Vad(mode)
            print(f"✅ WebRTC VAD инициализирован (mode={mode})")
        else:
            print("⚠️ webrtcvad не установлен, используется простой VAD")
            print("   Установите: pip install webrtcvad")
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Проверяет, содержит ли аудио речь.
        
        Args:
            audio: Аудио массив float32 [-1, 1]
            
        Returns:
            bool: True если есть речь
        """
        if len(audio) == 0:
            return False
        
        with self._lock:
            if self.vad is not None and WEBRTC_AVAILABLE:
                return self._webrtc_vad(audio)
            else:
                return self._simple_vad(audio)
    
    def _webrtc_vad(self, audio: np.ndarray) -> bool:
        """WebRTC VAD."""
        try:
            # Конвертируем в 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Вычисляем размер фрейма
            frame_size = int(self.config.sample_rate * self.config.frame_duration_ms / 1000)
            
            if len(audio_int16) < frame_size:
                return False
            
            # Проверяем фреймы
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_int16) - frame_size + 1, frame_size):
                frame = audio_int16[i:i+frame_size]
                if len(frame) != frame_size:
                    continue
                try:
                    if self.vad.is_speech(frame.tobytes(), self.config.sample_rate):
                        speech_frames += 1
                    total_frames += 1
                except Exception:
                    continue
                
                # Если уже есть достаточно голосовых фреймов, можно остановиться
                if speech_frames >= self.config.min_speech_frames:
                    return True
            
            if total_frames == 0:
                return False
            
            # Речь, если доля голосовых фреймов выше порога
            return speech_frames / total_frames >= self.config.speech_threshold
            
        except Exception as e:
            print(f"⚠️ Ошибка WebRTC VAD: {e}")
            return self._simple_vad(audio)
    
    def _simple_vad(self, audio: np.ndarray) -> bool:
        """Простая проверка по энергии сигнала."""
        if len(audio) == 0:
            return False
        
        # Нормализуем
        max_amp = np.abs(audio).max()
        if max_amp < 1e-10:
            return False
        
        audio_norm = audio / max_amp
        
        # Энергия сигнала
        energy = np.sqrt(np.mean(audio_norm ** 2))
        
        # Проверяем, не слишком ли равномерный сигнал (шум)
        std_dev = np.std(audio_norm)
        if std_dev < 0.01:
            return False
        
        # Речь, если энергия выше порога и есть вариативность
        return energy > 0.01 and std_dev > 0.005
    
    def get_speech_segments(
        self,
        audio: np.ndarray,
        min_duration: float = 0.3,
        max_gap: float = 0.3
    ) -> List[Tuple[int, int]]:
        """
        Находит сегменты речи в аудио.
        
        Args:
            audio: Аудио массив
            min_duration: Минимальная длительность сегмента (сек)
            max_gap: Максимальный разрыв между сегментами (сек)
            
        Returns:
            List[Tuple[int, int]]: Список (начало, конец) в сэмплах
        """
        if len(audio) == 0:
            return []
        
        # Размер фрейма в сэмплах
        frame_size = int(self.config.sample_rate * self.config.frame_duration_ms / 1000)
        
        # Определяем речь для каждого фрейма
        is_speech = []
        for i in range(0, len(audio) - frame_size + 1, frame_size):
            frame = audio[i:i+frame_size]
            is_speech.append(self.is_speech(frame))
        
        # Находим сегменты
        segments = []
        start = None
        min_samples = int(min_duration * self.config.sample_rate)
        gap_samples = int(max_gap * self.config.sample_rate)
        gap_frames = gap_samples // frame_size
        
        for i, speech in enumerate(is_speech):
            if speech:
                if start is None:
                    start = i * frame_size
            else:
                if start is not None:
                    # Проверяем длину сегмента
                    end = i * frame_size
                    if end - start >= min_samples:
                        segments.append((start, end))
                    start = None
        
        # Добавляем последний сегмент
        if start is not None:
            end = len(audio)
            if end - start >= min_samples:
                segments.append((start, end))
        
        # Объединяем близкие сегменты
        if len(segments) > 1:
            merged = []
            current_start, current_end = segments[0]
            
            for start, end in segments[1:]:
                if start - current_end <= gap_samples:
                    current_end = end
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = start, end
            
            merged.append((current_start, current_end))
            segments = merged
        
        return segments
    
    def set_mode(self, mode: int):
        """Устанавливает режим VAD (0-3)."""
        if 0 <= mode <= 3:
            self.config.mode = mode
            if self.vad is not None and WEBRTC_AVAILABLE:
                self.vad.set_mode(mode)
            print(f"VAD mode установлен на {mode}")
        else:
            raise ValueError("Mode должен быть от 0 до 3")
    
    @staticmethod
    def is_available() -> bool:
        """Проверяет доступность WebRTC VAD."""
        return WEBRTC_AVAILABLE