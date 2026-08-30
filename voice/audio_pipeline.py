# voice/audio_pipeline.py
"""Управление аудиопотоком: захват, AEC, подача в ASR."""

import threading
import queue
import time
import numpy as np
import sounddevice as sd
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from .aec_processor import AECProcessor


@dataclass
class PipelineConfig:
    """Конфигурация аудиопайплайна."""
    sample_rate: int = 16000
    channels: int = 1
    blocksize: int = 480  # 30ms при 16kHz
    latency: str = 'low'
    stream_delay_ms: int = 100
    noise_gate_threshold: float = 0.005


class AudioPipeline:
    """
    Управляет аудиопотоком:
    1. Захват с микрофона
    2. AEC обработка (подавление эха)
    3. Шумоподавление (noise gate)
    4. Подача в ASR
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_audio: Optional[Callable[[np.ndarray], None]] = None
    ):
        self.config = config or PipelineConfig()
        self.on_audio = on_audio
        
        # AEC процессор
        self.aec = AECProcessor(
            sample_rate=self.config.sample_rate,
            stream_delay_ms=self.config.stream_delay_ms
        )
        
        # Состояние
        self.is_running = False
        self._stream = None
        self._lock = threading.Lock()
        
        # Буфер для эталонного сигнала (far-end)
        self._far_buffer: queue.Queue = queue.Queue(maxsize=100)
        
        # Статистика
        self.stats = {
            "frames_processed": 0,
            "aec_active": self.aec.is_available(),
            "noise_gate_active": False,
            "dropped_frames": 0
        }
    
    def set_far_end_audio(self, audio: np.ndarray):
        """
        Подает эталонный сигнал (то, что воспроизводится) для AEC.
        
        Args:
            audio: Аудио для воспроизведения (float32)
        """
        if self.aec.is_available():
            try:
                self._far_buffer.put_nowait(audio.copy())
            except queue.Full:
                # Если буфер переполнен, очищаем старые данные
                try:
                    self._far_buffer.get_nowait()
                    self._far_buffer.put_nowait(audio.copy())
                except:
                    pass
    
    def _get_far_audio(self, length: int) -> Optional[np.ndarray]:
        """Возвращает эталонный сигнал нужной длины."""
        if not self.aec.is_available():
            return None
        
        # Собираем аудио из буфера
        far_audio = []
        total_samples = 0
        
        while total_samples < length:
            try:
                chunk = self._far_buffer.get_nowait()
                far_audio.append(chunk)
                total_samples += len(chunk)
            except queue.Empty:
                break
        
        if not far_audio:
            return None
        
        # Объединяем
        result = np.concatenate(far_audio)
        
        # Если недостаточно, дополняем нулями
        if len(result) < length:
            padded = np.zeros(length, dtype=np.float32)
            padded[:len(result)] = result
            return padded
        
        # Если больше, сохраняем остаток
        if len(result) > length:
            # Возвращаем в буфер остаток
            try:
                self._far_buffer.put_nowait(result[length:])
            except:
                pass
            return result[:length]
        
        return result
    
    def _noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Шумоподавление (простой noise gate)."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < self.config.noise_gate_threshold:
            self.stats["noise_gate_active"] = True
            return np.zeros_like(audio)
        
        self.stats["noise_gate_active"] = False
        return audio
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback для sounddevice."""
        if status:
            if "input overflow" in str(status):
                self.stats["dropped_frames"] += 1
                return
            print(f"⚠️ Статус записи: {status}")
            return
        
        if not self.is_running:
            return
        
        try:
            # Получаем аудио
            audio = indata.flatten().astype(np.float32)
            
            # AEC обработка
            if self.aec.is_available():
                far_audio = self._get_far_audio(len(audio))
                if far_audio is not None:
                    audio = self.aec.process(audio, far_audio)
            
            # Шумоподавление
            audio = self._noise_gate(audio)
            
            # Вызываем callback
            if self.on_audio and np.abs(audio).max() > 0.001:
                self.on_audio(audio)
            
            self.stats["frames_processed"] += 1
            
        except Exception as e:
            print(f"⚠️ Ошибка в аудио callback: {e}")
    
    def start(self):
        """Запускает захват аудио."""
        if self.is_running:
            print("⚠️ AudioPipeline уже запущен")
            return False
        
        try:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=self.config.blocksize,
                latency=self.config.latency
            )
            self._stream.start()
            self.is_running = True
            print("✅ AudioPipeline запущен (AEC активен)")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка запуска AudioPipeline: {e}")
            return False
    
    def stop(self):
        """Останавливает захват аудио."""
        self.is_running = False
        
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except:
                pass
            self._stream = None
        
        # Очищаем буфер
        while not self._far_buffer.empty():
            try:
                self._far_buffer.get_nowait()
            except:
                break
        
        print("⏹️ AudioPipeline остановлен")
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику."""
        stats = self.stats.copy()
        stats["is_running"] = self.is_running
        stats["far_buffer_size"] = self._far_buffer.qsize()
        return stats
    
    def reset_aec(self):
        """Сбрасывает AEC процессор."""
        if self.aec.is_available():
            self.aec.reset()
    
    @staticmethod
    def is_aec_available() -> bool:
        """Проверяет доступность AEC."""
        return AECProcessor.is_available()