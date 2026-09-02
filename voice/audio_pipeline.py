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
    max_buffer_size: int = 50  # Максимальный размер far-буфера
    reset_on_overflow: bool = True  # Сбрасывать ли AEC при переполнении


class AudioPipeline:
    """
    Управляет аудиопотоком.
    
    Архитектура AEC (Acoustic Echo Cancellation):
    1. Захват с микрофона (near_end) - голос пользователя + эхо от динамиков
    2. Эталонный сигнал (far_end) - то, что воспроизводит бот
    3. AEC вычитает far_end из near_end, оставляя только голос пользователя
    4. Очищенный сигнал передается в ASR
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_audio: Optional[Callable[[np.ndarray], None]] = None
    ):
        self.config = config or PipelineConfig()
        self.on_audio = on_audio
        
        # AEC процессор - будет создан в потоке захвата
        self._aec: Optional[AECProcessor] = None
        self._aec_initialized = False
        
        # Буфер для far_end сигнала (то, что воспроизводится)
        self._far_buffer: queue.Queue = queue.Queue(maxsize=self.config.max_buffer_size)
        
        # Состояние
        self.is_running = False
        self._stream = None
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        
        # Статистика
        self.stats = {
            "frames_processed": 0,
            "aec_active": False,
            "noise_gate_active": False,
            "dropped_frames": 0,
            "far_buffer_size": 0,
            "buffer_overflows": 0,
        }
        
        # Таймаут для остановки
        self._stop_timeout = 2.0
        
        print("✅ AudioPipeline создан")
    
    def feed_far_end(self, audio: np.ndarray) -> bool:
        """
        Подает far_end сигнал (то, что воспроизводится) для AEC.
        
        Это критически важно для работы AEC:
        - Бот говорит -> audio отправляется в динамики
        - Этот же audio нужно передать в AEC как эталон
        - AEC вычтет его из сигнала микрофона
        
        Args:
            audio: Аудио для воспроизведения (float32)
            
        Returns:
            bool: Успешность добавления в буфер
        """
        try:
            self._far_buffer.put_nowait(audio.copy())
            return True
        except queue.Full:
            self.stats["buffer_overflows"] += 1
            
            if self.config.reset_on_overflow:
                # Сбрасываем старые данные
                try:
                    self._far_buffer.get_nowait()
                    self._far_buffer.put_nowait(audio.copy())
                    return True
                except:
                    pass
            
            # Если не удалось, сбрасываем AEC
            self.reset_aec()
            return False
    
    def _get_far_end(self, length: int) -> Optional[np.ndarray]:
        """
        Извлекает far_end сигнал из буфера.
        
        Возвращает ровно `length` сэмплов или None, если недостаточно данных.
        """
        if not self._aec_initialized:
            return None
        
        far_audio = []
        total_samples = 0
        
        # Собираем аудио из буфера
        while total_samples < length:
            try:
                chunk = self._far_buffer.get_nowait()
                far_audio.append(chunk)
                total_samples += len(chunk)
            except queue.Empty:
                break
        
        if not far_audio:
            return None
        
        result = np.concatenate(far_audio)
        
        # Если недостаточно, дополняем нулями
        if len(result) < length:
            padded = np.zeros(length, dtype=np.float32)
            padded[:len(result)] = result
            return padded
        
        # Если больше, сохраняем остаток
        if len(result) > length:
            try:
                self._far_buffer.put_nowait(result[length:])
            except queue.Full:
                pass
            return result[:length]
        
        return result
    
    def _noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Простое шумоподавление (noise gate)."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < self.config.noise_gate_threshold:
            self.stats["noise_gate_active"] = True
            return np.zeros_like(audio)
        
        self.stats["noise_gate_active"] = False
        return audio
    
    def _ensure_aec(self):
        """Создаёт AEC в текущем потоке (потоке захвата)."""
        if self._aec_initialized:
            return
        
        try:
            # ✅ AEC создаётся ЗДЕСЬ, в потоке захвата
            self._aec = AECProcessor(
                sample_rate=self.config.sample_rate,
                stream_delay_ms=self.config.stream_delay_ms,
            )
            # Принудительно инициализируем в этом потоке
            self._aec._ensure_initialized()
            self._aec_initialized = True
            self.stats["aec_active"] = True
            print(f"🧵 AEC инициализирован в потоке захвата {threading.get_ident()}")
        except Exception as e:
            print(f"⚠️ Ошибка инициализации AEC: {e}")
            self._aec = None
            self._aec_initialized = True
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Основной callback sounddevice.
        
        Принцип работы AEC:
        1. indata - сигнал с микрофона (пользователь + эхо от динамиков)
        2. far_end - эталонный сигнал (то, что воспроизводит бот)
        3. AEC вычитает far_end из indata, получая чистый голос пользователя
        4. Чистый сигнал отправляется в ASR
        """
        if status:
            if "input overflow" in str(status):
                self.stats["dropped_frames"] += 1
                return
            print(f"⚠️ Статус записи: {status}")
            return
        
        if not self.is_running or self._shutdown_event.is_set():
            return
        
        try:
            # ✅ Инициализируем AEC в этом потоке
            self._ensure_aec()
            
            # 1. Берем сигнал с микрофона (near_end)
            mic_signal = indata.flatten().astype(np.float32)
            
            # 2. Получаем far_end сигнал (то, что воспроизводит бот)
            far_signal = self._get_far_end(len(mic_signal))
            
            # 3. AEC обработка: вычитаем far из near
            if self._aec_initialized and self._aec is not None:
                if far_signal is not None:
                    # Передаем оба сигнала в AEC
                    clean_signal = self._aec.process(mic_signal, far_signal)
                else:
                    # Нет far_end сигнала -> нечего вычитать
                    clean_signal = mic_signal
            else:
                clean_signal = mic_signal
            
            # 4. Шумоподавление
            clean_signal = self._noise_gate(clean_signal)
            
            # 5. Отправляем очищенный сигнал в ASR
            if self.on_audio and np.abs(clean_signal).max() > 0.001:
                self.on_audio(clean_signal)
            
            self.stats["frames_processed"] += 1
            self.stats["far_buffer_size"] = self._far_buffer.qsize()
            
        except Exception as e:
            print(f"⚠️ Ошибка в аудио callback: {e}")
            self.stats["dropped_frames"] += 1
    
    def start(self) -> bool:
        """Запускает захват аудио."""
        if self.is_running:
            print("⚠️ AudioPipeline уже запущен")
            return False
        
        try:
            # Сбрасываем состояние
            self._shutdown_event.clear()
            self._aec = None
            self._aec_initialized = False
            
            # Очищаем буфер far_end
            while not self._far_buffer.empty():
                try:
                    self._far_buffer.get_nowait()
                except:
                    break
            
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
            print("✅ AudioPipeline запущен (AEC будет активирован в аудио потоке)")
            print("   Для работы AEC нужно подавать far_end сигнал через feed_far_end()")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка запуска AudioPipeline: {e}")
            return False
    
    def stop(self) -> None:
        """Останавливает захват аудио."""
        print("⏹️ Остановка AudioPipeline...")
        
        # Сигнал остановки
        self._shutdown_event.set()
        self.is_running = False
        
        # Останавливаем стрим
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                print(f"⚠️ Ошибка закрытия стрима: {e}")
            self._stream = None
        
        # Очищаем буфер
        while not self._far_buffer.empty():
            try:
                self._far_buffer.get_nowait()
            except:
                break
        
        # Сбрасываем AEC
        self.reset_aec()
        
        print("⏹️ AudioPipeline остановлен")
    
    def reset_aec(self) -> None:
        """Сбрасывает AEC процессор."""
        if self._aec is not None:
            try:
                self._aec.reset()
                self._aec_initialized = False
                print("🔄 AEC сброшен")
            except Exception as e:
                print(f"⚠️ Ошибка сброса AEC: {e}")
        else:
            self._aec = None
            self._aec_initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику."""
        stats = self.stats.copy()
        stats["is_running"] = self.is_running
        stats["aec_initialized"] = self._aec_initialized
        
        if self._aec is not None:
            try:
                aec_stats = self._aec.get_stats()
                stats["aec_processed"] = aec_stats.get("processed_frames", 0)
                stats["aec_initialized"] = aec_stats.get("initialized", False)
            except:
                pass
        
        return stats
    
    def clear_far_buffer(self) -> None:
        """Очищает буфер far_end."""
        while not self._far_buffer.empty():
            try:
                self._far_buffer.get_nowait()
            except:
                break
        print("🧹 Far-буфер очищен")
    
    @staticmethod
    def is_aec_available() -> bool:
        return AECProcessor.is_available()