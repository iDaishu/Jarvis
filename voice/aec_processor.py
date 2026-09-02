# voice/aec_processor.py
"""AEC (Acoustic Echo Cancellation) процессор с буферизацией."""

import numpy as np
import threading
from typing import Optional, Dict, Any

try:
    import aec3_py
    AEC_AVAILABLE = True
except ImportError:
    AEC_AVAILABLE = False
    print("⚠️ aec3_py не установлен. Установите: maturin develop --release")


class AECProcessor:
    """
    AEC процессор на основе aec3 с буферизацией для оптимизации.
    
    Важно: AECProcessor должен быть инициализирован в ТОМ ЖЕ ПОТОКЕ,
    где будет вызываться process(). Обычно это поток захвата audio.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        stream_delay_ms: int = 100,
        render_channels: int = 1,
        capture_channels: int = 1,
        frame_ms: int = 10,  # Длительность одного фрейма AEC (10 мс)
        buffer_ms: int = 30,  # Сколько мс накапливать перед обработкой
    ):
        self.sample_rate = sample_rate
        self.stream_delay_ms = stream_delay_ms
        self.render_channels = render_channels
        self.capture_channels = capture_channels
        self.frame_ms = frame_ms
        self.buffer_ms = buffer_ms
        
        # Размеры в сэмплах
        self.frame_samples = int(sample_rate * frame_ms / 1000)  # 160 при 16 кГц
        self.buffer_samples = int(sample_rate * buffer_ms / 1000)  # 480 при 16 кГц
        
        # Буферы для накопления
        self._near_buffer = np.array([], dtype=np.float32)
        self._far_buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        
        # AEC создаётся ТОЛЬКО в потоке, где вызывается process()
        self._aec = None
        self._initialized = False
        self._init_lock = threading.Lock()
        
        # Статистика
        self._processed_frames = 0
        self._dropped_frames = 0
        
        self._thread_id = None
        print(f"🧵 AECProcessor создан")
        print(f"   frame_samples: {self.frame_samples}, buffer_samples: {self.buffer_samples}")
    
    def _ensure_initialized(self):
        """
        Гарантирует, что AEC инициализирован в текущем потоке.
        Должна вызываться из потока, где будет использоваться AEC.
        """
        if self._initialized:
            return
        
        with self._init_lock:
            if self._initialized:
                return
            
            current_thread = threading.get_ident()
            self._thread_id = current_thread
            self._init_aec()
            self._initialized = True
            print(f"✅ AEC инициализирован в потоке {self._thread_id}")
    
    def _init_aec(self):
        """Создаёт экземпляр AEC в текущем потоке."""
        if not AEC_AVAILABLE:
            self._aec = None
            return
        
        try:
            self._aec = aec3_py.Aec3(
                sample_rate_hz=self.sample_rate,
                render_channels=self.render_channels,
                capture_channels=self.capture_channels,
                initial_delay_ms=self.stream_delay_ms,
                enable_high_pass=True,
            )
            print(f"✅ AEC Processor инициализирован (задержка: {self.stream_delay_ms}ms)")
        except Exception as e:
            print(f"⚠️ Ошибка инициализации AEC: {e}")
            self._aec = None
    
    def process(self, near_audio: np.ndarray, far_audio: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Обрабатывает аудио с AEC с буферизацией.
        
        Args:
            near_audio: Аудио с микрофона (ближний сигнал) float32
            far_audio: Эталонный сигнал (то, что воспроизводится) float32
            
        Returns:
            np.ndarray: Очищенное аудио
        """
        # ✅ Инициализируем в этом потоке, если ещё не сделано
        self._ensure_initialized()
        
        if not self.available:
            return near_audio
        
        with self._lock:
            # Добавляем в буферы
            self._near_buffer = np.concatenate([self._near_buffer, near_audio.astype(np.float32)])
            
            # Far-буфер: берем только если есть far_audio
            if far_audio is not None and len(far_audio) > 0:
                self._far_buffer = np.concatenate([self._far_buffer, far_audio.astype(np.float32)])
            
            # Если накопилось достаточно данных
            if len(self._near_buffer) >= self.buffer_samples:
                return self._process_buffer()
            else:
                # Возвращаем тишину, пока не накопилось достаточно
                return np.zeros(len(near_audio), dtype=np.float32)
    
    def _process_buffer(self) -> np.ndarray:
        """Обрабатывает накопленный буфер."""
        output = []
        
        # Берем весь буфер или его часть
        process_samples = (len(self._near_buffer) // self.frame_samples) * self.frame_samples
        near_data = self._near_buffer[:process_samples]
        self._near_buffer = self._near_buffer[process_samples:]
        
        # Far-данные (если есть)
        far_data = None
        if len(self._far_buffer) >= process_samples:
            far_data = self._far_buffer[:process_samples]
            self._far_buffer = self._far_buffer[process_samples:]
        
        # Обрабатываем по фреймам
        for i in range(0, len(near_data), self.frame_samples):
            near_frame = near_data[i:i+self.frame_samples]
            
            if len(near_frame) < self.frame_samples:
                break
            
            try:
                # Подаем far_end сигнал (если есть)
                if far_data is not None and i < len(far_data):
                    far_frame = far_data[i:i+self.frame_samples]
                    if len(far_frame) == self.frame_samples:
                        self._aec.handle_render_frame(far_frame)
                
                # Обрабатываем capture
                out, metrics = self._aec.process_capture_frame(
                    near_frame,
                    level_change=False
                )
                output.append(out)
                self._processed_frames += 1
                
            except Exception as e:
                print(f"⚠️ Ошибка AEC обработки: {e}")
                self._dropped_frames += 1
                output.append(near_frame)
        
        if not output:
            return np.zeros(self.buffer_samples, dtype=np.float32)
        
        result = np.concatenate(output)
        
        # Если результат короче запрошенного, дополняем нулями
        if len(result) < self.buffer_samples:
            result = np.pad(result, (0, self.buffer_samples - len(result)))
        
        return result
    
    def flush(self) -> np.ndarray:
        """Сбрасывает оставшиеся данные в буфере."""
        self._ensure_initialized()
        
        with self._lock:
            if len(self._near_buffer) == 0:
                return np.array([], dtype=np.float32)
            
            # Дополняем нулями до целого числа фреймов
            remainder = len(self._near_buffer) % self.frame_samples
            if remainder > 0:
                pad = self.frame_samples - remainder
                self._near_buffer = np.pad(self._near_buffer, (0, pad))
            
            return self._process_buffer()
    
    def reset(self):
        """Сбрасывает AEC процессор и буферы."""
        with self._lock:
            self._near_buffer = np.array([], dtype=np.float32)
            self._far_buffer = np.array([], dtype=np.float32)
            
            # ✅ Безопасное удаление AEC в правильном потоке
            if self._aec is not None:
                try:
                    # Проверяем, что мы в том же потоке
                    if self._thread_id == threading.get_ident():
                        self._aec = None
                    else:
                        # Если в другом потоке - просто сбрасываем ссылку
                        # AEC будет пересоздан при следующем вызове
                        self._aec = None
                        self._initialized = False
                except Exception as e:
                    logger.error(f"Ошибка удаления AEC: {e}")
                    self._aec = None
            
            self._initialized = False
            self._processed_frames = 0
            self._dropped_frames = 0
        
        # Пересоздаём в этом же потоке если необходимо
        if self._thread_id == threading.get_ident():
            self._ensure_initialized()
    
    @staticmethod
    def is_available() -> bool:
        return AEC_AVAILABLE
    
    @property
    def available(self) -> bool:
        self._ensure_initialized()
        return self._aec is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику."""
        return {
            "processed_frames": self._processed_frames,
            "dropped_frames": self._dropped_frames,
            "buffer_size": len(self._near_buffer),
            "far_buffer_size": len(self._far_buffer),
            "is_available": self.available,
            "initialized": self._initialized,
            "thread_id": self._thread_id,
        }