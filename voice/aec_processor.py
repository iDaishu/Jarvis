# voice/aec_processor.py
"""AEC (Acoustic Echo Cancellation) процессор."""

import numpy as np
from typing import Optional

try:
    from pywebrtc_audio import AudioProcessor
    AEC_AVAILABLE = True
except ImportError:
    AEC_AVAILABLE = False


class AECProcessor:
    """AEC процессор на основе pywebrtc-audio."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        stream_delay_ms: int = 100,
        high_pass_filter: bool = True
    ):
        self.sample_rate = sample_rate
        self.stream_delay_ms = stream_delay_ms
        self.processor = None
        
        if AEC_AVAILABLE:
            try:
                self.processor = AudioProcessor(
                    sample_rate=sample_rate,
                    num_channels=1,
                    echo_cancellation=True,
                    high_pass_filter=high_pass_filter,
                    stream_delay_ms=stream_delay_ms,
                )
                print(f"✅ AEC Processor инициализирован (задержка: {stream_delay_ms}ms)")
            except Exception as e:
                print(f"⚠️ Ошибка инициализации AEC: {e}")
                self.processor = None
    
    def process(self, near_audio: np.ndarray, far_audio: np.ndarray) -> np.ndarray:
        """
        Обрабатывает аудио с AEC.
        
        Args:
            near_audio: Аудио с микрофона (ближний сигнал)
            far_audio: Эталонный сигнал (то, что воспроизводится)
            
        Returns:
            np.ndarray: Очищенное аудио
        """
        if not self.is_available():
            return near_audio
        
        if len(near_audio) != len(far_audio):
            # Если длины не совпадают, обрезаем или дополняем
            min_len = min(len(near_audio), len(far_audio))
            near_audio = near_audio[:min_len]
            far_audio = far_audio[:min_len]
        
        try:
            return self.processor.process(near_audio, far_audio)
        except Exception as e:
            print(f"⚠️ Ошибка AEC обработки: {e}")
            return near_audio
    
    def reset(self):
        """Сбрасывает AEC процессор."""
        if self.processor and hasattr(self.processor, 'reset'):
            try:
                self.processor.reset()
            except:
                pass
    
    @staticmethod
    def is_available() -> bool:
        """Проверяет доступность AEC."""
        return AEC_AVAILABLE
    
    @property
    def available(self) -> bool:
        return self.processor is not None