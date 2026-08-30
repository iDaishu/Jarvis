"""Шумоподавление для улучшения распознавания."""

import numpy as np
import threading
from pathlib import Path
from typing import Optional, Tuple

try:
    import noisereduce as nr
    NR_AVAILABLE = True
except ImportError:
    NR_AVAILABLE = False


class NoiseReducer:
    """Шумоподавление аудио."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        prop_decrease: float = 0.8,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        adaptive: bool = True
    ):
        self.sample_rate = sample_rate
        self.prop_decrease = prop_decrease
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.adaptive = adaptive
        
        self.noise_profile = None
        self.noise_profile_std = None
        self.has_profile = False
        self._lock = threading.Lock()
        
        if not NR_AVAILABLE:
            print("⚠️ noisereduce не установлен")
            print("   Установите: pip install noisereduce")
    
    def is_available(self) -> bool:
        """Проверяет доступность шумоподавления."""
        return NR_AVAILABLE
    
    def capture_noise_profile(self, audio: np.ndarray, duration: float = 1.0) -> bool:
        """Захватывает профиль шума."""
        if not NR_AVAILABLE:
            return False
        
        try:
            samples = int(duration * self.sample_rate)
            if len(audio) < samples:
                self.noise_profile = audio.copy()
            else:
                self.noise_profile = audio[:samples].copy()
            
            self.noise_profile_std = np.std(self.noise_profile)
            self.has_profile = True
            
            print(f"✅ Профиль шума сохранён (std: {self.noise_profile_std:.4f})")
            return True
            
        except Exception as e:
            print(f"⚠️ Ошибка захвата шума: {e}")
            return False
    
    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Удаляет шум из аудио."""
        if not self.has_profile or not NR_AVAILABLE:
            return audio
        
        if len(audio) < self.sample_rate * 0.1:  # Слишком коротко
            return audio
        
        try:
            with self._lock:
                reduced = nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    y_noise=self.noise_profile,
                    prop_decrease=self.prop_decrease,
                    time_constant_s=1.0,
                    freq_mask_smooth_hz=500,
                    time_mask_smooth_ms=50,
                    n_std_thresh_stationary=1.5,
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length,
                    use_tensorflow=False,
                )
                
                # Если результат слишком тихий, возвращаем оригинал
                if np.abs(reduced).max() < 0.01:
                    return audio
                
                return reduced
                
        except Exception as e:
            print(f"⚠️ Ошибка шумоподавления: {e}")
            return audio
    
    def reduce_noise_adaptive(self, audio: np.ndarray) -> np.ndarray:
        """Адаптивное шумоподавление."""
        if not self.adaptive or not self.has_profile:
            return self.reduce_noise(audio)
        
        try:
            # Разбиваем на сегменты
            segment_duration = 0.5
            segment_samples = int(segment_duration * self.sample_rate)
            
            if len(audio) < segment_samples:
                return self.reduce_noise(audio)
            
            segments = []
            for i in range(0, len(audio), segment_samples):
                segment = audio[i:i+segment_samples]
                if len(segment) >= segment_samples // 2:
                    segment_std = np.std(segment)
                    
                    # Корректируем prop_decrease в зависимости от уровня шума
                    if segment_std > self.noise_profile_std * 2:
                        prop = min(0.95, self.prop_decrease * 1.2)
                    elif segment_std < self.noise_profile_std * 0.5:
                        prop = max(0.5, self.prop_decrease * 0.8)
                    else:
                        prop = self.prop_decrease
                    
                    reduced = nr.reduce_noise(
                        y=segment,
                        sr=self.sample_rate,
                        y_noise=self.noise_profile,
                        prop_decrease=prop,
                        n_fft=self.n_fft,
                        win_length=self.win_length,
                        hop_length=self.hop_length,
                        use_tensorflow=False,
                    )
                    segments.append(reduced)
                else:
                    segments.append(segment)
            
            return np.concatenate(segments)
            
        except Exception as e:
            print(f"⚠️ Ошибка адаптивного шумоподавления: {e}")
            return audio
    
    def reset(self):
        """Сбрасывает профиль шума."""
        self.noise_profile = None
        self.noise_profile_std = None
        self.has_profile = False
        print("🔄 Профиль шума сброшен")