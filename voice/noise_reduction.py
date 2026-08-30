# voice/noise_reduction.py
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
    
    def __init__(self, sample_rate: int = 16000, 
                 prop_decrease: float = 0.8,
                 n_fft: int = 2048,
                 win_length: int = 2048,
                 hop_length: int = 512):
        self.sample_rate = sample_rate
        self.prop_decrease = prop_decrease
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        
        self.noise_profile = None
        self.noise_profile_std = None
        self.has_profile = False
        self._lock = threading.Lock()
        
        # Проверка доступности
        if not NR_AVAILABLE:
            print("⚠️ noisereduce не установлен. Установите: pip install noisereduce")
    
    def is_available(self) -> bool:
        """Проверяет доступность шумоподавления."""
        return NR_AVAILABLE
    
    def capture_noise_profile(self, audio: np.ndarray, 
                              duration: float = 1.0) -> bool:
        """Захватывает профиль шума."""
        if not NR_AVAILABLE:
            return False
        
        try:
            samples = int(duration * self.sample_rate)
            if len(audio) < samples:
                # Если аудио короче, используем всё
                self.noise_profile = audio.copy()
            else:
                self.noise_profile = audio[:samples].copy()
            
            # Вычисляем стандартное отклонение шума
            self.noise_profile_std = np.std(self.noise_profile)
            self.has_profile = True
            
            print(f"✅ Профиль шума сохранён (std: {self.noise_profile_std:.4f})")
            return True
            
        except Exception as e:
            print(f"⚠️ Ошибка захвата шума: {e}")
            return False
    
    def reduce_noise(self, audio: np.ndarray, 
                     use_adaptive: bool = True) -> np.ndarray:
        """Удаляет шум из аудио."""
        if not self.has_profile or not NR_AVAILABLE:
            return audio
        
        if len(audio) < self.sample_rate * 0.1:  # Слишком коротко
            return audio
        
        try:
            with self._lock:
                # Используем библиотеку noisereduce
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
                
                # Если результат слишком тихий, нормализуем
                if np.abs(reduced).max() < 0.01:
                    return audio
                
                return reduced
                
        except Exception as e:
            print(f"⚠️ Ошибка шумоподавления: {e}")
            return audio
    
    def reduce_noise_adaptive(self, audio: np.ndarray) -> np.ndarray:
        """Адаптивное шумоподавление."""
        if not self.has_profile or not NR_AVAILABLE:
            return audio
        
        try:
            # Разбиваем на сегменты
            segment_duration = 0.5  # 500ms
            segment_samples = int(segment_duration * self.sample_rate)
            
            if len(audio) < segment_samples:
                return self.reduce_noise(audio)
            
            # Адаптивно изменяем параметры для каждого сегмента
            segments = []
            for i in range(0, len(audio), segment_samples):
                segment = audio[i:i+segment_samples]
                if len(segment) > segment_samples // 2:
                    # Оцениваем уровень шума в сегменте
                    segment_std = np.std(segment)
                    
                    # Корректируем prop_decrease
                    if segment_std > self.noise_profile_std * 2:
                        prop = min(0.95, self.prop_decrease * 1.2)
                    elif segment_std < self.noise_profile_std * 0.5:
                        prop = max(0.5, self.prop_decrease * 0.8)
                    else:
                        prop = self.prop_decrease
                    
                    # Применяем шумоподавление с индивидуальными параметрами
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
            
            # Объединяем сегменты
            return np.concatenate(segments)
            
        except Exception as e:
            print(f"⚠️ Ошибка адаптивного шумоподавления: {e}")
            return audio

# Тестирование
def test_noise_reduction():
    """Тест шумоподавления."""
    print("\n" + "="*60)
    print("🎤 ТЕСТ ШУМОПОДАВЛЕНИЯ")
    print("="*60)
    
    if not NR_AVAILABLE:
        print("❌ noisereduce не установлен")
        return
    
    reducer = NoiseReducer()
    
    # Создаём тестовый сигнал с шумом
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Чистый сигнал
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Шум
    noise = 0.05 * np.random.randn(len(t))
    # Сигнал с шумом
    noisy = clean + noise
    
    # Захватываем профиль шума
    reducer.capture_noise_profile(noise[:int(0.3 * sample_rate)])
    
    # Шумоподавление
    reduced = reducer.reduce_noise(noisy)
    
    # Вычисляем SNR
    def compute_snr(signal, noise):
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        return 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    snr_before = compute_snr(clean, noisy - clean)
    snr_after = compute_snr(clean, reduced - clean)
    
    print(f"\n📊 SNR до: {snr_before:.2f} dB")
    print(f"📊 SNR после: {snr_after:.2f} dB")
    print(f"📈 Улучшение: {snr_after - snr_before:.2f} dB")
    
    print("\n✅ Тест завершён.")

if __name__ == "__main__":
    test_noise_reduction()