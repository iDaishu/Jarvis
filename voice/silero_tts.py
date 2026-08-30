# voice/silero_tts.py
"""Silero TTS для синтеза речи с оптимизацией."""

import torch
import sounddevice as sd
import numpy as np
import threading
import queue
import time
from pathlib import Path
from typing import Optional, List, Dict
import functools

class SileroTTS:
    """Синтез речи через Silero TTS с оптимизацией."""
    
    _model_cache: Dict[str, 'SileroTTS'] = {}
    
    def __init__(
        self,
        model_name: str = "v5_4_ru",
        speaker: str = "xenia",
        language: str = "ru",
        device: str = "cpu",
        sample_rate: int = 48000,
        use_cache: bool = True
    ):
        self.model_name = model_name
        self.speaker = speaker
        self.language = language
        self.sample_rate = sample_rate
        self.use_cache = use_cache
        
        # Выбор устройства (CPU для скорости)
        self.device = torch.device("cpu")
        print(f"🔧 Используется устройство: {self.device}")
        
        # Пути
        self.base_dir = Path(__file__).resolve().parent.parent
        self.models_dir = self.base_dir / "models" / "silero"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Кэш синтезированного аудио
        self.audio_cache: Dict[str, np.ndarray] = {}
        self.cache_size = 50
        
        # Состояние
        self.is_speaking = False
        self.running = True
        self._speaker_lock = threading.Lock()
        self._queue = queue.Queue()
        self._thread = None
        
        # Загрузка модели
        self.model = None
        self._load_model()
        
        # Запуск фонового потока
        self._start_worker()
    
    def _load_model(self):
        """Загружает модель с оптимизацией."""
        model_path = self.models_dir / f"{self.model_name}.pt"
        
        if not model_path.exists():
            print(f"📥 Скачивание модели {self.model_name}...")
            try:
                torch.hub.download_url_to_file(
                    f'https://models.silero.ai/models/tts/ru/{self.model_name}.pt',
                    str(model_path)
                )
                print("✅ Модель скачана")
            except Exception as e:
                print(f"⚠️ Ошибка скачивания: {e}")
                raise
        
        print(f"📂 Загрузка модели {model_path.name}...")
        try:
            # Загружаем модель через PackageImporter
            self.model = torch.package.PackageImporter(str(model_path)).load_pickle("tts_models", "model")
            
            # Перемещаем на CPU
            self.model.to(self.device)
            
            # Используем try/except для eval, так как не все модели имеют этот метод
            try:
                self.model.eval()
            except AttributeError:
                print("ℹ️ Модель не поддерживает eval(), продолжаем...")
            
            # Если CUDA доступна, используем её
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.model.to(self.device)
                print("🚀 Используется GPU для TTS")
            else:
                # На CPU используем оптимизации
                torch.set_num_threads(max(1, torch.get_num_threads() // 2))
                print(f"🧠 Используется CPU с {torch.get_num_threads()} потоками")
            
            print(f"✅ Silero TTS готов (голос: {self.speaker})")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            raise
    
    def _start_worker(self):
        """Запускает фоновый поток для асинхронного синтеза."""
        self.running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
    
    def _worker(self):
        """Фоновый поток для синтеза речи."""
        while self.running:
            try:
                text, speaker, callback = self._queue.get(timeout=0.1)
                if text:
                    audio = self._synthesize_sync(text, speaker)
                    if callback:
                        callback(audio)
                    self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Ошибка в TTS worker: {e}")
    
    def _synthesize_sync(self, text: str, speaker: Optional[str] = None) -> Optional[np.ndarray]:
        """Синхронный синтез речи."""
        if not text or not text.strip() or self.model is None:
            return None
        
        speaker = speaker or self.speaker
        
        try:
            # Проверяем кэш
            cache_key = f"{text[:100]}_{speaker}"
            if cache_key in self.audio_cache:
                return self.audio_cache[cache_key]
            
            # Получаем тензор от модели
            audio_tensor = self.model.apply_tts(
                text=text[:200],  # Ограничиваем длину для скорости
                speaker=speaker,
                sample_rate=self.sample_rate
            )
            
            # Конвертируем в numpy
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().detach().numpy().astype(np.float32)
            else:
                audio_np = np.array(audio_tensor, dtype=np.float32)
            
            if audio_np is None or len(audio_np) == 0:
                return None
            
            # Нормализация
            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val * 0.95
            
            # Сохраняем в кэш
            if len(self.audio_cache) >= self.cache_size:
                keys = list(self.audio_cache.keys())[:self.cache_size // 2]
                for key in keys:
                    del self.audio_cache[key]
            self.audio_cache[cache_key] = audio_np
            
            return audio_np
            
        except Exception as e:
            print(f"⚠️ Ошибка синтеза: {e}")
            return None
    
    def synthesize(self, text: str, speaker: Optional[str] = None) -> np.ndarray:
        """Синтезирует речь синхронно."""
        audio = self._synthesize_sync(text, speaker)
        if audio is not None:
            return audio
        return np.array([], dtype=np.float32)
    
    def synthesize_async(self, text: str, callback=None, speaker: Optional[str] = None):
        """Асинхронный синтез речи."""
        if not text or not text.strip():
            if callback:
                callback(None)
            return
        self._queue.put((text, speaker, callback))
    
    def speak(self, text: str, async_mode: bool = True, speaker: Optional[str] = None) -> bool:
        """Озвучивает текст."""
        if not text or not text.strip() or self.model is None:
            return False
        
        with self._speaker_lock:
            if self.is_speaking and async_mode:
                return False
        
        try:
            audio = self.synthesize(text, speaker)
            if len(audio) == 0:
                return False
            
            self.is_speaking = True
            
            if async_mode:
                def _play():
                    try:
                        sd.play(audio, self.sample_rate, blocking=False)
                        sd.wait()
                    except Exception as e:
                        print(f"⚠️ Ошибка воспроизведения: {e}")
                    finally:
                        self.is_speaking = False
                
                threading.Thread(target=_play, daemon=True).start()
                return True
            else:
                sd.play(audio, self.sample_rate)
                sd.wait()
                self.is_speaking = False
                return True
            
        except Exception as e:
            print(f"⚠️ Ошибка озвучивания: {e}")
            self.is_speaking = False
            return False
    
    def stop(self):
        """Останавливает TTS."""
        self.running = False
        try:
            sd.stop()
        except:
            pass
        self.is_speaking = False
        
        # Очищаем очередь
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except:
                break
    
    def clear_cache(self):
        """Очищает кэш аудио."""
        self.audio_cache.clear()
    
    def get_speakers(self) -> List[str]:
        if self.model and hasattr(self.model, 'speakers'):
            return self.model.speakers
        return ['xenia', 'baya', 'natasha', 'ruslan', 'irina', 'kseniya']