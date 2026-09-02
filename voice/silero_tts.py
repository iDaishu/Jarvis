# voice/silero_tts.py
"""
Silero TTS для синтеза речи с оптимизацией.
Поддерживает длинные фразы через разбивку на части.
"""

import torch
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any
from functools import lru_cache

from utils.logger import LoggerFactory
from utils.exceptions import VoiceError

logger = LoggerFactory.get_logger("silero_tts")


class SileroTTS:
    """
    Синтез речи через Silero TTS с поддержкой длинных фраз.
    """
    
    _model_cache: Dict[str, 'SileroTTS'] = {}
    
    def __init__(
        self,
        model_name: str = "v5_4_ru",
        speaker: str = "xenia",
        language: str = "ru",
        device: Optional[str] = None,
        sample_rate: int = 16000,
        cache_size: int = 50,
        max_text_length: int = 500,  # ✅ Увеличено с 200
        chunk_size: int = 150,        # ✅ Размер чанка для разбивки
        chunk_gap: float = 0.15,      # ✅ Пауза между чанками
    ):
        self.model_name = model_name
        self.speaker = speaker
        self.language = language
        self.sample_rate = sample_rate
        self.cache_size = cache_size
        self.max_text_length = max_text_length
        self.chunk_size = chunk_size
        self.chunk_gap = chunk_gap
        
        # Выбор устройства
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"TTS использует устройство: {self.device}, sample_rate: {sample_rate} Гц")
        
        # Пути
        self.base_dir = Path(__file__).resolve().parent.parent
        self.models_dir = self.base_dir / "models" / "silero"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Кэш
        self._audio_cache: Dict[str, np.ndarray] = {}
        self._cache_access: Dict[str, float] = {}
        self._cache_lock = threading.Lock()
        self._cache_size = cache_size
        
        # Состояние
        self._is_speaking = False
        self._running = True
        self._shutdown_event = threading.Event()
        self._speaker_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=100)  # ✅ Увеличено
        self._thread: Optional[threading.Thread] = None
        
        # Таймаут
        self._speaking_timeout = 30.0  # ✅ Увеличено для длинных фраз
        self._speaking_start_time = 0.0
        
        # ✅ Очередь для воспроизведения чанков
        self._play_queue: queue.Queue = queue.Queue(maxsize=50)
        self._play_thread: Optional[threading.Thread] = None
        
        # Модель
        self._model = None
        self._load_model()
        
        # Запуск
        self._start_worker()
        self._start_play_worker()
    
    def _load_model(self) -> None:
        """Загрузка модели Silero."""
        model_path = self.models_dir / f"{self.model_name}.pt"
        
        if not model_path.exists():
            logger.info(f"Скачивание модели {self.model_name}...")
            try:
                torch.hub.download_url_to_file(
                    f'https://models.silero.ai/models/tts/ru/{self.model_name}.pt',
                    str(model_path)
                )
                logger.info("Модель скачана")
            except Exception as e:
                logger.error(f"Ошибка скачивания модели: {e}")
                raise VoiceError(f"Не удалось скачать модель: {e}")
        
        try:
            logger.info(f"Загрузка модели {model_path.name}...")
            self._model = torch.package.PackageImporter(str(model_path)).load_pickle(
                "tts_models", "model"
            )
            self._model.to(self.device)
            
            try:
                self._model.eval()
                if self.device.type == "cpu":
                    torch.set_num_threads(min(2, torch.get_num_threads()))
                    logger.info(f"CPU оптимизация: {torch.get_num_threads()} потоков")
            except Exception as e:
                logger.debug(f"Оптимизация модели: {e}")
            
            logger.info(f"Silero TTS готов (голос: {self.speaker})")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise VoiceError(f"Не удалось загрузить модель: {e}")
    
    def _start_worker(self) -> None:
        """Запуск фонового потока для синтеза."""
        self._running = True
        self._shutdown_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="SileroTTS")
        self._thread.start()
    
    def _start_play_worker(self) -> None:
        """Запуск фонового потока для воспроизведения чанков."""
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True, name="TTSPlay")
        self._play_thread.start()
    
    def _worker(self) -> None:
        """Фоновый поток для синтеза речи."""
        while self._running and not self._shutdown_event.is_set():
            try:
                text, speaker, callback = self._queue.get(timeout=0.1)
                
                if not self._running or self._shutdown_event.is_set():
                    break
                
                if text:
                    try:
                        # ✅ Разбиваем длинный текст на чанки
                        chunks = self._split_text(text)
                        logger.debug(f"Текст разбит на {len(chunks)} чанков")
                        
                        for i, chunk in enumerate(chunks):
                            if not self._running or self._shutdown_event.is_set():
                                break
                            
                            audio = self._synthesize_sync(chunk, speaker)
                            if audio is not None and len(audio) > 0:
                                # ✅ Добавляем в очередь воспроизведения
                                self._play_queue.put((audio, i == len(chunks) - 1))
                            else:
                                logger.warning(f"Пустой аудио чанк {i+1}/{len(chunks)}")
                        
                        if callback:
                            callback(True)
                            
                    except Exception as e:
                        logger.error(f"Ошибка синтеза в worker: {e}")
                    finally:
                        self._queue.task_done()
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка в TTS worker: {e}")
    
    def _play_worker(self) -> None:
        """Фоновый поток для воспроизведения чанков."""
        while self._running and not self._shutdown_event.is_set():
            try:
                audio, is_last = self._play_queue.get(timeout=0.1)
                
                if not self._running or self._shutdown_event.is_set():
                    break
                
                if audio is not None and len(audio) > 0:
                    try:
                        sd.play(audio, self.sample_rate, blocking=False)
                        sd.wait()
                        
                        # ✅ Маленькая пауза между чанками (для естественности)
                        if not is_last:
                            time.sleep(self.chunk_gap)
                    except Exception as e:
                        logger.error(f"Ошибка воспроизведения чанка: {e}")
                    finally:
                        self._play_queue.task_done()
                        if is_last:
                            self._is_speaking = False
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка в play worker: {e}")
    
    def _split_text(self, text: str) -> List[str]:
        """
        Разбивает длинный текст на чанки по предложениям.
        
        Args:
            text: Исходный текст
            
        Returns:
            List[str]: Список чанков
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        # ✅ Разбиваем по предложениям
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # ✅ Если один чанк всё ещё слишком длинный - режем по словам
        if len(chunks) == 1 and len(chunks[0]) > self.chunk_size:
            words = chunks[0].split()
            chunks = []
            current_chunk = ""
            for word in words:
                if len(current_chunk) + len(word) <= self.chunk_size:
                    current_chunk += word + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word + " "
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def _resample_audio(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Ресемплинг аудио."""
        if from_rate == to_rate:
            return audio
        
        try:
            import scipy.signal
            duration = len(audio) / from_rate
            new_length = int(duration * to_rate)
            resampled = scipy.signal.resample(audio, new_length)
            return resampled.astype(np.float32)
        except ImportError:
            logger.warning("scipy не установлен, используется простая интерполяция")
            ratio = to_rate / from_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    @lru_cache(maxsize=50)
    def _synthesize_cached(self, text: str, speaker: str) -> Optional[np.ndarray]:
        """Синтез с кэшированием."""
        if not text or not text.strip():
            return None
        
        if self._model is None:
            logger.error("Модель не загружена")
            return None
        
        try:
            text = text[:self.max_text_length]
            
            audio_tensor = self._model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=48000
            )
            
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().detach().numpy().astype(np.float32)
            else:
                audio_np = np.array(audio_tensor, dtype=np.float32)
            
            if audio_np is None or len(audio_np) == 0:
                logger.warning("Пустой аудио результат")
                return None
            
            if self.sample_rate != 48000:
                audio_np = self._resample_audio(audio_np, 48000, self.sample_rate)
            
            max_val = np.abs(audio_np).max()
            if max_val > 0:
                audio_np = audio_np / max_val * 0.85
            
            return audio_np
            
        except Exception as e:
            logger.error(f"Ошибка синтеза: {e}")
            return None
    
    def _synthesize_sync(self, text: str, speaker: Optional[str] = None) -> Optional[np.ndarray]:
        """Синхронный синтез речи."""
        speaker = speaker or self.speaker
        return self._synthesize_cached(text[:self.max_text_length], speaker)
    
    def synthesize(self, text: str, speaker: Optional[str] = None) -> np.ndarray:
        """Синхронный синтез речи (весь текст целиком)."""
        audio = self._synthesize_sync(text, speaker)
        return audio if audio is not None else np.array([], dtype=np.float32)
    
    def synthesize_async(self, text: str, callback: Optional[Callable] = None,
                         speaker: Optional[str] = None) -> bool:
        """Асинхронный синтез речи с разбивкой на чанки."""
        if not text or not text.strip():
            if callback:
                callback(False)
            return False
        
        try:
            self._queue.put_nowait((text, speaker, callback))
            return True
        except queue.Full:
            logger.warning("Очередь TTS переполнена")
            return False
    
    def speak(self, text: str, async_mode: bool = True,
              speaker: Optional[str] = None) -> bool:
        """
        Озвучивание текста с поддержкой длинных фраз.
        """
        if not text or not text.strip():
            return False
        
        if self._model is None:
            logger.error("Модель не загружена")
            return False
        
        # ✅ Если текст короткий - используем прямой синтез
        if len(text) <= self.chunk_size:
            with self._speaker_lock:
                if self._is_speaking:
                    if time.time() - self._speaking_start_time > self._speaking_timeout:
                        logger.warning("TTS таймаут, принудительный сброс")
                        self._is_speaking = False
                        try:
                            sd.stop()
                        except:
                            pass
                    else:
                        logger.debug("TTS уже говорит, пропуск")
                        return False
                
                self._is_speaking = True
                self._speaking_start_time = time.time()
            
            try:
                audio = self.synthesize(text, speaker)
                if len(audio) == 0:
                    self._is_speaking = False
                    return False
                
                if async_mode:
                    def _play():
                        try:
                            sd.play(audio, self.sample_rate, blocking=False)
                            sd.wait()
                        except Exception as e:
                            logger.error(f"Ошибка воспроизведения: {e}")
                        finally:
                            self._is_speaking = False
                    
                    threading.Thread(target=_play, daemon=True, name="TTSPlay").start()
                    return True
                else:
                    sd.play(audio, self.sample_rate)
                    sd.wait()
                    self._is_speaking = False
                    return True
                
            except Exception as e:
                logger.error(f"Ошибка озвучивания: {e}")
                self._is_speaking = False
                return False
        
        # ✅ Длинный текст - используем асинхронный синтез с чанками
        else:
            with self._speaker_lock:
                if self._is_speaking:
                    if time.time() - self._speaking_start_time > self._speaking_timeout:
                        logger.warning("TTS таймаут, принудительный сброс")
                        self._is_speaking = False
                        try:
                            sd.stop()
                        except:
                            pass
                    else:
                        logger.debug("TTS уже говорит, пропуск")
                        return False
                
                self._is_speaking = True
                self._speaking_start_time = time.time()
            
            return self.synthesize_async(text, speaker=speaker)
    
    def stop(self, timeout: float = 3.0) -> None:
        """Остановка TTS."""
        print("⏹️ Остановка Silero TTS...")
        
        self._running = False
        self._shutdown_event.set()
        
        try:
            sd.stop()
        except Exception as e:
            logger.debug(f"Ошибка остановки звука: {e}")
        
        self._is_speaking = False
        
        # Очищаем очереди
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        
        while not self._play_queue.empty():
            try:
                self._play_queue.get_nowait()
                self._play_queue.task_done()
            except queue.Empty:
                break
        
        # Ждём завершения потоков
        for thread in [self._thread, self._play_thread]:
            if thread and thread.is_alive():
                print(f"⏳ Ожидание завершения {thread.name}...")
                thread.join(timeout=timeout)
                if thread.is_alive():
                    print(f"⚠️ {thread.name} не завершился")
        
        self.clear_cache()
        logger.info("TTS остановлен")
    
    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._synthesize_cached.cache_clear()
        logger.debug("TTS кэш очищен")
    
    def get_speakers(self) -> List[str]:
        """Список голосов."""
        if self._model and hasattr(self._model, 'speakers'):
            return self._model.speakers
        return ['xenia', 'baya', 'natasha', 'ruslan', 'irina', 'kseniya']
    
    @property
    def is_speaking(self) -> bool:
        """Проверка, говорит ли TTS."""
        if self._is_speaking and time.time() - self._speaking_start_time > self._speaking_timeout:
            self._is_speaking = False
            try:
                sd.stop()
            except:
                pass
        return self._is_speaking
    
    @property
    def available(self) -> bool:
        return self._model is not None
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_speaking": self.is_speaking,
            "queue_size": self._queue.qsize(),
            "play_queue_size": self._play_queue.qsize(),
            "cache_size": self._synthesize_cached.cache_info().currsize,
            "cache_hits": self._synthesize_cached.cache_info().hits,
            "cache_misses": self._synthesize_cached.cache_info().misses,
            "available": self.available,
            "device": str(self.device),
            "sample_rate": self.sample_rate,
        }