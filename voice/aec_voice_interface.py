# voice/aec_voice_interface.py
"""Голосовой интерфейс с AEC и AudioPipeline."""

import threading
import queue
import time
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from .audio_pipeline import AudioPipeline, PipelineConfig
from .vosk_asr import VoskASR
from .emotional_tts import EmotionTTS
from .voice_profile import VoiceProfile

try:
    from .silero_tts import SileroTTS
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


@dataclass
class VoiceConfig:
    """Конфигурация голосового интерфейса."""
    sample_rate: int = 16000
    stream_delay_ms: int = 100
    asr_model: str = "models/vosk/vosk-model-ru-0.22"
    tts_language: str = "ru"
    tts_speaker: str = "xenia"
    emotions_enabled: bool = True
    silence_timeout: float = 0.6
    min_phrase_length: int = 2
    noise_gate_threshold: float = 0.005


class AECVoiceInterface:
    """Голосовой интерфейс с AEC и AudioPipeline."""
    
    def __init__(
        self,
        config: Optional[VoiceConfig] = None,
        base_dir: Optional[Path] = None,
    ):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.config = config or VoiceConfig()
        
        # Компоненты
        self.asr = None
        self.tts = None
        self.emotion_tts = None
        self.voice_profile = None
        self.pipeline = None
        
        # Состояние
        self.is_listening = False
        self._is_speaking = False
        self.is_processing = False
        self.command_queue: queue.Queue = queue.Queue(maxsize=100)
        self.running = True
        self._shutdown_event = threading.Event()
        
        # Callback
        self.on_transcription = None
        
        # Потоки
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop_event = threading.Event()
        
        # Статистика
        self.stats = {
            "commands_processed": 0,
            "asr_errors": 0,
            "tts_errors": 0,
            "echo_prevented": 0,
            "queue_size": 0,
        }
        
        # Инициализация
        self._init_components()
        self._init_pipeline()
        self._start_worker()
        
        print("✅ AEC Voice Interface готов")
    
    def _init_components(self):
        """Инициализирует компоненты."""
        self.asr = self._init_asr()
        self.tts = self._init_tts()
        self.emotion_tts = self._init_emotion_tts()
        self.voice_profile = self._init_voice_profile()
    
    def _init_asr(self):
        """Инициализация ASR."""
        try:
            asr = VoskASR(
                model_path=Path(self.config.asr_model),
                sample_rate=self.config.sample_rate,
                partial_results=True,
                silence_timeout=self.config.silence_timeout,
                min_phrase_length=self.config.min_phrase_length,
            )
            print("✅ ASR: Vosk")
            return asr
        except Exception as e:
            print(f"❌ Ошибка ASR: {e}")
            return None
    
    def _init_tts(self):
        """Инициализация TTS."""
        if SILERO_AVAILABLE:
            try:
                tts = SileroTTS(
                    language=self.config.tts_language,
                    speaker=self.config.tts_speaker
                )
                print("✅ TTS: Silero")
                return tts
            except Exception as e:
                print(f"⚠️ Ошибка Silero: {e}")
        
        if PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                for voice in voices:
                    if 'ru' in str(voice.languages) or 'Russian' in voice.name:
                        engine.setProperty('voice', voice.id)
                        break
                engine.setProperty('rate', 170)
                print("✅ TTS: pyttsx3 (fallback)")
                
                class TTSWrapper:
                    def __init__(self, eng):
                        self.engine = eng
                        self._speaking = False
                        self._lock = threading.Lock()
                    
                    @property
                    def is_speaking(self):
                        return self._speaking
                    
                    def speak(self, text, async_mode=True):
                        if not text:
                            return False
                        try:
                            with self._lock:
                                self._speaking = True
                                self.engine.say(text)
                                if not async_mode:
                                    self.engine.runAndWait()
                                    self._speaking = False
                                else:
                                    def _speak():
                                        try:
                                            self.engine.runAndWait()
                                        finally:
                                            self._speaking = False
                                    threading.Thread(target=_speak, daemon=True).start()
                            return True
                        except Exception as e:
                            print(f"⚠️ TTS ошибка: {e}")
                            self._speaking = False
                            return False
                    
                    def stop(self):
                        try:
                            self.engine.stop()
                            self._speaking = False
                        except:
                            pass
                    
                    def synthesize(self, text):
                        # pyttsx3 не поддерживает синтез без воспроизведения
                        return None
                
                return TTSWrapper(engine)
            except Exception as e:
                print(f"⚠️ Ошибка pyttsx3: {e}")
        
        print("❌ TTS не доступен")
        return None
    
    def _init_emotion_tts(self):
        """Инициализация эмоционального TTS."""
        if self.config.emotions_enabled:
            try:
                return EmotionTTS()
            except Exception as e:
                print(f"⚠️ Ошибка эмоций: {e}")
        return None
    
    def _init_voice_profile(self):
        """Инициализация профиля голоса."""
        try:
            profile = VoiceProfile(self.base_dir / "memory" / "voice_profile.json")
            return profile
        except Exception as e:
            print(f"⚠️ Ошибка профиля: {e}")
            return None
    
    def _init_pipeline(self):
        """Инициализация AudioPipeline."""
        config = PipelineConfig(
            sample_rate=self.config.sample_rate,
            stream_delay_ms=self.config.stream_delay_ms,
            noise_gate_threshold=self.config.noise_gate_threshold,
        )
        
        self.pipeline = AudioPipeline(
            config=config,
            on_audio=self._on_audio_from_pipeline
        )
    
    def _on_audio_from_pipeline(self, audio: np.ndarray):
        """Callback из AudioPipeline."""
        if not self.is_listening or not self.asr:
            return
        
        # Конвертируем в 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        self.asr.feed_audio(audio_int16.tobytes())
    
    def _start_worker(self):
        """Запуск потока обработки команд."""
        self._worker_stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="VoiceWorker")
        self._worker_thread.start()
        print("🧵 VoiceWorker запущен")
    
    def _worker(self):
        """Фоновый поток для обработки команд."""
        while self.running and not self._worker_stop_event.is_set():
            try:
                text = self.command_queue.get(timeout=0.1)
                self.stats["queue_size"] = self.command_queue.qsize()
                
                if text and self.on_transcription:
                    self.is_processing = True
                    try:
                        self.on_transcription(text)
                        self.stats["commands_processed"] += 1
                    except Exception as e:
                        print(f"⚠️ Ошибка обработки команды: {e}")
                    finally:
                        self.is_processing = False
                        self.command_queue.task_done()
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Ошибка в worker: {e}")
                self.is_processing = False
    
    @property
    def is_speaking(self):
        """Возвращает, говорит ли агент."""
        if self.tts and hasattr(self.tts, 'is_speaking'):
            return self.tts.is_speaking
        return self._is_speaking
    
    def _on_asr_final(self, text: str):
        """Обработка финального результата ASR."""
        if not text or len(text) < self.config.min_phrase_length:
            return
        
        # Защита от дублирования
        if hasattr(self, '_last_text') and text == self._last_text:
            return
        self._last_text = text
        
        try:
            self.command_queue.put_nowait(text)
        except queue.Full:
            print(f"⚠️ Очередь команд переполнена: {text[:30]}...")
            self.stats["asr_errors"] += 1
    
    def _on_asr_partial(self, text: str):
        """Обработка частичного результата ASR."""
        if text and len(text) > 1:
            print(f"⌛ {text}", end="\r")
    
    def _on_asr_error(self, error: str):
        """Обработка ошибки ASR."""
        print(f"\n⚠️ Ошибка ASR: {error}")
        self.stats["asr_errors"] += 1
    
    def speak(self, text: str, async_mode: bool = True, emotion: Optional[str] = None) -> bool:
        """Озвучивает текст с поддержкой длинных фраз."""
        if not text or not text.strip():
            return False
        
        if not self.tts:
            print(f"🔊 {text}")
            return False
        
        # ✅ Защита от зависания
        if self._is_speaking and async_mode:
            if time.time() - self._speaking_start_time > 15.0:
                print("⚠️ TTS завис, принудительный сброс")
                self._is_speaking = False
                try:
                    sd.stop()
                except:
                    pass
            else:
                print("⚠️ TTS уже говорит, пропуск")
                return False
        
        self._is_speaking = True
        self._speaking_start_time = time.time()
        
        try:
            clean_text = text
            emotion_params = None
            emotion_name = emotion or "нейтральная"
            
            if self.emotion_tts:
                clean_text, emotion_params, emotion_name = self.emotion_tts.enhance_text(text)
            
            # ✅ Не обрезаем текст для длинных фраз
            # TTS сам разобьёт на чанки
            
            if async_mode:
                # ✅ Используем асинхронный синтез с чанками
                def on_tts_done(success):
                    if not success:
                        print("⚠️ TTS синтез не удался")
                    self._is_speaking = False
                
                # ✅ Передаём far_end сигнал в AEC через callback
                def on_chunk_ready(audio):
                    if audio is not None and len(audio) > 0:
                        if self.pipeline:
                            self.pipeline.feed_far_end(audio.copy())
                
                # ✅ Используем синтез с чанками
                return self.tts.synthesize_async(clean_text, on_tts_done)
            else:
                # Синхронный режим (для тестирования)
                audio = self.tts.synthesize(clean_text)
                if audio is not None and len(audio) > 0:
                    if self.pipeline:
                        self.pipeline.feed_far_end(audio.copy())
                    sd.play(audio, self.config.sample_rate)
                    sd.wait()
                    self._is_speaking = False
                    return True
                else:
                    self._is_speaking = False
                    return False
                    
        except Exception as e:
            print(f"⚠️ Ошибка озвучивания: {e}")
            self.stats["tts_errors"] += 1
            self._is_speaking = False
            return False
    
    def speak_with_emotion(self, text: str, emotion: str = "радость", async_mode: bool = True) -> bool:
        return self.speak(text, async_mode, emotion=emotion)
    
    def start_listening(self, on_transcription: Optional[Callable] = None) -> bool:
        """Запускает прослушивание."""
        if not self.asr:
            print("❌ ASR не доступен")
            return False
        
        if self.is_listening:
            print("⚠️ Уже слушаю")
            return True
        
        self.on_transcription = on_transcription
        self.is_listening = True
        self._last_text = ""
        
        self.asr.start_listening(
            on_final=self._on_asr_final,
            on_partial=self._on_asr_partial,
            on_error=self._on_asr_error
        )
        
        if self.pipeline:
            self.pipeline.start()
        
        print("🎤 Прослушивание запущено (AEC активен)")
        return True
    
    def stop_listening(self) -> None:
        """Останавливает прослушивание."""
        self.is_listening = False
        
        if self.pipeline:
            self.pipeline.stop()
        
        if self.asr:
            self.asr.stop_listening()
        
        # Очищаем очередь команд
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except:
                break
        
        print("⏹️ Прослушивание остановлено")
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats["is_listening"] = self.is_listening
        stats["is_speaking"] = self.is_speaking
        stats["is_processing"] = self.is_processing
        stats["queue_size"] = self.command_queue.qsize()
        
        if self.pipeline:
            stats["pipeline"] = self.pipeline.get_stats()
        if self.asr:
            stats["asr"] = self.asr.get_stats()
        return stats
    
    def stop(self, timeout: float = 2.0) -> None:
        """Полная остановка с ожиданием потоков."""
        print("\n⏹️ Остановка AEC Voice Interface...")
        
        # 1. Сигнал остановки
        self.running = False
        self.is_listening = False
        self._shutdown_event.set()
        self._worker_stop_event.set()
        
        # 2. Останавливаем прослушивание
        self.stop_listening()
        
        # 3. Останавливаем TTS
        if self.tts and hasattr(self.tts, 'stop'):
            try:
                self.tts.stop()
            except:
                pass
        
        # 4. Ждём завершения worker
        if self._worker_thread and self._worker_thread.is_alive():
            print("⏳ Ожидание завершения VoiceWorker...")
            self._worker_thread.join(timeout=timeout)
            if self._worker_thread.is_alive():
                print("⚠️ VoiceWorker не завершился за отведённое время")
        
        # 5. Очищаем очередь
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except:
                break
        
        print(f"📊 Статистика: {self.stats}")
        print("✅ AEC Voice Interface остановлен")