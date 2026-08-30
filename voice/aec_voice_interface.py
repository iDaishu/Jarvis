# voice/aec_voice_interface.py
"""Голосовой интерфейс с AEC и AudioPipeline."""

import threading
import queue
import time
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional, Callable, Dict, Any

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


class AECVoiceInterface:
    """Голосовой интерфейс с AEC и AudioPipeline."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        base_dir: Optional[Path] = None,
        sample_rate: int = 16000,
        stream_delay_ms: int = 100
    ):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.config = config or {}
        self.sample_rate = sample_rate
        
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
        self.command_queue = queue.Queue()
        self.running = True
        
        # Callback
        self.on_transcription = None
        
        # Статистика
        self.stats = {
            "commands_processed": 0,
            "asr_errors": 0,
            "tts_errors": 0,
            "echo_prevented": 0,
        }
        
        # Инициализация
        self._init_components()
        self._init_pipeline(stream_delay_ms)
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
                model_path=Path('models/vosk/vosk-model-ru-0.22'),
                sample_rate=self.sample_rate,
                partial_results=True
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
                tts = SileroTTS(language='ru', speaker='xenia')
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
                    
                    @property
                    def is_speaking(self):
                        return self._speaking
                    
                    def speak(self, text, async_mode=True):
                        if not text:
                            return False
                        try:
                            self._speaking = True
                            self.engine.say(text)
                            if not async_mode:
                                self.engine.runAndWait()
                                self._speaking = False
                            else:
                                def _speak():
                                    self.engine.runAndWait()
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
                
                return TTSWrapper(engine)
            except Exception as e:
                print(f"⚠️ Ошибка pyttsx3: {e}")
        
        print("❌ TTS не доступен")
        return None
    
    def _init_emotion_tts(self):
        """Инициализация эмоционального TTS."""
        if self.config.get('emotions', {}).get('enabled', True):
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
    
    def _init_pipeline(self, stream_delay_ms: int):
        """Инициализация AudioPipeline."""
        config = PipelineConfig(
            sample_rate=self.sample_rate,
            stream_delay_ms=stream_delay_ms
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
        def worker():
            while self.running:
                try:
                    text = self.command_queue.get(timeout=0.1)
                    if text and self.on_transcription:
                        self.is_processing = True
                        try:
                            self.on_transcription(text)
                        finally:
                            self.is_processing = False
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"⚠️ Ошибка обработки: {e}")
                    self.is_processing = False
        
        threading.Thread(target=worker, daemon=True).start()
    
    @property
    def is_speaking(self):
        """Возвращает, говорит ли агент."""
        if self.tts and hasattr(self.tts, 'is_speaking'):
            return self.tts.is_speaking
        return self._is_speaking
    
    def _on_asr_final(self, text: str):
        """Обработка финального результата ASR."""
        if not text or len(text) < 2:
            return
        
        # Защита от дублирования
        if hasattr(self, '_last_text') and text == self._last_text:
            return
        self._last_text = text
        
        self.stats["commands_processed"] += 1
        self.command_queue.put(text)
    
    def _on_asr_partial(self, text: str):
        """Обработка частичного результата ASR."""
        if text and len(text) > 1:
            print(f"⌛ {text}", end="\r")
    
    def _on_asr_error(self, error: str):
        """Обработка ошибки ASR."""
        print(f"\n⚠️ Ошибка ASR: {error}")
        self.stats["asr_errors"] += 1
    
    def speak(self, text: str, async_mode: bool = True, emotion: Optional[str] = None) -> bool:
        """Озвучивает текст с оптимизацией скорости."""
        if not text or not text.strip():
            return False
        
        if not self.tts:
            print(f"🔊 {text}")
            return False
        
        # Проверяем, не говорим ли уже
        if self.is_speaking and not async_mode:
            return False
        
        self._is_speaking = True
        
        try:
            # Убираем эмодзи и лишние символы
            clean_text = text
            if self.emotion_tts:
                clean_text, _, _ = self.emotion_tts.enhance_text(text)
            
            # Укорачиваем текст если слишком длинный
            if len(clean_text) > 500:
                clean_text = clean_text[:500] + "..."
            
            # Проверяем, есть ли метод synthesize
            if hasattr(self.tts, 'synthesize'):
                audio = self.tts.synthesize(clean_text)
            else:
                # Fallback на speak
                result = self.tts.speak(clean_text, async_mode)
                if not async_mode:
                    self._is_speaking = False
                return result
            
            # Проверяем аудио
            if audio is not None and len(audio) > 0:
                if not isinstance(audio, np.ndarray):
                    audio = np.array(audio, dtype=np.float32)
                
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                
                # Нормализация
                max_val = np.abs(audio).max()
                if max_val > 0:
                    audio = audio / max_val * 0.95
                
                # Определяем частоту дискретизации TTS
                tts_sample_rate = getattr(self.tts, 'sample_rate', 48000)
                
                # Отправляем эталон в AEC (ресемплинг до 16kHz)
                if self.pipeline and tts_sample_rate != 16000:
                    # Простой ресемплинг: берем каждый N-й семпл
                    ratio = tts_sample_rate // 16000
                    if ratio > 1 and len(audio) > ratio:
                        audio_aec = audio[::ratio]
                        if len(audio_aec) > 0:
                            self.pipeline.set_far_end_audio(audio_aec.copy())
                elif self.pipeline:
                    self.pipeline.set_far_end_audio(audio.copy())
                
                # Воспроизводим с минимальной задержкой
                if async_mode:
                    def _play():
                        try:
                            sd.play(audio, tts_sample_rate, blocking=False)
                            sd.wait()
                        except Exception as e:
                            print(f"⚠️ Ошибка воспроизведения: {e}")
                        finally:
                            self._is_speaking = False
                    
                    threading.Thread(target=_play, daemon=True).start()
                    return True
                else:
                    sd.play(audio, tts_sample_rate)
                    sd.wait()
                    self._is_speaking = False
                    return True
            else:
                # Fallback
                result = self.tts.speak(clean_text, async_mode)
                if not async_mode:
                    self._is_speaking = False
                else:
                    def _release():
                        time.sleep(0.5)
                        self._is_speaking = False
                    threading.Thread(target=_release, daemon=True).start()
                return result
                
        except Exception as e:
            print(f"⚠️ Ошибка озвучивания: {e}")
            self.stats["tts_errors"] += 1
            self._is_speaking = False
            return False

    
    def speak_with_emotion(self, text: str, emotion: str = "радость", async_mode: bool = True) -> bool:
        return self.speak(text, async_mode, emotion=emotion)
    
    def start_listening(self, on_transcription: Optional[Callable] = None):
        """Запускает прослушивание."""
        if not self.asr:
            print("❌ ASR не доступен")
            return False
        
        if self.is_listening:
            print("⚠️ Уже слушаю")
            return True
        
        self.on_transcription = on_transcription
        self.is_listening = True
        
        # Запускаем ASR
        self.asr.start_listening(
            on_final=self._on_asr_final,
            on_partial=self._on_asr_partial,
            on_error=self._on_asr_error
        )
        
        # Запускаем AudioPipeline
        if self.pipeline:
            self.pipeline.start()
        
        print("🎤 Прослушивание запущено (AEC активен)")
        return True
    
    def stop_listening(self):
        """Останавливает прослушивание."""
        self.is_listening = False
        
        if self.pipeline:
            self.pipeline.stop()
        
        if self.asr:
            self.asr.stop_listening()
        
        print("⏹️ Прослушивание остановлено")
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats["is_listening"] = self.is_listening
        stats["is_speaking"] = self.is_speaking
        if self.pipeline:
            stats["pipeline"] = self.pipeline.get_stats()
        if self.asr:
            stats["asr"] = self.asr.get_stats()
        return stats
    
    def stop(self):
        """Полная остановка."""
        print("\n⏹️ Остановка AEC Voice Interface...")
        self.running = False
        self.is_listening = False
        self._is_speaking = False
        
        self.stop_listening()
        
        if self.tts and hasattr(self.tts, 'stop'):
            try:
                self.tts.stop()
            except:
                pass
        
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except:
                break
        
        print(f"📊 Статистика: {self.stats}")
        print("✅ AEC Voice Interface остановлен")