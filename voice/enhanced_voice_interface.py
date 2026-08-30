# voice/enhanced_voice_interface.py
"""Голосовой интерфейс с надёжным перезапуском ASR."""

import threading
import queue
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np

from .vosk_asr import VoskASR
from .noise_reduction import NoiseReducer
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


class EnhancedVoiceInterface:
    """Голосовой интерфейс с надёжным перезапуском ASR."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        base_dir: Optional[Path] = None
    ):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.config = config or {}
        self.sample_rate = 16000
        
        # Пути
        self.memory_dir = self.base_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # TTS engine
        self.tts_engine = None
        
        # Компоненты
        self.asr = None
        self.tts = None
        self.noise_reducer = None
        self.emotion_tts = None
        self.voice_profile = None
        
        # Состояние
        self.is_listening = False
        self._is_speaking = False
        self.is_processing = False
        self.command_queue = queue.Queue()
        self.running = True
        self._lock = threading.Lock()
        
        # Защита от эхо
        self.speech_lock = threading.Lock()
        self.speech_cooldown = 0.8  # Уменьшил до 0.8 сек
        self.last_speech_time = 0
        
        # Для перезапуска ASR
        self._asr_restart_needed = False
        self._restart_thread = None
        self._monitor_running = True
        self._asr_starting = False  # Флаг, что ASR запускается
        
        # Callback
        self.on_transcription = None
        
        # Статистика
        self.stats = {
            "commands_processed": 0,
            "asr_errors": 0,
            "tts_errors": 0,
            "echo_prevented": 0,
            "asr_restarts": 0
        }
        
        # Инициализация
        self._init_components()
        
        # Запускаем обработку
        self._start_worker()
        
        # Запускаем фоновый поток для перезапуска ASR
        self._start_restart_monitor()
        
        print("✅ Enhanced Voice Interface готов")
    
    def _init_components(self):
        """Инициализирует все компоненты."""
        self.asr = self._init_asr()
        self.tts = self._init_tts()
        self.noise_reducer = self._init_noise_reducer()
        self.emotion_tts = self._init_emotion_tts()
        self.voice_profile = self._init_voice_profile()
    
    def _init_asr(self):
        """Инициализация ASR."""
        asr_config = self.config.get('asr', {})
        model_path = asr_config.get('model_path', 'models/vosk/vosk-model-ru-0.22')
        
        try:
            asr = VoskASR(
                model_path=Path(model_path),
                sample_rate=self.sample_rate,
                partial_results=True
            )
            print("✅ ASR: Vosk")
            return asr
        except Exception as e:
            print(f"❌ Ошибка инициализации Vosk: {e}")
            print("💡 Запустите: python download_vosk_model.py")
            return None
    
    def _init_tts(self):
        """Инициализация TTS с fallback."""
        if SILERO_AVAILABLE:
            try:
                tts_config = self.config.get('tts', {})
                language = tts_config.get('language', 'ru')
                speaker = tts_config.get('speaker', 'xenia')
                
                tts = SileroTTS(language=language, speaker=speaker)
                print(f"✅ TTS: Silero ({speaker})")
                return tts
            except Exception as e:
                print(f"⚠️ Ошибка Silero: {e}")
        
        # Fallback на pyttsx3
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                
                voices = self.tts_engine.getProperty('voices')
                for voice in voices:
                    if 'ru' in str(voice.languages) or 'Russian' in voice.name:
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                
                self.tts_engine.setProperty('rate', 170)
                print("✅ TTS: pyttsx3 (fallback)")
                
                class Pyttsx3Wrapper:
                    def __init__(self, engine):
                        self.engine = engine
                        self._is_speaking = False
                        self._lock = threading.Lock()
                    
                    @property
                    def is_speaking(self):
                        return self._is_speaking
                    
                    def speak(self, text, async_mode=True, **kwargs):
                        if not text:
                            return False
                        try:
                            with self._lock:
                                self._is_speaking = True
                                self.engine.say(text)
                                if not async_mode:
                                    self.engine.runAndWait()
                                    self._is_speaking = False
                                else:
                                    def _speak():
                                        self.engine.runAndWait()
                                        self._is_speaking = False
                                    thread = threading.Thread(target=_speak, daemon=True)
                                    thread.start()
                            return True
                        except Exception as e:
                            print(f"⚠️ Ошибка pyttsx3: {e}")
                            self._is_speaking = False
                            return False
                    
                    def stop(self):
                        try:
                            self.engine.stop()
                            self._is_speaking = False
                        except:
                            pass
                
                return Pyttsx3Wrapper(self.tts_engine)
                
            except Exception as e:
                print(f"⚠️ Ошибка pyttsx3: {e}")
        
        print("❌ TTS не доступен")
        return None
    
    def _init_noise_reducer(self):
        """Инициализация шумоподавления."""
        nr_config = self.config.get('noise_reduction', {})
        if nr_config.get('enabled', True):
            try:
                from .noise_reduction import NoiseReducer
                reducer = NoiseReducer()
                print("✅ Шумоподавление включено")
                return reducer
            except Exception as e:
                print(f"⚠️ Ошибка шумоподавления: {e}")
        return None
    
    def _init_emotion_tts(self):
        """Инициализация эмоционального TTS."""
        if self.config.get('emotions', {}).get('enabled', True):
            try:
                emotion = EmotionTTS()
                print("✅ Эмоциональный TTS включён")
                return emotion
            except Exception as e:
                print(f"⚠️ Ошибка эмоционального TTS: {e}")
        return None
    
    def _init_voice_profile(self):
        """Инициализация профиля голоса."""
        try:
            profile_path = self.memory_dir / "voice_profile.json"
            profile = VoiceProfile(profile_path)
            if profile.is_calibrated():
                print(f"✅ Профиль голоса загружен")
            else:
                print("📝 Профиль голоса не калиброван")
            return profile
        except Exception as e:
            print(f"⚠️ Ошибка профиля: {e}")
        return None
    
    @property
    def is_speaking(self):
        """Возвращает, говорит ли агент."""
        if self.tts and hasattr(self.tts, 'is_speaking'):
            return self.tts.is_speaking
        return self._is_speaking
    
    def _can_listen(self) -> bool:
        """Проверяет, можно ли слушать."""
        if self.is_speaking:
            return False
        if time.time() - self.last_speech_time < self.speech_cooldown:
            return False
        return True
    
    def _on_asr_result(self, text: str):
        """Вызывается при результате ASR."""
        if not self._can_listen():
            self.stats["echo_prevented"] += 1
            return
        
        if text and len(text) > 1:
            self.stats["commands_processed"] += 1
            self.command_queue.put(text)
            # Помечаем, что нужен перезапуск ASR, но с задержкой
            self._schedule_asr_restart()
    
    def _schedule_asr_restart(self):
        """Планирует перезапуск ASR через задержку."""
        def delayed_restart():
            # Ждём, пока закончится обработка и речь
            time.sleep(self.speech_cooldown + 0.5)
            # Проверяем, не говорит ли агент
            while self.is_speaking:
                time.sleep(0.2)
            if self.is_listening and not self._asr_starting:
                self._asr_restart_needed = True
        
        thread = threading.Thread(target=delayed_restart, daemon=True)
        thread.start()
    
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
                            self.last_speech_time = time.time()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"⚠️ Ошибка обработки: {e}")
                    self.is_processing = False
        
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
    
    def _start_restart_monitor(self):
        """Фоновый поток для перезапуска ASR."""
        def monitor():
            while self.running and self._monitor_running:
                try:
                    # Проверяем, нужен ли перезапуск
                    if (self._asr_restart_needed and 
                        self.is_listening and 
                        not self.is_speaking and 
                        not self.is_processing and
                        not self._asr_starting):
                        
                        self._asr_restart_needed = False
                        # Перезапускаем ASR
                        self._restart_asr()
                    
                    time.sleep(0.5)
                except Exception as e:
                    print(f"⚠️ Ошибка в мониторе: {e}")
                    time.sleep(1)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        print("🔄 Монитор ASR запущен")
    
    def _restart_asr(self):
        """Перезапускает ASR."""
        if not self.asr or not self.is_listening:
            return
        
        if self.is_speaking or self.is_processing:
            return
        
        self._asr_starting = True
        
        try:
            # Останавливаем текущий ASR
            self.asr.stop()
            time.sleep(0.2)
            
            # Сбрасываем и запускаем заново
            self.asr.reset()
            
            # Обёртка для on_final с защитой от дублирования
            last_final_text = [""]
            last_final_time = [0]
            
            def on_final(text):
                # Защита от дублирования
                current_time = time.time()
                if text == last_final_text[0] and current_time - last_final_time[0] < 2.0:
                    return
                last_final_text[0] = text
                last_final_time[0] = current_time
                self._on_asr_result(text)
            
            def on_partial(text):
                if self._can_listen():
                    # Печатаем только если текст изменился
                    if hasattr(self, '_last_partial') and self._last_partial != text:
                        print(f"⌛ {text}", end="\r")
                    self._last_partial = text
            
            def on_error(error):
                if "input overflow" in str(error):
                    return
                print(f"\n⚠️ Ошибка ASR: {error}")
                self.stats["asr_errors"] += 1
                # При ошибке планируем перезапуск
                if self.is_listening:
                    self._schedule_asr_restart()
            
            self.asr.start_listening(
                on_final=on_final,
                on_partial=on_partial,
                on_error=on_error
            )
            self.stats["asr_restarts"] += 1
            
        except Exception as e:
            print(f"⚠️ Ошибка перезапуска ASR: {e}")
            self._asr_restart_needed = True
        finally:
            self._asr_starting = False
    
    def speak(self, text: str, async_mode: bool = True, emotion: Optional[str] = None) -> bool:
        """Озвучивает текст."""
        if not text or not text.strip():
            return False
        
        if not self.tts:
            print(f"🔊 {text}")
            return False
        
        # Останавливаем ASR перед речью
        if self.asr:
            try:
                self.asr.stop()
            except:
                pass
        
        try:
            clean_text = text
            
            # Эмоциональная окраска
            if self.emotion_tts:
                if emotion:
                    emotions = self.emotion_tts.analyze_emotion(text)
                    emotions[emotion] = 1.0
                    for markers in self.emotion_tts.emotion_markers.values():
                        for marker in markers:
                            clean_text = clean_text.replace(marker, "")
                else:
                    clean_text, params, detected_emotion = self.emotion_tts.enhance_text(text)
                    emotion = detected_emotion
            
            if len(clean_text) > 500:
                clean_text = clean_text[:500] + "..."
            
            # Блокируем прослушивание
            self._is_speaking = True
            self.last_speech_time = time.time()
            self._asr_restart_needed = False
            
            # Останавливаем воспроизведение
            if self.tts_engine:
                try:
                    self.tts_engine.stop()
                except:
                    pass
            
            # Говорим
            result = self.tts.speak(clean_text, async_mode)
            
            # Снимаем блокировку через 0.5 секунды
            def release_lock():
                time.sleep(0.5)
                self._is_speaking = False
                # Планируем перезапуск ASR после речи
                if self.is_listening:
                    self._schedule_asr_restart()
            
            threading.Thread(target=release_lock, daemon=True).start()
            
            return result
            
        except Exception as e:
            print(f"⚠️ Ошибка озвучивания: {e}")
            self.stats["tts_errors"] += 1
            self._is_speaking = False
            if self.is_listening:
                self._schedule_asr_restart()
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
        self._asr_restart_needed = True
        self._asr_starting = False
        
        print("🎤 Прослушивание запущено")
        return True
    
    def stop_listening(self):
        """Останавливает прослушивание."""
        self.is_listening = False
        self._asr_restart_needed = False
        self._asr_starting = False
        
        if self.asr:
            try:
                self.asr.stop()
            except:
                pass
        
        print("⏹️ Прослушивание остановлено")
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        if self.asr:
            stats["asr_stats"] = self.asr.get_stats()
        return stats
    
    def stop(self):
        """Полная остановка."""
        print("\n⏹️ Остановка голосового интерфейса...")
        self.running = False
        self._monitor_running = False
        self.is_listening = False
        self._is_speaking = False
        self._asr_restart_needed = False
        self._asr_starting = False
        
        self.stop_listening()
        
        if self.tts:
            try:
                self.tts.stop()
            except:
                pass
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except:
                break
        
        print(f"📊 Статистика: {self.stats}")
        print("✅ Голосовой интерфейс остановлен")