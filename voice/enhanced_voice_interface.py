# voice/enhanced_voice_interface.py (с fallback TTS)

import threading
import time
import queue
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np

from .whisper_asr import WhisperASR
from .noise_reduction import NoiseReducer
from .emotional_tts import EmotionTTS
from .voice_profile import VoiceProfile

# Пробуем импортировать TTS
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
    """Голосовой интерфейс с улучшенным ASR и TTS."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.config = config or {}
        
        # Пути
        self.memory_dir = self.base_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # TTS engine (pyttsx3 как резерв)
        self.tts_engine = None
        
        # Инициализация компонентов
        self._init_components()
        
        # Состояние
        self.is_listening = False
        self.command_queue = queue.Queue()
        self.running = True
        self.is_speaking = False
        self._lock = threading.Lock()
        
        # Статистика
        self.stats = {
            "commands_processed": 0,
            "listening_time": 0,
            "asr_errors": 0,
            "tts_errors": 0
        }
        
        # Запуск обработки
        self._start_worker()
        
        print("✅ Enhanced Voice Interface готов")
    
    def _init_components(self):
        """Инициализирует все компоненты."""
        # ASR - Whisper
        self.asr = self._init_asr()
        
        # TTS - пробуем Silero, затем pyttsx3
        self.tts = self._init_tts()
        
        # Шумоподавление
        self.noise_reducer = self._init_noise_reducer()
        
        # Эмоциональный TTS
        self.emotion_tts = self._init_emotion_tts()
        
        # Профиль голоса
        self.voice_profile = self._init_voice_profile()
    
    def _init_asr(self):
        """Инициализация ASR."""
        asr_config = self.config.get('asr', {})
        model_size = asr_config.get('whisper_model', 'small')
        language = asr_config.get('language', 'ru')
        
        try:
            asr = WhisperASR(model_size=model_size, language=language)
            print(f"✅ ASR: Whisper ({model_size})")
            return asr
        except Exception as e:
            print(f"❌ Ошибка инициализации Whisper: {e}")
            print("   Установите: pip install openai-whisper")
            return None
    
    def _init_tts(self):
        """Инициализация TTS с fallback."""
        # Пробуем Silero
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
                
                # Настройка голоса
                voices = self.tts_engine.getProperty('voices')
                for voice in voices:
                    if 'ru' in str(voice.languages) or 'Russian' in voice.name:
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                
                self.tts_engine.setProperty('rate', 170)
                print("✅ TTS: pyttsx3 (fallback)")
                
                # Создаём обёртку для совместимости
                class Pyttsx3Wrapper:
                    def __init__(self, engine):
                        self.engine = engine
                        self.is_speaking = False
                    
                    def speak(self, text, async_mode=True, **kwargs):
                        if not text:
                            return False
                        try:
                            self.is_speaking = True
                            self.engine.say(text)
                            if not async_mode:
                                self.engine.runAndWait()
                            else:
                                # Для асинхронности запускаем в отдельном потоке
                                import threading
                                def _speak():
                                    self.engine.runAndWait()
                                    self.is_speaking = False
                                thread = threading.Thread(target=_speak, daemon=True)
                                thread.start()
                            return True
                        except Exception as e:
                            print(f"⚠️ Ошибка pyttsx3: {e}")
                            self.is_speaking = False
                            return False
                    
                    def stop(self):
                        try:
                            self.engine.stop()
                        except:
                            pass
                
                return Pyttsx3Wrapper(self.tts_engine)
                
            except Exception as e:
                print(f"⚠️ Ошибка pyttsx3: {e}")
        
        # Если ничего не работает
        print("❌ TTS не доступен")
        return None
    
    def _init_noise_reducer(self):
        """Инициализация шумоподавления."""
        nr_config = self.config.get('noise_reduction', {})
        if nr_config.get('enabled', True):
            try:
                reducer = NoiseReducer()
                print("✅ Шумоподавление включено")
                return reducer
            except Exception as e:
                print(f"⚠️ Ошибка инициализации шумоподавления: {e}")
        return None
    
    def _init_emotion_tts(self):
        """Инициализация эмоционального TTS."""
        if self.config.get('emotions', {}).get('enabled', True):
            try:
                emotion = EmotionTTS()
                print("✅ Эмоциональный TTS включён")
                return emotion
            except Exception as e:
                print(f"⚠️ Ошибка инициализации эмоционального TTS: {e}")
        return None
    
    def _init_voice_profile(self):
        """Инициализация профиля голоса."""
        try:
            profile_path = self.memory_dir / "voice_profile.json"
            profile = VoiceProfile(profile_path)
            if profile.is_calibrated():
                print(f"✅ Профиль голоса загружен ({profile.get_stats()['sample_count']} образцов)")
            else:
                print("📝 Профиль голоса не калиброван")
            return profile
        except Exception as e:
            print(f"⚠️ Ошибка инициализации профиля: {e}")
        return None
    
    def speak(self, text: str, async_mode: bool = True,
              emotion: Optional[str] = None,
              speed: float = 1.0, pitch: float = 1.0,
              energy: float = 1.0) -> bool:
        """Озвучивание с эмоциональной окраской."""
        if not text or not text.strip():
            return False
        
        if not self.tts:
            print(f"🔊 {text}")
            return False
        
        try:
            clean_text = text
            
            # Добавляем эмоциональную окраску (только для текста)
            if self.emotion_tts:
                clean_text, params, emotion_name = self.emotion_tts.enhance_text(text)
                speed = params.get('speed', speed)
                energy = params.get('energy', energy)
            
            # Проверяем длину
            if len(clean_text) > 500:
                clean_text = clean_text[:500] + "..."
            
            # Озвучиваем
            self.is_speaking = True
            result = self.tts.speak(clean_text, async_mode)
            if not async_mode:
                self.is_speaking = False
            return result
            
        except Exception as e:
            print(f"⚠️ Ошибка озвучивания: {e}")
            self.stats["tts_errors"] += 1
            self.is_speaking = False
            return False
    
    def speak_with_emotion(self, text: str, emotion: str, 
                          async_mode: bool = True) -> bool:
        """Озвучивает текст с указанной эмоцией."""
        return self.speak(text, async_mode, emotion=emotion)
    
    def start_listening(self, on_transcription: Optional[Callable] = None):
        """Запуск прослушивания."""
        if not self.asr:
            print("⚠️ ASR не доступен")
            return False
        
        if self.is_listening:
            print("🎤 Уже слушаю")
            return True
        
        self.is_listening = True
        
        def asr_callback(text: str, is_final: bool = False):
            if is_final and text:
                print(f"📝 Распознано: {text}")
                self.stats["commands_processed"] += 1
                
                # Добавляем в профиль
                if self.voice_profile:
                    try:
                        self.voice_profile.add_sample(text, np.array([0]))
                    except:
                        pass
                
                # Отправляем в очередь
                self.command_queue.put((text, on_transcription))
        
        try:
            self.asr.start_listening(asr_callback)
            print("🎤 Прослушивание запущено")
            return True
        except Exception as e:
            print(f"❌ Ошибка запуска прослушивания: {e}")
            self.is_listening = False
            return False
    
    def stop_listening(self):
        """Останавливает прослушивание."""
        self.is_listening = False
        
        if self.asr:
            try:
                self.asr.stop()
            except:
                pass
        
        print("⏹️ Прослушивание остановлено")
    
    def _start_worker(self):
        """Запуск потока обработки команд."""
        def worker():
            while self.running:
                try:
                    text, callback = self.command_queue.get(timeout=0.1)
                    if callback and text:
                        try:
                            callback(text)
                        except Exception as e:
                            print(f"⚠️ Ошибка в callback: {e}")
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"⚠️ Ошибка обработки: {e}")
        
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
    
    def is_speaking(self) -> bool:
        """Проверяет, говорит ли агент."""
        return self.is_speaking
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику."""
        stats = self.stats.copy()
        
        if self.voice_profile:
            stats["voice_profile"] = self.voice_profile.get_stats()
        
        return stats
    
    def calibrate_voice(self, duration: float = 3.0) -> bool:
        """Калибрует голос пользователя."""
        if not self.noise_reducer or not self.noise_reducer.is_available():
            print("⚠️ Шумоподавление не доступно")
            return False
        
        print(f"🎤 Калибровка голоса... Говорите {duration} секунд")
        
        try:
            import sounddevice as sd
            
            # Записываем аудио
            audio = sd.rec(int(duration * 16000), 16000, channels=1, dtype=np.float32)
            sd.wait()
            audio = audio.flatten()
            
            # Сохраняем профиль шума
            result = self.noise_reducer.capture_noise_profile(audio, duration)
            
            if result:
                print("✅ Калибровка завершена")
                return True
            else:
                print("❌ Ошибка калибровки")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка калибровки: {e}")
            return False
    
    def stop(self):
        """Полная остановка интерфейса."""
        print("\n⏹️ Остановка голосового интерфейса...")
        
        self.running = False
        self.is_listening = False
        
        # Останавливаем компоненты
        if hasattr(self, 'asr') and self.asr:
            try:
                self.asr.stop()
            except:
                pass
        
        if hasattr(self, 'tts') and self.tts:
            try:
                self.tts.stop()
            except:
                pass
        
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        
        # Очищаем очередь
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except:
                break
        
        print(f"📊 Статистика: {self.stats}")
        print("✅ Голосовой интерфейс остановлен")