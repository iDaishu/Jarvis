"""
voice_interface.py
ГОЛОСОВОЙ ИНТЕРФЕЙС - Vosk + Piper (постоянное прослушивание)
"""

import threading
import time
import queue
from pathlib import Path
from typing import Optional, Callable

# TTS - Piper
try:
    from piper_tts import PiperTTS
except ImportError:
    PiperTTS = None

# ASR - Vosk
try:
    from vosk_interface import VoskInterface, VoiceConfig
except ImportError:
    VoskInterface = None
    VoiceConfig = None


class VoiceInterface:
    """
    Голосовой интерфейс с постоянным прослушиванием.
    Распознаёт любую речь и передаёт её на обработку.
    """
    
    def __init__(self, wake_words: list = None, language: str = "ru-RU"):
        self.language = language
        self.asr = None
        self.tts = None
        
        self.is_initialized = False
        self.is_listening = False
        self.voice_mode_active = False
        
        # Для обработки очереди команд
        self.command_queue = queue.Queue()
        self.command_thread = None
        self.is_processing = False
        
        # Инициализация
        self._init_tts()
        self._init_asr()
        
        if self.tts and self.asr:
            self.is_initialized = True
            print("🎤 Голосовой интерфейс готов (постоянное прослушивание)")
            
            # Запускаем поток обработки команд
            self._start_command_processor()
    
    def _init_tts(self):
        """Инициализация Piper TTS."""
        if PiperTTS is None:
            print("⚠️ PiperTTS не доступен")
            return
        
        try:
            self.tts = PiperTTS()
            if self.tts.is_ready:
                print("✅ TTS (Piper) готов")
        except Exception as e:
            print(f"⚠️ Ошибка Piper TTS: {e}")
    
    def _init_asr(self):
        """Инициализация Vosk ASR."""
        if VoskInterface is None:
            print("⚠️ VoskInterface не доступен")
            return

        try:
            from vosk_interface import find_vosk_model
            model_path = find_vosk_model()
            
            if model_path:
                config = VoiceConfig()
                config.model_path = str(model_path)
                config.wake_words = []  # Не используем wake word
                self.asr = VoskInterface(config)
                print("✅ ASR (Vosk) готов")
            else:
                print("❌ Модель Vosk не найдена")
                
        except Exception as e:
            print(f"⚠️ Ошибка Vosk: {e}")
    
    def _start_command_processor(self):
        """Запускает поток для обработки команд из очереди."""
        self.command_thread = threading.Thread(target=self._process_commands, daemon=True)
        self.command_thread.start()
        print("🔄 Поток обработки команд запущен")
    
    def _process_commands(self):
        """Обрабатывает команды из очереди асинхронно."""
        while self.voice_mode_active or True:  # Работает пока интерфейс активен
            try:
                # Получаем команду из очереди с таймаутом
                command_data = self.command_queue.get(timeout=0.1)
                if command_data:
                    text, is_final, callback = command_data
                    
                    # Если это финальный результат, обрабатываем
                    if is_final and callback and text:
                        self.is_processing = True
                        try:
                            # Вызываем callback с текстом
                            callback(text)
                        except Exception as e:
                            print(f"⚠️ Ошибка обработки команды: {e}")
                        finally:
                            self.is_processing = False
                    elif not is_final:
                        # Промежуточный результат - показываем, но не обрабатываем
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Ошибка в потоке обработки команд: {e}")
    
    def speak(self, text: str, async_mode: bool = True) -> None:
        """Озвучивание через Piper TTS."""
        if not text or not text.strip():
            return
            
        if self.tts:
            self.tts.speak(text, async_mode)
        else:
            print(f"🔊 (без TTS) {text}")
    
    def start_voice_mode(self, on_wake: Callable = None, on_transcription: Optional[Callable] = None):
        """
        Запускает постоянное прослушивание.
        
        Args:
            on_wake: Не используется (для совместимости)
            on_transcription: Вызывается при распознавании команды (только финальный результат)
        """
        if not self.is_initialized:
            print("⚠️ Голосовой интерфейс не инициализирован")
            return
        
        if self.voice_mode_active:
            print("🎤 Голосовой режим уже включён")
            return
        
        self.voice_mode_active = True
        
        # Запускаем Vosk в режиме постоянного распознавания
        if self.asr:
            print("🎤 Запуск постоянного прослушивания...")
            
            def on_transcription_callback(text: str, is_final: bool = False):
                """
                Callback из Vosk.
                Если is_final=True - отправляем в очередь на обработку.
                """
                if not self.voice_mode_active:
                    return
                
                if not text or not text.strip():
                    return
                
                # Очищаем текст от лишних пробелов
                clean_text = text.strip()
                
                if is_final:
                    print(f"📝 Финальное распознавание: {clean_text}")
                    # Отправляем в очередь для обработки
                    if on_transcription:
                        self.command_queue.put((clean_text, True, on_transcription))
                else:
                    # Промежуточный результат - просто показываем
                    if len(clean_text) > 0:
                        print(f"📝 ... {clean_text}", end="\r")
            
            self.asr.start_continuous_listening(
                on_transcription=on_transcription_callback
            )
            print("✅ Постоянное прослушивание запущено")
    
    def stop_voice_mode(self):
        """Останавливает голосовой режим."""
        self.voice_mode_active = False
        
        if self.asr:
            self.asr.stop()
        
        if self.tts:
            self.tts.stop()
        
        # Очищаем очередь
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except:
                break
        
        print("🎤 Голосовой режим остановлен")
    
    def stop(self):
        """Полная остановка."""
        self.voice_mode_active = False
        
        if self.asr:
            self.asr.stop()
        
        if self.tts:
            self.tts.stop()
        
        # Очищаем очередь
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except:
                break
        
        print("🎤 Голосовой интерфейс остановлен")
    
    def test_microphone(self) -> bool:
        """Тестирует микрофон."""
        if self.asr:
            try:
                from vosk_interface import test_microphone
                return test_microphone()
            except:
                pass
        return False


# ---------- Тестирование ----------
def test_voice():
    """Тест голосового интерфейса с постоянным прослушиванием."""
    print("\n" + "="*60)
    print("🎤 ТЕСТ ГОЛОСОВОГО ИНТЕРФЕЙСА (постоянное прослушивание)")
    print("="*60)
    
    voice = VoiceInterface()
    
    if not voice.is_initialized:
        print("\n❌ Голосовой интерфейс не инициализирован")
        return
    
    print("\n📌 Говорите в микрофон...")
    print("   Распознанный текст будет появляться в консоли")
    print("   (нажмите Ctrl+C для выхода)\n")
    
    def on_transcription(text):
        print(f"\n🤖 Обработка: {text}")
        # Ответное сообщение для теста
        voice.speak(f"Вы сказали: {text}")
    
    # Запускаем голосовой режим
    voice.start_voice_mode(on_transcription=on_transcription)
    
    try:
        # Держим программу активной
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n⏹️ Остановка...")
        voice.stop()
        print("✅ Тест завершен")


if __name__ == "__main__":
    test_voice()