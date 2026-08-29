"""
vosk_interface.py
Голосовой интерфейс с Vosk - БЫСТРОЕ РАСПОЗНАВАНИЕ
Оптимизирован для минимальной задержки
"""

import json
import queue
import threading
import time
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, List

import numpy as np
import sounddevice as sd
import webrtcvad
from vosk import Model, KaldiRecognizer

# ---------- Базовая директория ----------
BASE_DIR = Path(__file__).resolve().parent

def find_vosk_model() -> Optional[Path]:
    """Автоматически ищет модель Vosk."""
    possible_paths = [
        BASE_DIR / "models" / "vosk-model-small-ru-0.22",
        BASE_DIR / "models" / "vosk-model-ru-0.22",
        Path.home() / "models" / "vosk-model-small-ru-0.22",
        Path.home() / "models" / "vosk-model-ru-0.22",
        BASE_DIR / "vosk-model-small-ru-0.22",
        BASE_DIR / "vosk-model-ru-0.22",
        Path("C:/models/vosk-model-small-ru-0.22"),
        Path("D:/models/vosk-model-small-ru-0.22"),
    ]
    
    for path in possible_paths:
        if path.exists():
            if (path / "am").exists() and (path / "conf").exists():
                print(f"✅ Найдена модель Vosk: {path}")
                return path
            elif (path / "final.mdl").exists():
                print(f"✅ Найдена модель Vosk: {path}")
                return path
    
    models_dir = BASE_DIR / "models"
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir() and "vosk-model" in item.name.lower():
                print(f"✅ Найдена потенциальная модель Vosk: {item}")
                return item
    
    return None

@dataclass
class VoiceConfig:
    """Конфигурация голосового интерфейса."""
    model_path: Optional[str] = None
    sample_rate: int = 16000
    chunk_duration: float = 0.02  # Уменьшено для меньшей задержки
    wake_words: List[str] = field(default_factory=list)
    vad_mode: int = 1  # Менее агрессивный VAD для быстрого реагирования
    language: str = "ru-RU"
    # Оптимизированные параметры для быстрого распознавания
    min_speech_duration: float = 0.3  # Минимальная длительность речи
    max_silence_duration: float = 0.6  # Максимальная тишина перед завершением
    partial_results: bool = True  # Показывать частичные результаты


class VoskInterface:
    """Голосовой интерфейс с быстрым распознаванием."""

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self.config.wake_words = self.config.wake_words or []

        # Поиск модели
        model_path = None
        
        if self.config.model_path:
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                print(f"⚠️ Модель не найдена: {model_path}")
                model_path = None
        
        if not model_path:
            model_path = find_vosk_model()
        
        if not model_path:
            print("❌ Модель Vosk не найдена!")
            print("\n📥 Скачайте модель с официального сайта:")
            print("   https://alphacephei.com/vosk/models")
            print(f"\n📂 Распакуйте архив в папку: {BASE_DIR / 'models'}")
            raise FileNotFoundError("Модель Vosk не найдена")

        # Загрузка модели
        print(f"📂 Загрузка модели: {model_path}")
        try:
            self.model = Model(str(model_path))
            self.recognizer = KaldiRecognizer(self.model, self.config.sample_rate)
            self.recognizer.SetWords(True)
            print("✅ Модель Vosk загружена успешно")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise

        # VAD с оптимизированными настройками
        self.vad = webrtcvad.Vad(self.config.vad_mode)

        # Аудио поток
        self.audio_queue = queue.Queue(maxsize=100)  # Ограниченная очередь
        self.is_listening = False
        self.stream = None
        self.transcription_callback = None
        self.audio_thread = None

        # Буфер для накопления аудио
        self.audio_buffer = b""
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.last_speech_time = 0
        
        # Для быстрого распознавания
        self.frame_size = int(self.config.sample_rate * self.config.chunk_duration)
        self.chunk_size = self.frame_size * 2
        self.min_speech_frames = int(self.config.min_speech_duration / self.config.chunk_duration)
        self.max_silence_frames = int(self.config.max_silence_duration / self.config.chunk_duration)
        
        # Для предотвращения дублирования
        self.last_result = ""
        self.last_result_time = 0
        self.result_cooldown = 0.5  # Защита от дублирования
        
        # Для отладки
        self.frame_count = 0
        self.speech_count = 0
        
        print(f"🎤 Vosk инициализирован (быстрый режим)")
        print(f"   Чанк: {self.config.chunk_duration*1000:.0f}мс, VAD: {self.config.vad_mode}")
        print(f"   Мин. речь: {self.config.min_speech_duration}с, Макс. тишина: {self.config.max_silence_duration}с")

    def _audio_callback(self, indata, frames, time, status):
        """Callback для sounddevice."""
        if status:
            if status == sd.CallbackFlags.input_overflow:
                pass  # Игнорируем переполнение
            else:
                print(f"⚠️ Аудио статус: {status}")
                
        if self.is_listening:
            try:
                int16_data = (indata * 32767).astype(np.int16).tobytes()
                # Неблокирующая вставка в очередь
                try:
                    self.audio_queue.put_nowait(int16_data)
                except queue.Full:
                    pass  # Пропускаем если очередь переполнена
            except Exception as e:
                pass

    def _process_audio_stream(self):
        """
        Быстрая обработка аудио потока с минимальной задержкой.
        """
        buffer = b""
        speech_buffer = b""
        is_speech_active = False
        speech_frames = 0
        silence_frames = 0
        
        # Для частичных результатов
        partial_text = ""
        last_partial_time = 0
        
        while self.is_listening:
            try:
                audio_data = self.audio_queue.get(timeout=0.05)
                buffer += audio_data

                while len(buffer) >= self.chunk_size:
                    frame = buffer[:self.chunk_size]
                    buffer = buffer[self.chunk_size:]
                    self.frame_count += 1

                    try:
                        is_speech = self.vad.is_speech(frame, self.config.sample_rate)
                        
                        if is_speech:
                            self.speech_count += 1
                            
                            if not is_speech_active:
                                # Начало речи
                                is_speech_active = True
                                speech_buffer = b""
                                speech_frames = 0
                                silence_frames = 0
                                self.is_speaking = True
                                self.last_speech_time = time.time()
                            
                            speech_buffer += frame
                            speech_frames += 1
                            silence_frames = 0
                            
                            # Быстрое распознавание во время речи (частичные результаты)
                            if self.config.partial_results and speech_frames % 5 == 0:
                                try:
                                    # Используем частичный результат
                                    partial = json.loads(self.recognizer.PartialResult())
                                    partial_text = partial.get("partial", "").strip()
                                    if partial_text and len(partial_text) > 1:
                                        current_time = time.time()
                                        if current_time - last_partial_time > 0.2:  # Не чаще 5 раз в секунду
                                            last_partial_time = current_time
                                            if self.transcription_callback:
                                                self.transcription_callback(partial_text, is_final=False)
                                except Exception as e:
                                    pass
                                    
                        else:
                            # Тишина
                            if is_speech_active:
                                silence_frames += 1
                                
                                # Проверяем, достаточно ли говорили
                                if speech_frames >= self.min_speech_frames:
                                    # Если тишина превышает лимит
                                    if silence_frames > self.max_silence_frames:
                                        # Отправляем на распознавание
                                        if speech_buffer:
                                            try:
                                                # Добавляем оставшиеся фреймы
                                                for i in range(0, len(speech_buffer), self.chunk_size):
                                                    chunk = speech_buffer[i:i+self.chunk_size]
                                                    if len(chunk) == self.chunk_size:
                                                        self.recognizer.AcceptWaveform(chunk)
                                                
                                                # Получаем финальный результат
                                                result = json.loads(self.recognizer.Result())
                                                text = result.get("text", "").strip()
                                                
                                                if text:
                                                    current_time = time.time()
                                                    # Защита от дублирования
                                                    if (text != self.last_result or 
                                                        current_time - self.last_result_time > self.result_cooldown):
                                                        self.last_result = text
                                                        self.last_result_time = current_time
                                                        
                                                        print(f"\n📝 Распознано: {text}")
                                                        if self.transcription_callback:
                                                            self.transcription_callback(text, is_final=True)
                                                
                                                self.recognizer.Reset()
                                                
                                            except Exception as e:
                                                print(f"⚠️ Ошибка распознавания: {e}")
                                        
                                        # Сбрасываем состояние
                                        is_speech_active = False
                                        speech_buffer = b""
                                        speech_frames = 0
                                        silence_frames = 0
                                        self.is_speaking = False
                                        partial_text = ""
                                
                                # Если говорили слишком мало
                                elif silence_frames > 5:  # ~0.1 секунды
                                    is_speech_active = False
                                    speech_buffer = b""
                                    speech_frames = 0
                                    silence_frames = 0
                                    self.is_speaking = False
                                    self.recognizer.Reset()
                                    partial_text = ""

                    except Exception as e:
                        # Игнорируем ошибки VAD
                        pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Ошибка обработки: {e}")

    def start_continuous_listening(self, on_transcription: Optional[Callable] = None):
        """
        Запускает постоянное прослушивание с быстрым распознаванием.
        """
        self.transcription_callback = on_transcription
        self.is_listening = True
        self.last_result = ""
        self.last_result_time = 0

        # Запускаем поток обработки
        self.audio_thread = threading.Thread(target=self._process_audio_stream, daemon=True)
        self.audio_thread.start()

        # Настройка аудио устройства
        try:
            devices = sd.query_devices()
            default_device = sd.default.device[0]
            
            if default_device is None:
                # Ищем первое устройство ввода
                for idx, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0:
                        default_device = idx
                        break
            
            print(f"🎤 Использую устройство: {default_device}")
            
        except Exception as e:
            print(f"⚠️ Ошибка устройств: {e}")
            default_device = None

        # Запускаем захват
        try:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=self.frame_size,
                device=default_device,
                latency='low',  # Минимальная задержка
            )
            self.stream.start()
            print(f"\n🎤 Постоянное прослушивание включено (быстрый режим)...")
            print(f"   Задержка: ~{self.config.chunk_duration*1000:.0f}мс")
            print("   Говорите в микрофон\n")
            
        except Exception as e:
            print(f"❌ Ошибка запуска: {e}")
            self.is_listening = False

    def stop(self):
        """Останавливает все процессы."""
        self.is_listening = False
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        if hasattr(self, 'audio_thread') and self.audio_thread:
            self.audio_thread.join(timeout=1)
        
        # Очищаем очередь
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        print(f"✅ Vosk остановлен (обработано {self.frame_count} фреймов)")

    def listen_for_command(self, timeout: float = 10.0) -> str:
        """Для обратной совместимости."""
        return ""


def test_microphone():
    """Тест микрофона."""
    print("\n🎤 Тест микрофона...")
    
    duration = 2
    sample_rate = 16000
    
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, 
                          channels=1, dtype=np.float32)
        sd.wait()
        
        max_amplitude = np.max(np.abs(recording))
        rms = np.sqrt(np.mean(recording**2))
        
        print(f"📊 Пиковая амплитуда: {max_amplitude:.3f}")
        print(f"📊 RMS: {rms:.3f}")
        
        if max_amplitude > 0.01:
            print("✅ Микрофон работает")
            return True
        else:
            print("❌ Микрофон не слышен")
            return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False


def test_vosk():
    """Тест быстрого распознавания."""
    print("\n" + "="*60)
    print("🎤 ТЕСТ БЫСТРОГО РАСПОЗНАВАНИЯ")
    print("="*60)
    
    if not test_microphone():
        print("\n❌ Микрофон не работает")
        return
    
    try:
        vosk = VoskInterface()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return

    print("\n📌 Говорите в микрофон...")
    print("   Распознавание будет происходить с минимальной задержкой")
    print("   (нажмите Ctrl+C для выхода)\n")
    
    def on_transcription(text, is_final=False):
        if is_final:
            print(f"\n✅ ФИНАЛЬНО: {text}")
        else:
            print(f"📝 ... {text}", end="\r")

    vosk.start_continuous_listening(on_transcription)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        vosk.stop()
        print("\n✅ Тест завершён.")


if __name__ == "__main__":
    # Создаем папку models если её нет
    models_dir = BASE_DIR / "models"
    if not models_dir.exists():
        models_dir.mkdir(parents=True)
    
    test_vosk()