# voice/whisper_asr.py
"""Улучшенный ASR на базе Whisper."""

import threading
import queue
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional, Callable
import time
import json

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

class WhisperASR:
    """Распознавание речи через Whisper (локально)."""
    
    def __init__(self, model_size: str = "small", language: str = "ru"):
        """
        model_size: "tiny", "base", "small", "medium", "large"
        Для русского языка рекомендуется "small" или "medium"
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Установите: pip install openai-whisper")
        
        self.model_size = model_size
        self.language = language
        self.sample_rate = 16000
        self.is_listening = False
        self.audio_buffer = []
        self.audio_queue = queue.Queue(maxsize=100)
        self.transcription_callback = None
        self.model = None
        
        # Параметры распознавания
        self.silence_threshold = 0.01  # Порог тишины
        self.min_speech_duration = 0.3  # Минимальная длительность речи (сек)
        self.max_silence_duration = 0.8  # Максимальная тишина перед распознаванием (сек)
        self.max_buffer_duration = 8.0  # Максимальная длительность буфера (сек)
        
        # Загрузка модели
        self._load_model()
        
    def _load_model(self):
        """Загружает модель Whisper."""
        print(f"📂 Загрузка модели Whisper ({self.model_size})...")
        try:
            self.model = whisper.load_model(self.model_size)
            print(f"✅ Whisper ASR готов (модель: {self.model_size})")
        except Exception as e:
            print(f"❌ Ошибка загрузки Whisper: {e}")
            raise
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback для sounddevice."""
        if status:
            print(f"⚠️ Статус записи: {status}")
        
        if self.is_listening:
            try:
                # Добавляем аудио в очередь
                self.audio_queue.put_nowait(indata.copy())
            except queue.Full:
                # Если очередь переполнена, удаляем старые данные
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(indata.copy())
                except:
                    pass
    
    def _is_silence(self, audio_chunk: np.ndarray) -> bool:
        """Проверяет, является ли аудио тишиной."""
        # Вычисляем максимальную амплитуду
        max_amplitude = np.abs(audio_chunk).max()
        return max_amplitude < self.silence_threshold
    
    def _process_audio(self):
        """Обработка аудио с распознаванием."""
        print("🔄 Поток обработки аудио запущен")
        
        audio_chunks = []
        silence_duration = 0
        speech_duration = 0
        chunk_duration = 0.03  # 30ms на чанк
        
        while self.is_listening:
            try:
                chunk = self.audio_queue.get(timeout=0.05)
                
                # Проверяем размер чанка
                if chunk is None or len(chunk) == 0:
                    continue
                
                # Добавляем в буфер
                audio_chunks.append(chunk)
                
                # Проверяем на тишину
                if self._is_silence(chunk):
                    silence_duration += chunk_duration
                    # Сбрасываем счетчик речи, если тишина долгая
                    if silence_duration > self.max_silence_duration:
                        if speech_duration > self.min_speech_duration and len(audio_chunks) > 10:
                            # Распознаем накопленную речь
                            self._recognize_audio(audio_chunks)
                        # Очищаем буфер
                        audio_chunks = []
                        speech_duration = 0
                        silence_duration = 0
                else:
                    # Есть речь
                    silence_duration = 0
                    speech_duration += chunk_duration
                
                # Проверяем максимальную длительность
                total_duration = len(audio_chunks) * chunk_duration
                if total_duration > self.max_buffer_duration:
                    if speech_duration > self.min_speech_duration:
                        self._recognize_audio(audio_chunks)
                    audio_chunks = []
                    speech_duration = 0
                    silence_duration = 0
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Ошибка обработки: {e}")
                import traceback
                traceback.print_exc()
    
    def _recognize_audio(self, chunks: list):
        """Распознаёт накопленное аудио."""
        if not chunks:
            return
        
        try:
            # Объединяем чанки
            audio = np.concatenate(chunks, axis=0).flatten()
            
            # Проверяем длительность
            duration = len(audio) / self.sample_rate
            if duration < self.min_speech_duration:
                print(f"🔇 Слишком короткое аудио: {duration:.2f} сек")
                return
            
            # Проверяем амплитуду
            max_amplitude = np.abs(audio).max()
            if max_amplitude < self.silence_threshold:
                print("🔇 Слишком тихое аудио")
                return
            
            # Нормализация
            if max_amplitude > 0:
                audio = audio / max_amplitude * 0.9
            
            print(f"🔍 Распознавание аудио ({duration:.2f} сек)...")
            
            # Распознавание
            result = self.model.transcribe(
                audio,
                language=self.language,
                fp16=False,
                task="transcribe",
                verbose=False
            )
            
            text = result.get("text", "").strip()
            
            if text and len(text) > 1:
                print(f"✅ Whisper: {text}")
                if self.transcription_callback:
                    self.transcription_callback(text, is_final=True)
            else:
                print("🔇 Речь не распознана")
                    
        except Exception as e:
            print(f"⚠️ Ошибка Whisper: {e}")
            import traceback
            traceback.print_exc()
    
    def start_listening(self, on_transcription: Optional[Callable] = None):
        """Запускает прослушивание."""
        if self.is_listening:
            print("⚠️ Уже слушаю")
            return
        
        self.transcription_callback = on_transcription
        self.is_listening = True
        self.audio_buffer = []
        
        # Очищаем очередь
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        # Запускаем поток обработки
        self.process_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.process_thread.start()
        
        # Запускаем захват аудио
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.03),  # 30ms блоки
                latency='low'
            )
            self.stream.start()
            print("🎤 Whisper ASR запущен")
            print(f"📊 Параметры: sample_rate={self.sample_rate}, blocksize={int(self.sample_rate * 0.03)}")
            
        except Exception as e:
            print(f"❌ Ошибка запуска: {e}")
            print("💡 Проверьте:")
            print("   1. Подключен ли микрофон")
            print("   2. Есть ли права доступа к микрофону")
            print("   3. Не занят ли микрофон другой программой")
            self.is_listening = False
    
    def stop(self):
        """Останавливает ASR."""
        print("⏹️ Остановка Whisper ASR...")
        self.is_listening = False
        
        # Останавливаем поток
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        # Ждем завершения потока обработки
        if hasattr(self, 'process_thread') and self.process_thread:
            try:
                self.process_thread.join(timeout=1)
            except:
                pass
        
        # Очищаем очередь
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        print("✅ Whisper ASR остановлен")


# Тестирование
def test_whisper():
    """Тест Whisper ASR."""
    print("\n" + "="*60)
    print("🎤 ТЕСТ WHISPER ASR")
    print("="*60)
    
    # Показываем доступные устройства
    print("\n📋 Доступные аудио устройства:")
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  🎤 [{i}] {device['name']}")
    except Exception as e:
        print(f"  ⚠️ Ошибка получения устройств: {e}")
    
    try:
        asr = WhisperASR(model_size="small")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return
    
    print("\n📌 Говорите в микрофон...")
    print("   (нажмите Ctrl+C для выхода)\n")
    
    def on_transcription(text, is_final=False):
        if is_final:
            print(f"✅ ФИНАЛЬНО: {text}")
    
    asr.start_listening(on_transcription)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        asr.stop()
        print("\n✅ Тест завершён.")

if __name__ == "__main__":
    test_whisper()