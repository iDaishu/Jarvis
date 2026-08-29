"""
piper_tts.py
TTS на базе Piper для Windows с fallback на pyttsx3
"""

import json
import subprocess
import threading
import queue
import time
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Fallback TTS
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

BASE_DIR = Path(__file__).resolve().parent

@dataclass
class PiperConfig:
    model_path: str = str(BASE_DIR / "models" / "piper" / "model.onnx")
    config_path: str = str(BASE_DIR / "models" / "piper" / "model.onnx.json")
    sample_rate: int = 22050
    noise_scale: float = 0.667
    length_scale: float = 1.0
    noise_w: float = 0.8


class PiperTTS:
    """
    Синтез речи на базе Piper TTS с fallback на pyttsx3.
    """
    
    def __init__(self, config: Optional[PiperConfig] = None):
        self.config = config or PiperConfig()
        self.session = None
        self.phoneme_id_map = None
        self.espeak_voice = None
        self.is_ready = False
        self.use_pyttsx3 = False
        self.tts_engine = None
        
        # TTS очередь для асинхронной работы
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.is_speaking = False
        self.running = True
        
        # Пытаемся инициализировать Piper
        self._init_model()
        
        # Если Piper не готов, используем pyttsx3
        if not self.is_ready:
            self._init_pyttsx3()
        
        # Запускаем поток для озвучивания
        self._start_tts_thread()
    
    def _init_model(self):
        """Инициализация ONNX модели."""
        if ort is None:
            print("ℹ️ ONNX Runtime не установлен, буду использовать pyttsx3")
            return
        
        model_path = Path(self.config.model_path)
        config_path = Path(self.config.config_path)
        
        if not model_path.exists():
            print(f"ℹ️ Модель Piper не найдена: {model_path}")
            print("   Буду использовать pyttsx3 как запасной вариант")
            return
        
        if not config_path.exists():
            print(f"ℹ️ Конфиг Piper не найден: {config_path}")
            print("   Буду использовать pyttsx3 как запасной вариант")
            return
        
        try:
            # Загружаем конфиг
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.phoneme_id_map = config["phoneme_id_map"]
            self.espeak_voice = config["espeak"]["voice"]
            
            # Загружаем модель
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(
                str(model_path),
                providers=providers
            )
            
            self.is_ready = True
            print(f"✅ Piper TTS готов. Голос: {self.espeak_voice}")
            
        except Exception as e:
            print(f"ℹ️ Ошибка загрузки Piper: {e}")
            print("   Буду использовать pyttsx3 как запасной вариант")
            self.is_ready = False
    
    def _init_pyttsx3(self):
        """Инициализация pyttsx3 как fallback."""
        if pyttsx3 is None:
            print("⚠️ pyttsx3 не установлен. Установите: pip install pyttsx3")
            return
        
        try:
            self.tts_engine = pyttsx3.init()
            
            # Настройка голоса (русский)
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'ru' in voice.languages or 'Russian' in voice.name:
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            
            # Настройка скорости
            self.tts_engine.setProperty('rate', 170)
            
            self.use_pyttsx3 = True
            print("✅ pyttsx3 TTS готов (русский голос)")
            
        except Exception as e:
            print(f"⚠️ Ошибка pyttsx3: {e}")
    
    def _phonemize(self, text: str) -> list:
        """Конвертирует текст в фонемы через espeak-ng."""
        if not self.espeak_voice:
            return []
        
        try:
            # Проверяем наличие espeak-ng
            result = subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                print("⚠️ espeak-ng не установлен!")
                print("📥 Установите: winget install espeak-ng")
                return []
            
            # Фонемизация
            result = subprocess.run(
                ["espeak-ng", "-v", self.espeak_voice, "-q", "--ipa=2", "-x", text],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                return []
            
            lines = result.stdout.strip().split('\n')
            return [list(line.replace("_", " ").strip()) for line in lines if line.strip()]
            
        except FileNotFoundError:
            print("⚠️ espeak-ng не установлен!")
            print("📥 Установите: winget install espeak-ng")
            return []
        except Exception as e:
            print(f"⚠️ Ошибка фонемизации: {e}")
            return []
    
    def _to_ids(self, phonemes: list) -> list:
        """Конвертирует фонемы в ID для модели."""
        if not self.phoneme_id_map:
            return []
        
        ids = [self.phoneme_id_map["^"][0], self.phoneme_id_map["_"][0]]
        
        for p in phonemes:
            if p in self.phoneme_id_map:
                ids.extend(self.phoneme_id_map[p])
            ids.append(self.phoneme_id_map["_"][0])
        
        ids.append(self.phoneme_id_map["$"][0])
        return ids
    
    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """
        Синтезирует речь из текста через Piper.
        """
        if not self.is_ready:
            return None
        
        if not text.strip():
            return None
        
        # Фонемизируем текст
        sentences = self._phonemize(text)
        if not sentences:
            print("⚠️ Не удалось сфонемизировать текст, использую pyttsx3")
            return None
        
        audio_chunks = []
        
        for sentence in sentences:
            ids = self._to_ids(sentence)
            
            if len(ids) < 3:
                continue
            
            try:
                audio = self.session.run(
                    None, {
                        "input": np.array([ids], dtype=np.int64),
                        "input_lengths": np.array([len(ids)], dtype=np.int64),
                        "scales": np.array([
                            self.config.noise_scale,
                            self.config.length_scale,
                            self.config.noise_w
                        ], dtype=np.float32)
                    }
                )[0]
                
                audio_chunks.append(audio.squeeze())
                
            except Exception as e:
                print(f"⚠️ Ошибка синтеза: {e}")
                continue
        
        if audio_chunks:
            return np.concatenate(audio_chunks).astype(np.float32)
        
        return None
    
    def _speak_with_pyttsx3(self, text: str):
        """Синтез через pyttsx3."""
        if not self.tts_engine:
            return
        
        try:
            self.is_speaking = True
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"⚠️ Ошибка pyttsx3: {e}")
        finally:
            self.is_speaking = False
    
    def _speak_sync(self, text: str):
        """Синхронное озвучивание."""
        # Сначала пробуем Piper
        if self.is_ready:
            audio = self.synthesize(text)
            if audio is not None:
                try:
                    import sounddevice as sd
                    self.is_speaking = True
                    sd.play(audio, self.config.sample_rate)
                    sd.wait()
                    self.is_speaking = False
                    return
                except Exception as e:
                    print(f"⚠️ Ошибка воспроизведения Piper: {e}")
        
        # Fallback на pyttsx3
        if self.use_pyttsx3:
            self._speak_with_pyttsx3(text)
        else:
            print(f"🔊 (без TTS) {text}")
    
    def speak(self, text: str, async_mode: bool = True) -> None:
        """
        Озвучивает текст.
        """
        if not text or not text.strip():
            return
        
        clean_text = text.replace('*', '').replace('_', '').strip()
        
        if async_mode:
            self.tts_queue.put(clean_text)
        else:
            self._speak_sync(clean_text)
    
    def _tts_worker(self):
        """Поток для асинхронного озвучивания."""
        while self.running:
            try:
                text = self.tts_queue.get(timeout=0.1)
                if text:
                    self._speak_sync(text)
                    self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Ошибка в TTS потоке: {e}")
    
    def _start_tts_thread(self):
        """Запускает поток для асинхронного TTS."""
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        print("🔄 TTS поток запущен")
    
    def stop(self):
        """Останавливает TTS."""
        self.running = False
        if self.tts_thread:
            self.tts_thread.join(timeout=1)
        print("⏹️ TTS остановлен")


# ---------- Тестирование ----------
def test_piper():
    """Тест Piper TTS."""
    print("\n" + "="*60)
    print("🎤 ТЕСТ TTS")
    print("="*60)
    
    tts = PiperTTS()
    
    print("\n🔊 Тест синтеза...")
    tts.speak("Привет! Я Джарвис, ваш голосовой ассистент.")
    
    time.sleep(2)
    tts.speak("Это тест синтеза речи.", async_mode=True)
    
    time.sleep(3)
    tts.stop()
    print("\n✅ Тест завершен")


if __name__ == "__main__":
    test_piper()