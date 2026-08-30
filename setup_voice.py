# setup_voice.py
"""Скрипт для установки голосовых компонентов."""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Устанавливает зависимости."""
    print("📦 Установка зависимостей...")
    
    requirements = [
        # Основные
        "requests>=2.31.0",
        "pyyaml>=6.0",
        
        # Аудио
        "sounddevice>=0.4.6",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "webrtcvad>=2.0.10",
        
        # ASR
        "openai-whisper>=20231117",
        "vosk>=0.3.45",
        
        # TTS
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "onnxruntime>=1.15.0",
        "pyttsx3>=2.90",
        
        # Шумоподавление
        "noisereduce>=3.0.0",
        
        # Другое
        "setuptools>=69.0.0",
        "wheel>=0.42.0"
    ]
    
    for req in requirements:
        try:
            print(f"   Установка {req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except Exception as e:
            print(f"   ⚠️ Ошибка установки {req}: {e}")

def create_directories():
    """Создаёт необходимые директории."""
    print("\n📁 Создание директорий...")
    
    dirs = [
        "memory",
        "models",
        "voice",
        "screenshots",
        "training"
    ]
    
    for d in dirs:
        path = Path(d)
        path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {path}")

def download_whisper_model():
    """Скачивает модель Whisper."""
    print("\n📂 Скачивание модели Whisper...")
    
    try:
        import whisper
        print("   Загрузка модели small...")
        model = whisper.load_model("small")
        print("   ✅ Модель Whisper загружена")
    except Exception as e:
        print(f"   ⚠️ Ошибка загрузки Whisper: {e}")

def main():
    print("="*60)
    print("🎤 УСТАНОВКА ГОЛОСОВЫХ КОМПОНЕНТОВ JARVIS")
    print("="*60)
    
    # Установка зависимостей
    install_requirements()
    
    # Создание директорий
    create_directories()
    
    # Скачивание модели Whisper
    download_whisper_model()
    
    print("\n" + "="*60)
    print("✅ Установка завершена!")
    print("="*60)
    print("\n📌 Для запуска:")
    print("   python agent_core.py")
    print("\n📌 Для теста голоса:")
    print("   python voice/enhanced_voice_interface.py")

if __name__ == "__main__":
    main()