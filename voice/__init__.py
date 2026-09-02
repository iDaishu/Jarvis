# voice/__init__.py
"""
Голосовой модуль для JARVIS с AEC.

Модуль обеспечивает полный дуплекс голосового общения с:
- Acoustic Echo Cancellation (AEC) для подавления эха
- Распознавание речи через Vosk
- Синтез речи через Silero TTS
- Эмоциональная окраска голоса
- Профиль пользователя
"""

from .aec_voice_interface import AECVoiceInterface, VoiceConfig
from .audio_pipeline import AudioPipeline, PipelineConfig
from .aec_processor import AECProcessor
from .vosk_asr import VoskASR
from .silero_tts import SileroTTS
from .emotional_tts import EmotionTTS
from .voice_profile import VoiceProfile

__version__ = "2.0.0"
__all__ = [
    'AECVoiceInterface',
    'VoiceConfig',
    'AudioPipeline',
    'PipelineConfig',
    'AECProcessor',
    'VoskASR',
    'SileroTTS',
    'EmotionTTS',
    'VoiceProfile'
]