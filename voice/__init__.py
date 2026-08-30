# voice/__init__.py
"""Голосовой модуль для JARVIS с AEC."""

# Новые компоненты
from .aec_voice_interface import AECVoiceInterface
from .audio_pipeline import AudioPipeline, PipelineConfig
from .aec_processor import AECProcessor
from .vosk_asr import VoskASR
from .silero_tts import SileroTTS
from .emotional_tts import EmotionTTS
from .voice_profile import VoiceProfile

# Старые компоненты (удаляем)
# from .audio_buffer import AudioBuffer
# from .vad import VAD
# from .noise_reduction import NoiseReducer
# from .enhanced_voice_interface import EnhancedVoiceInterface

__all__ = [
    'AECVoiceInterface',
    'AudioPipeline',
    'PipelineConfig',
    'AECProcessor',
    'VoskASR',
    'SileroTTS',
    'EmotionTTS',
    'VoiceProfile'
]