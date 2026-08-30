# voice/__init__.py
"""Голосовой модуль для JARVIS."""

from .enhanced_voice_interface import EnhancedVoiceInterface
from .whisper_asr import WhisperASR
from .silero_tts import SileroTTS
from .noise_reduction import NoiseReducer
from .emotional_tts import EmotionTTS
from .voice_profile import VoiceProfile

__all__ = [
    'EnhancedVoiceInterface',
    'WhisperASR',
    'SileroTTS',
    'NoiseReducer',
    'EmotionTTS',
    'VoiceProfile'
]