# voice/__init__.py
"""Голосовой модуль для JARVIS с Vosk."""

from .audio_buffer import AudioBuffer
from .vad import VAD

from .vosk_asr import VoskASR
from .noise_reduction import NoiseReducer
from .silero_tts import SileroTTS
from .emotional_tts import EmotionTTS
from .voice_profile import VoiceProfile
from .enhanced_voice_interface import EnhancedVoiceInterface

__all__ = [
    'AudioBuffer',
    'VAD',
    'VoskASR',
    'NoiseReducer',
    'SileroTTS',
    'EmotionTTS',
    'VoiceProfile',
    'EnhancedVoiceInterface'
]