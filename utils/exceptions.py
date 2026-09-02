"""
Кастомные исключения для JARVIS AI Assistant.
"""

class JARVISError(Exception):
    """Базовое исключение для JARVIS."""
    pass

class ModelLoadError(JARVISError):
    """Ошибка загрузки модели."""
    pass

class AudioError(JARVISError):
    """Ошибка аудио-системы."""
    pass

class ConfigError(JARVISError):
    """Ошибка конфигурации."""
    pass

class MemoryError(JARVISError):
    """Ошибка памяти."""
    pass

class ToolExecutionError(JARVISError):
    """Ошибка выполнения инструмента."""
    pass

class VoiceError(JARVISError):
    """Ошибка голосовой системы."""
    pass