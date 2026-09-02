"""
Настройка логирования для JARVIS AI Assistant.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

class LoggerFactory:
    """Фабрика логгеров для JARVIS."""
    
    _loggers: dict[str, logging.Logger] = {}
    _initialized = False
    
    @classmethod
    def setup(cls, log_dir: Optional[Path] = None) -> None:
        """Настройка глобального логирования."""
        if cls._initialized:
            return
            
        log_dir = log_dir or Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка корневого логгера
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Очищаем существующие хендлеры
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Консольный хендлер
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Файловый хендлер с ротацией
        file_handler = RotatingFileHandler(
            log_dir / "jarvis.log",
            maxBytes=10_000_000,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Отдельный хендлер для ошибок
        error_handler = RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Получить логгер по имени."""
        if not cls._initialized:
            cls.setup()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]