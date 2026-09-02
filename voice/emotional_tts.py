# voice/emotional_tts.py
"""
TTS с эмоциональной окраской.

Анализирует текст и определяет подходящую эмоциональную окраску для синтеза речи.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils.logger import LoggerFactory

logger = LoggerFactory.get_logger("emotional_tts")


class EmotionTTS:
    """
    Добавляет эмоциональную окраску в синтез речи.
    
    Особенности:
    - Анализ текста для определения эмоций
    - Настройка параметров речи под эмоцию
    - Очистка текста от эмодзи и спецсимволов
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Инициализация эмоционального TTS.
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self._emotion_markers = {
            "радость": ["😊", "😄", "🥳", "👏", "🎉", "🤗", "🌟"],
            "удивление": ["😮", "😲", "🤯", "💥", "🎯", "💫"],
            "грусть": ["😢", "😔", "💔", "🥺", "😥", "😪"],
            "злость": ["😠", "😡", "💢", "🔥", "⚡", "💥"],
            "восхищение": ["😍", "🌟", "✨", "💫", "👌", "💯"],
            "смех": ["😂", "🤣", "😅", "😆", "🤪", "😹"],
            "задумчивость": ["🤔", "🧐", "💭", "📝", "🤓", "🧠"],
            "поддержка": ["💪", "🤗", "🌟", "🌈", "❤️", "🤝"],
            "ирония": ["😏", "🙃", "😉", "😜", "🤭"],
        }
        
        self._emotion_keywords = {
            "радость": ["отлично", "прекрасно", "замечательно", "радость", "счастье", 
                       "ура", "класс", "супер", "здорово", "великолепно", "чудесно"],
            "удивление": ["невероятно", "удивительно", "неожиданно", "вот это да", 
                        "ничего себе", "ого", "вау", "потрясающе", "фантастика"],
            "грусть": ["жаль", "грустно", "печально", "сожалею", "к сожалению", 
                      "обидно", "неудачно", "плохо", "сложно", "трудно"],
            "злость": ["возмутительно", "недопустимо", "ужасно", "отвратительно", 
                      "безобразно", "скандально", "возмущён"],
            "восхищение": ["великолепно", "восхитительно", "гениально", "превосходно", 
                          "божественно", "идеально", "мастерски", "шедеврально"],
            "смех": ["смешно", "забавно", "шутка", "прикол", "юмор", "засмеяться"],
            "задумчивость": ["подумать", "возможно", "наверное", "вероятно", 
                           "кажется", "пожалуй", "видимо"],
            "поддержка": ["не волнуйтесь", "всё будет хорошо", "вы справитесь", 
                        "я помогу", "поддерживаю", "вместе справимся"],
            "ирония": ["конечно", "разумеется", "естественно", "очевидно", 
                      "безусловно", "несомненно"],
        }
        
        self._emotion_params = {
            "радость": {"speed": 1.15, "pitch": 1.1, "energy": 1.15},
            "удивление": {"speed": 1.1, "pitch": 1.2, "energy": 1.1},
            "грусть": {"speed": 0.85, "pitch": 0.9, "energy": 0.85},
            "злость": {"speed": 1.1, "pitch": 1.05, "energy": 1.2},
            "восхищение": {"speed": 0.95, "pitch": 1.15, "energy": 1.1},
            "смех": {"speed": 1.05, "pitch": 1.1, "energy": 1.05},
            "задумчивость": {"speed": 0.9, "pitch": 0.95, "energy": 0.9},
            "поддержка": {"speed": 0.95, "pitch": 1.05, "energy": 1.05},
            "ирония": {"speed": 0.95, "pitch": 1.05, "energy": 0.95},
            "нейтральная": {"speed": 1.0, "pitch": 1.0, "energy": 1.0},
        }
        
        # Загрузка пользовательской конфигурации
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom = json.load(f)
                    self._emotion_markers.update(custom.get('markers', {}))
                    self._emotion_keywords.update(custom.get('keywords', {}))
                    self._emotion_params.update(custom.get('params', {}))
                logger.info(f"Загружена конфигурация эмоций: {config_path}")
            except Exception as e:
                logger.warning(f"Ошибка загрузки конфигурации эмоций: {e}")
    
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """
        Анализ эмоциональной окраски текста.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Dict[str, float]: Словарь эмоций с весами
        """
        emotions = {emotion: 0.0 for emotion in self._emotion_markers}
        emotions["нейтральная"] = 0.0
        text_lower = text.lower()
        
        # Проверка эмодзи
        for emotion, markers in self._emotion_markers.items():
            for marker in markers:
                if marker in text:
                    emotions[emotion] += 0.6
                    break
        
        # Проверка ключевых слов
        for emotion, keywords in self._emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotions[emotion] += 0.3
                    break
        
        # Проверка пунктуации
        if '!' in text:
            emotions["удивление"] += 0.1
            emotions["восхищение"] += 0.1
        
        if '?' in text:
            emotions["удивление"] += 0.15
        
        if '...' in text:
            emotions["задумчивость"] += 0.25
        
        # Если эмоций нет, добавляем нейтральную
        if all(v < 0.1 for v in emotions.values()):
            emotions["нейтральная"] = 1.0
        
        # Нормализация
        total = sum(emotions.values())
        if total > 0:
            for emotion in emotions:
                emotions[emotion] = min(1.0, emotions[emotion] / total * 2)
        
        return emotions
    
    def get_dominant_emotion(self, emotions: Dict[str, float]) -> str:
        """
        Определение доминирующей эмоции.
        
        Args:
            emotions: Словарь эмоций с весами
            
        Returns:
            str: Название доминирующей эмоции
        """
        if not emotions:
            return "нейтральная"
        
        # Убираем нейтральную для определения максимума
        filtered = {k: v for k, v in emotions.items() if k != "нейтральная"}
        if not filtered:
            return "нейтральная"
        
        max_emotion = max(filtered, key=filtered.get)
        if filtered.get(max_emotion, 0) < 0.1:
            return "нейтральная"
        
        return max_emotion
    
    def get_speaking_params(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """
        Определение параметров речи на основе эмоций.
        
        Args:
            emotions: Словарь эмоций с весами
            
        Returns:
            Dict[str, float]: Параметры речи
        """
        params = {"speed": 1.0, "pitch": 1.0, "energy": 1.0}
        
        dominant = self.get_dominant_emotion(emotions)
        intensity = emotions.get(dominant, 0.0)
        
        if dominant in self._emotion_params and intensity > 0.2:
            base_params = self._emotion_params[dominant]
            for key in params:
                delta = (base_params.get(key, 1.0) - 1.0) * intensity
                params[key] = 1.0 + delta
                params[key] = max(0.7, min(1.3, params[key]))
        
        # Округление
        for key in params:
            params[key] = round(params[key], 2)
        
        return params
    
    def enhance_text(self, text: str) -> Tuple[str, Dict[str, float], str]:
        """
        Улучшение текста для синтеза.
        
        Args:
            text: Исходный текст
            
        Returns:
            Tuple[str, Dict[str, float], str]: 
                (Очищенный текст, параметры речи, эмоция)
        """
        # Анализ эмоций
        emotions = self.analyze_emotion(text)
        params = self.get_speaking_params(emotions)
        emotion = self.get_dominant_emotion(emotions)
        
        # Удаление эмодзи
        clean_text = text
        for markers in self._emotion_markers.values():
            for marker in markers:
                clean_text = clean_text.replace(marker, "")
        
        # Замена цифр на слова (для лучшего TTS)
        clean_text = self._replace_numbers(clean_text)
        
        # Удаление лишних символов
        clean_text = re.sub(r'[^\w\s.,!?-]', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        clean_text = re.sub(r'([.!?])', r'\1 ', clean_text)
        
        # Удаляем лишние пробелы
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text, params, emotion
    
    def _replace_numbers(self, text: str) -> str:
        """
        Замена чисел словами.
        
        Args:
            text: Исходный текст
            
        Returns:
            str: Текст с заменёнными числами
        """
        digit_map = {
            '0': 'ноль',
            '1': 'один',
            '2': 'два',
            '3': 'три',
            '4': 'четыре',
            '5': 'пять',
            '6': 'шесть',
            '7': 'семь',
            '8': 'восемь',
            '9': 'девять',
        }
        
        def replace_number(match: re.Match) -> str:
            num = match.group(1)
            if len(num) == 1:
                return digit_map.get(num, num)
            # Для длинных чисел заменяем по цифрам
            return ' '.join(digit_map.get(d, d) for d in num)
        
        return re.sub(r'\b(\d+)\b', replace_number, text)