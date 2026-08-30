"""TTS с эмоциональной окраской."""

import re
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class EmotionTTS:
    """Добавляет эмоциональную окраску в синтез речи."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.emotion_markers = {
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
        
        self.emotion_keywords = {
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
        
        self.emotion_params = {
            "радость": {"speed": 1.15, "pitch": 1.1, "energy": 1.15},
            "удивление": {"speed": 1.1, "pitch": 1.2, "energy": 1.1},
            "грусть": {"speed": 0.85, "pitch": 0.9, "energy": 0.85},
            "злость": {"speed": 1.1, "pitch": 1.05, "energy": 1.2},
            "восхищение": {"speed": 0.95, "pitch": 1.15, "energy": 1.1},
            "смех": {"speed": 1.05, "pitch": 1.1, "energy": 1.05},
            "задумчивость": {"speed": 0.9, "pitch": 0.95, "energy": 0.9},
            "поддержка": {"speed": 0.95, "pitch": 1.05, "energy": 1.05},
            "ирония": {"speed": 0.95, "pitch": 1.05, "energy": 0.95},
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    custom = json.load(f)
                    self.emotion_markers.update(custom.get('markers', {}))
                    self.emotion_keywords.update(custom.get('keywords', {}))
                    self.emotion_params.update(custom.get('params', {}))
            except:
                pass
    
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """Анализирует эмоциональную окраску текста."""
        emotions = {emotion: 0.0 for emotion in self.emotion_markers}
        
        # Проверяем эмодзи
        for emotion, markers in self.emotion_markers.items():
            for marker in markers:
                if marker in text:
                    emotions[emotion] += 0.6
                    text = text.replace(marker, "")
        
        # Проверяем ключевые слова
        lower_text = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in lower_text:
                    emotions[emotion] += 0.3
        
        # Проверяем восклицательные знаки
        if '!' in text:
            emotions["удивление"] += 0.1
            emotions["восхищение"] += 0.1
        
        # Проверяем вопросительные знаки
        if '?' in text:
            emotions["удивление"] += 0.2
        
        # Проверяем многоточие
        if '...' in text:
            emotions["задумчивость"] += 0.3
        
        # Нормализация
        total = sum(emotions.values())
        if total > 0:
            for emotion in emotions:
                emotions[emotion] = min(1.0, emotions[emotion] / total * 2)
        
        return emotions
    
    def get_dominant_emotion(self, emotions: Dict[str, float]) -> str:
        """Возвращает доминирующую эмоцию."""
        if not emotions:
            return "нейтральная"
        
        max_emotion = max(emotions, key=emotions.get)
        if emotions[max_emotion] < 0.1:
            return "нейтральная"
        
        return max_emotion
    
    def get_speaking_params(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """Определяет параметры речи на основе эмоций."""
        params = {"speed": 1.0, "pitch": 1.0, "energy": 1.0}
        
        dominant = self.get_dominant_emotion(emotions)
        intensity = emotions.get(dominant, 0)
        
        if dominant in self.emotion_params and intensity > 0.2:
            base_params = self.emotion_params[dominant]
            for key in params:
                delta = (base_params[key] - 1.0) * intensity
                params[key] = 1.0 + delta
                params[key] = max(0.7, min(1.3, params[key]))
        
        params["speed"] = round(params["speed"], 2)
        params["pitch"] = round(params["pitch"], 2)
        params["energy"] = round(params["energy"], 2)
        
        return params
    
    def enhance_text(self, text: str) -> Tuple[str, Dict[str, float], str]:
        """Улучшает текст для синтеза."""
        emotions = self.analyze_emotion(text)
        params = self.get_speaking_params(emotions)
        emotion = self.get_dominant_emotion(emotions)
        
        # Удаляем эмодзи для синтеза
        clean_text = text
        for markers in self.emotion_markers.values():
            for marker in markers:
                clean_text = clean_text.replace(marker, "")
        
        # Заменяем проблемные символы для Silero
        # Цифры нужно заменять словами или убирать
        clean_text = re.sub(r'(\d+)', lambda m: self._number_to_words(m.group(1)), clean_text)
        
        # Убираем специальные символы
        clean_text = re.sub(r'[^\w\s.,!?-]', ' ', clean_text)
        
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        clean_text = re.sub(r'([.!?])', r'\1 ', clean_text)
        
        return clean_text, params, emotion

    def _number_to_words(self, num: str) -> str:
        """Преобразует число в слова."""
        # Простое преобразование для цифр
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
        return ' '.join(digit_map.get(d, d) for d in num)