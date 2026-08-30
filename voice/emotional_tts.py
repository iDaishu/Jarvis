# voice/emotional_tts.py
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
        
        # Загружаем кастомные настройки
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
        # Базовые параметры
        params = {
            "speed": 1.0,
            "pitch": 1.0,
            "energy": 1.0,
        }
        
        # Находим доминирующую эмоцию
        dominant = self.get_dominant_emotion(emotions)
        intensity = emotions.get(dominant, 0)
        
        if dominant in self.emotion_params and intensity > 0.2:
            base_params = self.emotion_params[dominant]
            # Применяем параметры с учётом интенсивности
            for key in params:
                delta = (base_params[key] - 1.0) * intensity
                params[key] = 1.0 + delta
                # Ограничиваем
                params[key] = max(0.7, min(1.3, params[key]))
        
        # Добавляем небольшие вариации
        params["speed"] = round(params["speed"], 2)
        params["pitch"] = round(params["pitch"], 2)
        params["energy"] = round(params["energy"], 2)
        
        return params
    
    def enhance_text(self, text: str) -> Tuple[str, Dict[str, float]]:
        """Улучшает текст для синтеза."""
        # Анализируем эмоции
        emotions = self.analyze_emotion(text)
        params = self.get_speaking_params(emotions)
        emotion = self.get_dominant_emotion(emotions)
        
        # Удаляем эмодзи для синтеза
        clean_text = text
        for markers in self.emotion_markers.values():
            for marker in markers:
                clean_text = clean_text.replace(marker, "")
        
        # Убираем лишние пробелы
        clean_text = re.sub(r'\s+', ' ', clean_text)
        clean_text = clean_text.strip()
        
        # Добавляем паузы для лучшего звучания
        clean_text = re.sub(r'([.!?])', r'\1 ', clean_text)
        
        return clean_text, params, emotion
    
    def get_emotion_description(self, emotion: str, intensity: float = 0.5) -> str:
        """Возвращает описание эмоции."""
        descriptions = {
            "радость": "радостно и воодушевлённо",
            "удивление": "с удивлением и восхищением",
            "грусть": "грустно и задумчиво",
            "злость": "решительно и категорично",
            "восхищение": "восхищённо и восторженно",
            "смех": "весело и игриво",
            "задумчивость": "задумчиво и неторопливо",
            "поддержка": "уверенно и ободряюще",
            "ирония": "с иронией и улыбкой",
            "нейтральная": "спокойно и нейтрально"
        }
        
        if intensity < 0.2:
            return "спокойно и нейтрально"
        
        return descriptions.get(emotion, "нейтрально")

# Тестирование
def test_emotion_tts():
    """Тест эмоционального TTS."""
    print("\n" + "="*60)
    print("🎭 ТЕСТ ЭМОЦИОНАЛЬНОГО TTS")
    print("="*60)
    
    emotion_tts = EmotionTTS()
    
    test_texts = [
        "Отлично! У нас всё получилось! 😊",
        "К сожалению, это не сработало... 😢",
        "Невероятно! Вы просто гений! 😮",
        "Не волнуйтесь, я вам помогу 💪",
        "Что же нам делать? 🤔",
        "Конечно, это же очевидно! 😏",
    ]
    
    print("\n📌 Анализ эмоций:\n")
    
    for text in test_texts:
        emotions = emotion_tts.analyze_emotion(text)
        dominant = emotion_tts.get_dominant_emotion(emotions)
        params = emotion_tts.get_speaking_params(emotions)
        
        print(f"📝 {text}")
        print(f"   Эмоция: {dominant} (интенсивность: {emotions.get(dominant, 0):.2f})")
        print(f"   Параметры: скорость={params['speed']:.2f}, "
              f"высота={params['pitch']:.2f}, энергия={params['energy']:.2f}")
        print()
    
    print("✅ Тест завершён.")

if __name__ == "__main__":
    test_emotion_tts()