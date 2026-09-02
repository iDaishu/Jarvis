"""
model_training.py
Система накопления примеров и дообучения локальной модели.

Обеспечивает сбор обратной связи от пользователя и дообучение модели.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from utils.logger import LoggerFactory
from utils.exceptions import JARVISError

# ---------- Настройка логирования ----------
LoggerFactory.setup()
logger = LoggerFactory.get_logger("model_training")

# ---------- Определяем базовую директорию ----------
BASE_DIR = Path(__file__).resolve().parent

# ---------- Загрузка конфигурации ----------
def load_config() -> Dict[str, Any]:
    """Безопасная загрузка конфигурации."""
    config_path = BASE_DIR / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        sys.exit(1)

CONFIG = load_config()

TRAINING_DIR = BASE_DIR / "training"
RAW_FILE = TRAINING_DIR / CONFIG["training"]["examples_file"]
DATASET_FILE = TRAINING_DIR / CONFIG["training"]["dataset_file"]
OUTPUT_DIR = TRAINING_DIR / CONFIG["training"]["lora_output"]

SYSTEM_PROMPT = (
    "Ты JARVIS, локальный персональный ИИ-помощник. Отвечай на русском языке, "
    "не выдумывай факты, соблюдай разрешения пользователя и проси "
    "подтверждение перед опасными действиями."
)

# Шаблоны для санитайзинга
SECRET_PATTERNS = [
    (re.compile(r"(?i)(api[_ -]?key|token|password|пароль)\\s*[:=]\\s*\\S+"), r"\1: [REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9_-]{15,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)bearer\\s+[A-Za-z0-9._-]{15,}"), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"\\b\\d{3}-\\d{2}-\\d{4}\\b"), "[REDACTED_SSN]"),  # SSN
]


def sanitize(text: str) -> str:
    """
    Санитайзинг текста от конфиденциальной информации.
    
    Args:
        text: Исходный текст
        
    Returns:
        str: Очищенный текст
    """
    if not text:
        return ""
    
    result = text
    for pattern, replacement in SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result.strip()


def ensure_dirs() -> None:
    """Создание необходимых директорий."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    """
    Добавление записи в JSONL файл.
    
    Args:
        path: Путь к файлу
        item: Данные для добавления
        
    Raises:
        JARVISError: При ошибке записи
    """
    ensure_dirs()
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.debug(f"Запись добавлена: {path}")
    except Exception as e:
        logger.error(f"Ошибка записи: {e}")
        raise JARVISError(f"Не удалось добавить запись: {e}")


def add_example(instruction: str, answer: str, rating: int, 
                source: str = "user_feedback") -> None:
    """
    Добавление одобренного примера для обучения.
    
    Args:
        instruction: Инструкция пользователя
        answer: Ответ ассистента
        rating: Оценка (1-5)
        source: Источник примера
        
    Raises:
        ValueError: При невалидных данных
    """
    if not instruction or not instruction.strip():
        raise ValueError("instruction не должен быть пустым")
    if not answer or not answer.strip():
        raise ValueError("answer не должен быть пустым")
    if rating < 1 or rating > 5:
        raise ValueError("rating должен быть от 1 до 5")
    if rating < 4:
        raise ValueError("Для обучения пример должен иметь rating 4 или 5")

    item = {
        "instruction": sanitize(instruction),
        "answer": sanitize(answer),
        "rating": rating,
        "source": source,
        "approved": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    
    append_jsonl(RAW_FILE, item)
    logger.info(f"Пример сохранён: {RAW_FILE}")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Чтение JSONL файла.
    
    Args:
        path: Путь к файлу
        
    Returns:
        List[Dict[str, Any]]: Список записей
    """
    if not path.exists():
        return []
    
    result = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        result.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Ошибка парсинга строки: {e}")
                        continue
    except Exception as e:
        logger.error(f"Ошибка чтения файла {path}: {e}")
    
    return result


def export_dataset() -> None:
    """Экспорт датасета для обучения."""
    examples = read_jsonl(RAW_FILE)
    exported = []
    seen = set()
    
    for item in examples:
        instruction = sanitize(str(item.get("instruction", "")))
        answer = sanitize(str(item.get("answer", "")))
        
        if not item.get("approved", False):
            continue
        if int(item.get("rating", 0)) < 4:
            continue
        if not instruction or not answer:
            continue
        
        key = (instruction, answer)
        if key in seen:
            continue
        seen.add(key)
        
        exported.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": answer},
            ]
        })
    
    # Запись датасета
    try:
        with DATASET_FILE.open("w", encoding="utf-8") as f:
            for item in exported:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Экспортировано примеров: {len(exported)}")
        logger.info(f"Файл: {DATASET_FILE}")
    except Exception as e:
        logger.error(f"Ошибка экспорта: {e}")
        raise JARVISError(f"Не удалось экспортировать датасет: {e}")


def validate_dataset() -> bool:
    """
    Валидация датасета.
    
    Returns:
        bool: Корректен ли датасет
    """
    source_file = DATASET_FILE if DATASET_FILE.exists() else RAW_FILE
    examples = read_jsonl(source_file)
    errors = []
    
    for idx, item in enumerate(examples, 1):
        if "messages" in item:
            messages = item["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                errors.append(f"пример {idx}: неверное поле messages")
            else:
                # Проверка ролей
                roles = [msg.get("role") for msg in messages]
                if "user" not in roles or "assistant" not in roles:
                    errors.append(f"пример {idx}: отсутствуют роли user/assistant")
        elif not item.get("instruction") or not item.get("answer"):
            errors.append(f"пример {idx}: пустой instruction/answer")
    
    if errors:
        logger.error("Ошибки в датасете:")
        for error in errors[:20]:
            logger.error(f"  • {error}")
        return False
    
    logger.info(f"Датасет корректен. Примеров: {len(examples)}")
    return True


def train_lora(model_name: str, epochs: int, batch_size: int) -> None:
    """
    Обучение LoRA-адаптера.
    
    Args:
        model_name: Имя базовой модели
        epochs: Количество эпох
        batch_size: Размер батча
        
    Raises:
        JARVISError: При ошибке обучения
        ImportError: Если не установлены зависимости
    """
    if not validate_dataset():
        raise JARVISError("Исправьте датасет перед обучением")
    
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
    except ImportError as e:
        raise ImportError(
            "Не установлены зависимости обучения. Выполните:\n"
            "pip install torch transformers datasets peft trl accelerate"
        ) from e
    
    try:
        # Загрузка датасета
        dataset = load_dataset("json", data_files=str(DATASET_FILE), split="train")
        
        # Токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Модель
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        
        # LoRA конфигурация
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Форматирование примеров
        def format_example(example: Dict[str, Any]) -> str:
            return tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        
        # Аргументы обучения
        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none",
            fp16=torch.cuda.is_available(),
        )
        
        # Тренер
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            formatting_func=format_example,
            args=args,
            tokenizer=tokenizer,
        )
        
        # Обучение
        logger.info(f"Начало обучения на {len(dataset)} примерах...")
        trainer.train()
        
        # Сохранение
        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        logger.info(f"LoRA-адаптер сохранён в: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Ошибка обучения: {e}")
        raise JARVISError(f"Не удалось обучить модель: {e}")


def main() -> None:
    """Точка входа."""
    parser = argparse.ArgumentParser(description="Дообучение локальной ИИ-модели")
    sub = parser.add_subparsers(dest="command", required=True, help="Команда")
    
    # add
    add = sub.add_parser("add", help="Добавить одобренный пример")
    add.add_argument("--instruction", required=True, help="Инструкция пользователя")
    add.add_argument("--answer", required=True, help="Ответ ассистента")
    add.add_argument("--rating", type=int, default=5, choices=range(1, 6), 
                     help="Оценка (1-5)")
    add.add_argument("--source", default="user_feedback", 
                     help="Источник примера")
    
    # export
    sub.add_parser("export", help="Создать датасет для обучения")
    
    # validate
    sub.add_parser("validate", help="Проверить датасет")
    
    # train
    train = sub.add_parser("train", help="Обучить LoRA-адаптер")
    train.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                       help="Имя базовой модели")
    train.add_argument("--epochs", type=int, default=2, help="Количество эпох")
    train.add_argument("--batch-size", type=int, default=1, help="Размер батча")
    
    args = parser.parse_args()
    
    try:
        if args.command == "add":
            add_example(args.instruction, args.answer, args.rating, args.source)
        elif args.command == "export":
            export_dataset()
        elif args.command == "validate":
            sys.exit(0 if validate_dataset() else 1)
        elif args.command == "train":
            train_lora(args.model, args.epochs, args.batch_size)
    except KeyboardInterrupt:
        logger.info("Операция прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()