"""
model_training.py
Система накопления примеров и дообучения локальной модели.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------- Определяем базовую директорию ----------
BASE_DIR = Path(__file__).resolve().parent

# ---------- Загрузка конфигурации ----------
with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

TRAINING_DIR = BASE_DIR / "training"
RAW_FILE = TRAINING_DIR / CONFIG["training"]["examples_file"]
DATASET_FILE = TRAINING_DIR / CONFIG["training"]["dataset_file"]
OUTPUT_DIR = TRAINING_DIR / CONFIG["training"]["lora_output"]

SYSTEM_PROMPT = (
    "Ты JARVIS, локальный персональный ИИ-помощник. Отвечай на русском языке, "
    "не выдумывай факты, соблюдай разрешения пользователя и проси "
    "подтверждение перед опасными действиями."
)

SECRET_PATTERNS = [
    (re.compile(r"(?i)(api[_ -]?key|token|password|пароль)\\s*[:=]\\s*\\S+"), r"\1: [REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9_-]{15,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)bearer\\s+[A-Za-z0-9._-]{15,}"), "Bearer [REDACTED_TOKEN]"),
]


def sanitize(text: str) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


def ensure_dirs() -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    ensure_dirs()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def add_example(instruction: str, answer: str, rating: int, source: str = "user_feedback") -> None:
    if not instruction.strip() or not answer.strip():
        raise ValueError("instruction и answer не должны быть пустыми")
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
    print(f"✅ Пример сохранён: {RAW_FILE}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return result


def export_dataset() -> None:
    examples = read_jsonl(RAW_FILE)
    exported = []
    seen = set()

    for item in examples:
        instruction = sanitize(str(item.get("instruction", "")))
        answer = sanitize(str(item.get("answer", "")))
        if not item.get("approved") or int(item.get("rating", 0)) < 4:
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

    with DATASET_FILE.open("w", encoding="utf-8") as f:
        for item in exported:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"📊 Экспортировано примеров: {len(exported)}")
    print(f"📁 Файл: {DATASET_FILE}")


def validate_dataset() -> bool:
    examples = read_jsonl(DATASET_FILE if DATASET_FILE.exists() else RAW_FILE)
    errors = []

    for idx, item in enumerate(examples, 1):
        if "messages" in item:
            messages = item["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                errors.append(f"пример {idx}: неверное поле messages")
        elif not item.get("instruction") or not item.get("answer"):
            errors.append(f"пример {idx}: пустой instruction/answer")

    if errors:
        print("❌ Ошибки в датасете:")
        print("\n".join(errors[:20]))
        return False

    print(f"✅ Датасет корректен. Примеров: {len(examples)}")
    return True


def train_lora(model_name: str, epochs: int, batch_size: int) -> None:
    if not validate_dataset():
        raise RuntimeError("Исправьте датасет перед обучением")

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig
        from trl import SFTTrainer
    except ImportError as e:
        raise RuntimeError(
            "Не установлены зависимости обучения. Выполните:\n"
            "pip install torch transformers datasets peft trl accelerate"
        ) from e

    dataset = load_dataset("json", data_files=str(DATASET_FILE), split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    def format_example(example: dict[str, Any]) -> str:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=format_example,
        args=args,
    )
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"✅ LoRA-адаптер сохранён в: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Дообучение локальной ИИ-модели")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="добавить одобренный пример")
    add.add_argument("--instruction", required=True)
    add.add_argument("--answer", required=True)
    add.add_argument("--rating", type=int, default=5)
    add.add_argument("--source", default="user_feedback")

    sub.add_parser("export", help="создать датасет для обучения")
    sub.add_parser("validate", help="проверить датасет")

    train = sub.add_parser("train", help="обучить LoRA-адаптер")
    train.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()

    if args.command == "add":
        add_example(args.instruction, args.answer, args.rating, args.source)
    elif args.command == "export":
        export_dataset()
    elif args.command == "validate":
        sys.exit(0 if validate_dataset() else 1)
    elif args.command == "train":
        train_lora(args.model, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()