import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# 1. Загрузка токенизатора и 4-битной модели
model_name = "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# Настраиваем quantization config для 4-битной загрузки (NF4 quantization, FP16 вычисления)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # использовать нормализованное 4-битное квантование (NF4)
    bnb_4bit_use_double_quant=True,   # включить двойное квантование (для стабильности)
    bnb_4bit_compute_dtype=torch.float16  # вычисления градиентов в FP16 (можно заменить на torch.bfloat16 при необходимости)
)
# Загружаем модель в 4-битном режиме с автоматическим распределением по устройствам
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",       # автоматически разместит части модели на доступных устройствах (GPU/CPU)
    trust_remote_code=True   # доверять кастомному коду (если модель не стандартной архитектуры)
)

# Включаем gradient checkpointing для экономии памяти:
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)  # подготавливаем модель для 4-битного обучения (напр. отключает ненужные градиенты)

# 2. Конфигурация LoRA-адаптеров
lora_config = LoraConfig(
    r=16,               # Rank=16, рекомендуемое значение для 12B на 10-12ГБ VRAM (баланс качества и памяти)
    lora_alpha=32,      # Alpha=16 (равен r, стандартный выбор; можно 32 для усиления влияния LoRA)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,   # Dropout=0, т.к. незначительно влияет на качество в большинстве случаев
    bias="none",        # не добавляем обучаемый смещение (обычно не нужно для LoRA)
    task_type="CAUSAL_LM"  # тип задачи - авторегрессионная LM
)
model = get_peft_model(model, lora_config)
print("LoRA адаптеры добавлены. Обучаемые параметры:")
model.print_trainable_parameters()  # выведет информацию о числе обучаемых параметров vs замороженных

# 3. Загрузка и подготовка датасета
# Предполагается, что у вас есть файл с инструкциями и ответами в формате JSON или другом.
# Здесь мы ожидаем формат с полями "instruction" и "response" или "input" и "output".
# Замените пути и поля на свои при необходимости.
data_files = {"train": "C:\\Users\\Araize\\Desktop\\server_model\\data\\train_data_expanded.jsonl"}  # Укажите путь к своему датасету
dataset = load_dataset("json", data_files=data_files)
train_data = dataset["train"]

# Функция препроцессинга: объединяем инструкцию и ответ в подсказку для модели.
# Пример предполагаемого формата: {"instruction": "...", "input": "...", "output": "..."}
# Функция препроцессинга: объединяем инструкцию и ответ в подсказку для модели.
def format_example(example):
    instruction = example.get("instruction") or example.get("prompt") or example.get("question")
    input_text = example.get("input")  # дополнительный контекст или None
    if input_text:
        prompt = f"### Инструкция:\n{instruction}\n### Ввод:\n{input_text}\n### Ответ:\n"
    else:
        prompt = f"### Инструкция:\n{instruction}\n### Ответ:\n"
    
    target = example.get("output") or example.get("response") or example.get("answer")
    
    # Токенизация с паддингом и обрезкой
    prompt_ids = tokenizer(prompt, add_special_tokens=True, padding="max_length", truncation=True, max_length=1024, return_tensors="pt").input_ids.squeeze()
    target_ids = tokenizer(target, add_special_tokens=True, padding="max_length", truncation=True, max_length=1024, return_tensors="pt").input_ids.squeeze()

    # Объединяем вход и ответ
    input_ids = torch.cat([prompt_ids, target_ids])
    
    # Создаем метки с игнорированием токенов инструкции
    labels = torch.cat([torch.full(prompt_ids.shape, -100), target_ids])  # игнорируем потери на части prompt
    
    return {"input_ids": input_ids, "labels": labels}


# Применяем препроцессинг к датасету
train_data = train_data.map(format_example, remove_columns=train_data.column_names, batched=False)

# 4. Настройка тренировки
output_dir = "vikhr_nemo_12B_lora_out"
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=2,   # небольшой batch на GPU (2) ...
    gradient_accumulation_steps=8,   # ... компенсируется накоплением градиентов до эффективного batch 16
    num_train_epochs=2,              # 2 эпохи (можно 1-3 в зависимости от размера данных)
    learning_rate=2e-4,              # начальный learning rate 2e-4 (оптимальный стартовый)
    warmup_ratio=0.05,               # 5% шагов на разогрев LR
    lr_scheduler_type="cosine",      # планировщик (можно "linear" или "cosine")
    fp16=True,                      # использовать 16-битные вычисления (FP16 AMP) для ускорения
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    dataloader_num_workers=2,
    report_to="none",               # отключаем лишние логгеры (не нужен Wandb и т.п. если не используется)
)

# 5. Запуск обучения с Trainer
if __name__ == '__main__':
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
    )
    trainer.train()

    # После обучения сохраним полученные LoRA-веса
    model.save_pretrained(output_dir)
    print(f"LoRA fine-tuning завершён. Адаптеры сохранены в {output_dir}")
