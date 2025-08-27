from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import os

def load_model(model_name: str = "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24", lora_path="vikhr_nemo_12B_lora_out"):
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # важно для генерации

    # Настройки 8-bit загрузки
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False
    )

    # Загружаем базовую модель
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Проверяем наличие LoRA-модели
    print(f"Ищем LoRA-модель в папке: {lora_path}")
    if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        print("LoRA-модель найдена, загружаем...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        print("✅ LoRA-модель успешно загружена!")
    else:
        model = base_model
        print("⚠️ LoRA-модель не найдена. Используется только базовая модель.")

    model.eval()  # Переводим модель в режим генерации

    print(f"📦 Модель загружена на устройство: {model.device}")
    return model, tokenizer

