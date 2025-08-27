import torch
import re
from app.model_loader import load_model
from app.config import MODEL_NAME

model, tokenizer = load_model(MODEL_NAME)

def generate_response(prompt: str, max_tokens: int = 200) -> str:
    # Форматированный prompt
    formatted_prompt = f"<s>[INST] Инструкция: {prompt} Ответь строго по регламенту, без лишней информации. [/INST]"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.95,
            top_p=0.9,
            repetition_penalty=1.2,
            use_cache=True
        )

    # Отделяем только ответ, исключая prompt
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Обрезаем текст по ключевым "галлюциногенным" фразам
    cutoffs = [
        "Если не помогло — обратитесь",
        "Также вы можете",
        "Если проблема сохраняется",
        "В заключение",
        "Надеюсь, это помогло",
        "Если устройство подключено",
        "Если не получилось"
    ]
    for cutoff in cutoffs:
        if cutoff in answer:
            answer = answer.split(cutoff)[0].strip()

    # Ограничим количество предложений (макс 3)
    sentences = re.split(r'(?<=[.!?]) +', answer)
    answer = ' '.join(sentences[:3]).strip()

    return answer
