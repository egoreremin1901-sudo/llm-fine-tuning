#библы
import torch
import transformers, trl, peft, datasets, evaluate
import google.protobuf
import google.protobuf
import pandas as pd
from datasets import Dataset
from evaluate import load
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from tqdm.auto import tqdm
import os
from evaluate import load



def main():
    #загружаем датасет и чутка фомратируем 
    df = pd.read_json("/content/llm_ft_data1.jsonl", lines=True)

    df = df.drop(columns=["fname", "topic"], errors="ignore")

    df = df.dropna(subset=["dialogue", "summary"])

    df["dialogue"] = df["dialogue"].astype(str)
    df["summary"] = df["summary"].astype(str)

    df = df[
        (df["dialogue"].str.strip() != "") &
        (df["summary"].str.strip() != "") &
        (df["dialogue"].str.lower().str.strip() != "nan") &
        (df["summary"].str.lower().str.strip() != "nan")
    ]

    #разделение данных 
    train_df = df.iloc[:10000].copy()
    test_df = df.iloc[10000:11000].copy()

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
    
    # функции для создания промтов под трейн и инфиренс
    def make_train_prompt(dialogue):
        return (
            "Summarize the following dialogue in 1-2 short sentences.\n\n"
            f"Dialogue:\n{dialogue}\n\n"
            "Summary:\n"
        )


    def make_infer_prompt(dialogue):
        return (
            "Summarize the following dialogue in 1-2 short sentences.\n\n"
            f"Dialogue:\n{dialogue}\n\n"
            "Summary:\n"
        )

    #загрузка модели и токенизатора 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # начинаем обучать 
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )


    #функиця где добавляем токен конца предложения(а то бля лосс раза 3 нулевой был, все перепробывал)
    def formatting_func(examples):
        texts = []

        for dialogue, summary in zip(examples["dialogue"], examples["summary"]):
            text = make_train_prompt(dialogue) + summary + tokenizer.eos_token
            texts.append(text)

        return texts
    # трейнер закидываем 
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        packing=False,
        args=training_args, #ну я их тут не вставлял, они в конфиге будут, хз как правильно
    )



    #функция  для того, что модель пыталась сумаризацию предсказать, а не еще  и диалог
    trainer = train_on_responses_only(
        trainer,
        instruction_part="Summarize the following dialogue",
        response_part="Summary:\n",
    )

    #тренировка 
    trainer_stats = trainer.train()
    
    #сохраняю модель чтобы потом ва другом колабе llm_as_judge использовать
    model.save_pretrained("qwen_summary_lora")
    tokenizer.save_pretrained("qwen_summary_lora")

    #переводим модель в инференс + функия для него
    FastLanguageModel.for_inference(model)

    def generate_summary(dialogue):
        prompt = make_infer_prompt(dialogue)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]

        return decoded.strip()

    #применям функцию для инференса  
    test_small = test_df.iloc[:100].copy()

    preds = []

    for dialogue in tqdm(test_small["dialogue"].tolist()):
        pred = generate_summary(dialogue)
        preds.append(pred)

    test_small["lora_pred"] = preds  

    #считаем метрики и записываем их файл итоговый 
    rouge = load("rouge")
    bleu = load("bleu")

    preds = test_small["lora_pred"].astype(str).tolist()
    refs = test_small["summary"].astype(str).tolist()

    rouge_result = rouge.compute(
        predictions=preds,
        references=refs
    )

    bleu_result = bleu.compute(
        predictions=preds,
        references=[[r] for r in refs]
    )

    new_experiment = pd.DataFrame([{
        "experiment_name": "llm_qwen25_05b_finetuned_lora",
        "model": "Qwen/Qwen2.5-0.5B-Instruct + LoRA",
        "notes": "fine-tuned with LoRA on dialogue summarization dataset",

        "rouge1": float(rouge_result["rouge1"]),
        "rouge2": float(rouge_result["rouge2"]),
        "rougeL": float(rouge_result["rougeL"]),
        "rougeLsum": float(rouge_result["rougeLsum"]),
        "bleu": float(bleu_result["bleu"]),
    }])

    results_path = "/content/results_metrics.csv"

    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)

        if "test_size" in results_df.columns:
            results_df = results_df.drop(columns=["test_size"])

        results_df = pd.concat([results_df, new_experiment], ignore_index=True)
    else:
        results_df = new_experiment

    results_df.to_csv(results_path, index=False)


