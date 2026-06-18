import os
import json
import math
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from evaluate import load
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from vllm import LLM, SamplingParams
from rouge_score import rouge_scorer
import sacrebleu
import kagglehub
import wandb


def main():

    #загружаем датасет и EDA
    df = pd.read_json(r"/content/llm_ft_data1.jsonl", lines=True )

    df = df.drop(['fname', 'topic'], axis=1)
    # я хочу обрезать датасет, чтобы 5к примеров осталось и не обучать кучу данных
    df = df.head(5000)
    df.isna().sum()

    df["dialogue_chars"] = df["dialogue"].astype(str).str.len()
    df["summary_chars"] = df["summary"].astype(str).str.len()

    df["dialogue_words"] = df["dialogue"].astype(str).str.split().str.len()
    df["summary_words"] = df["summary"].astype(str).str.split().str.len()

    df[["dialogue_chars", "summary_chars", "dialogue_words", "summary_words"]].describe()

    # смотрю сколько слов в датасете под суммаризацию.   чтобы в параметрах своей оптимальные значения поставить
    df["summary_words"].quantile(0.95)
    # ну кароче 64 токена плюс минус норм на 42 слова в 95 перцентиле( потом использовал128)

    # делаем суммаризацию без мл (первой плюс ласт предложение)
    def first_last_baseline(dialogue):


        replicas = [rep.strip() for rep in str(dialogue).split("\n") if rep.strip()]

        if len(replicas) == 0:
            return ""
        elif len(replicas) == 1:
            return replicas[0]
        else:
            return replicas[0] + " " + replicas[-1]

    df["baseline_pred"] = df["dialogue"].apply(first_last_baseline)

    df[["dialogue", "summary", "baseline_pred"]].head()

    #считавем метрики и записываем в итоговую таблицу
    rouge = load("rouge")
    bleu = load("bleu")

    preds = df["baseline_pred"].astype(str).tolist()
    refs = df["summary"].astype(str).tolist()

    rouge_result = rouge.compute(predictions=preds, references=refs)
    bleu_result = bleu.compute(predictions=preds, references=[[r] for r in refs])


    experiments = []
    experiments.append({
        "experiment_name": "baseline_first_last",
        "model": "rule_based",
        "notes": "first replica + last replica",
        "rouge1": float(rouge_result["rouge1"]),
        "rouge2": float(rouge_result["rouge2"]),
        "rougeL": float(rouge_result["rougeL"]),
        "rougeLsum": float(rouge_result["rougeLsum"]),
        "bleu": float(bleu_result["bleu"])
    })
    results_df = pd.DataFrame(experiments)


    # теперь делаем бейзлайн llM БЕЗ FT

    #маленький кусок для теста
    llm_df = df[["dialogue", "summary"]].copy()

    df_small = llm_df.iloc[:50].copy()
    def build_prompt(dialogue: str) -> str:
        return (
            "Summarize the following dialogue in 1-2 short sentences.\n\n"
            f"Dialogue:\n{dialogue}\n\n"
            "Summary:"
        )

    prompts = df_small["dialogue"].apply(build_prompt).tolist()

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    llm = LLM(
        model=model_name,
        dtype="float16"
    )


    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64
    )


    outputs = llm.generate(prompts, sampling_params)

    # Забираем текст ответов
    preds = [out.outputs[0].text.strip() for out in outputs]


    df_small["llm_pred"] = preds

    df_small[["dialogue", "summary", "llm_pred"]].head()


    #МЕТРИКИ ДЛЯ LLM
    preds = df_small["llm_pred"].astype(str).tolist()
    refs = df_small["summary"].astype(str).tolist()

    rouge_result = rouge.compute(predictions=preds, references=refs)
    bleu_result = bleu.compute(predictions=preds, references=[[r] for r in refs])

    experiments.append({
        "experiment_name": "llm_qwen25_05b_zero_shot",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "notes": "zero-shot summarization with prompt, no fine-tuning",
        "rouge1": float(rouge_result["rouge1"]),
        "rouge2": float(rouge_result["rouge2"]),
        "rougeL": float(rouge_result["rougeL"]),
        "rougeLsum": float(rouge_result["rougeLsum"]),
        "bleu": float(bleu_result["bleu"])
    })

    results_df = pd.DataFrame(experiments)
    # ну норм, моделька лучше чем просто первой плюс ласт, значит ее можно дальше пробывать

if __name__ == "__main__":
    main()

