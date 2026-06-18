# импорты братские 
from unsloth import FastLanguageModel
import pandas as pd
from tqdm.auto import tqdm
import os
from getpass import getpass
from groq import Groq
import json
import re
from tqdm.auto import tqdm

def main():
    print("Hello from llm-fine-tuning!")
    #загружаем нашу дообученную на лоре модель
    max_seq_length = 1024

    MODEL_PATH = "/content/content/qwen_summary_lora" 

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    #грузим датасет и выделяем для нее тестовую выборку диалогов для проверки модели
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

    test_df = df.iloc[11000:11100].copy()
    
    # функция  для генерации промтов 
    def make_infer_prompt(dialogue):
        return (
            "Summarize the following dialogue in 1-2 short sentences.\n\n"
            f"Dialogue:\n{dialogue}\n\n"
            "Summary:\n"
    )

    #функция для инференса
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
    #инференс 
    preds = []

    for dialogue in tqdm(test_df["dialogue"].tolist()):
        pred = generate_summary(dialogue)
        preds.append(pred)

    test_df["lora_pred"] = preds

    #подключаем groq
    os.environ["GROQ_API_KEY"] = getpass("Вставь Groq API key: ")
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY")
    )

    #фунция-промт для groq
    JUDGE_MODEL = "llama-3.3-70b-versatile"

    def extract_json(text):
        text = text.strip()


        text = text.replace("```json", "").replace("```", "").strip()


        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

        return json.loads(text)


    def llm_judge(dialogue, reference_summary, model_summary): # тут после promt = f""" добавить надо, а то с ним в файле багается выделение красным
        prompt = """
    You are an expert evaluator for dialogue summarization.

    Evaluate the MODEL SUMMARY compared to the REFERENCE SUMMARY and the original DIALOGUE.

    Return ONLY raw JSON. Do not use markdown. Do not wrap the answer in a code block.

    JSON format:
    {{
      "faithfulness": 1,
      "relevance": 1,
      "completeness": 1,
      "fluency": 1,
      "overall": 1,
      "comment": "short explanation in English"
    }}

    Use integer scores from 1 to 5.
    

    Criteria:
    - faithfulness: no hallucinations, facts match the dialogue
    - relevance: summary captures important information
    - completeness: summary covers the main points
    - fluency: summary is grammatically clear
    - overall: general quality

    DIALOGUE:
    {dialogue}

    REFERENCE SUMMARY:
    {reference_summary}
    MODEL SUMMARY:
    {model_summary}
    """
    

        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Return only raw valid JSON. No markdown. No code block."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
        )

        text = response.choices[0].message.content.strip()

        try:
            result = extract_json(text)
        except:
            result = {
                "faithfulness": None,
                "relevance": None,
                "completeness": None,
                "fluency": None,
                "overall": None,
                "comment": text
            }

        return result

    #берем их трейцна 21 пример для того, чтобы groq оценил результат нашей модели
    judge_df = test_df.iloc[20:41].copy()
    # запускаем llm_as_judge
    judge_results = []

    for _, row in tqdm(judge_df.iterrows(), total=len(judge_df)):
        result = llm_judge(
            dialogue=row["dialogue"],
            reference_summary=row["summary"],
            model_summary=row["lora_pred"]
        )

        judge_results.append(result)

        # чтобы не упереться в лимиты
        time.sleep(2)

    # создаем датафрейм результатов модели 
    judge_scores_df = pd.DataFrame(judge_results)

    judge_df = pd.concat(
        [judge_df.reset_index(drop=True), judge_scores_df.reset_index(drop=True)],
        axis=1
    )

    judge_df.head()
    
    #сохраняем результат
    judge_df.to_csv("/content/llama-3.3-70b-versatile.csv", index=False)

if __name__ == "__main__":
    main()
