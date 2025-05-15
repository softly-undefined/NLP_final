# order_bias_eval.py

import os
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
backend = "openai"  # options: "openai", "anthropic", "ollama"
chosen_model = "gpt-4o" #"gpt-4o-mini"
open_ai_api_key = ""
anthropic_api_key = ""
input_csv = "data/mmlu_EN-US_balanced.csv"
output_csv = f"output/order_bias_{chosen_model.replace(':', '_').replace('/', '_')}_en.csv"
max_examples = 2000

print(input_csv)
print(chosen_model)

# === PROMPT TEMPLATE ===
# The Prompt is from here: https://github.com/openai/simple-evals/blob/main/common.py
QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# === LOAD TRANSLATORS ===
if backend == "openai":
    from openai import OpenAI

    class OpenAITranslator:
        def __init__(self, api_key=None, model="gpt-4", temperature=0, max_tokens=1000):
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.client = OpenAI(api_key=self.api_key)

        def prompt(self, prompt):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return resp.choices[0].message.content
            except Exception as e:
                print(f"[OpenAI:{self.model}] error: {e}")
                return None

    translator = OpenAITranslator(api_key=open_ai_api_key, model=chosen_model)

elif backend == "anthropic":
    import anthropic

    class AnthropicTranslator:
        def __init__(self, api_key=None, model="claude-3-opus-20240229", temperature=0, max_tokens=1000):
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.model = model
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.client = anthropic.Anthropic(api_key=self.api_key)

        def prompt(self, prompt):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                )
                return resp.content[0].text
            except Exception as e:
                print(f"[Anthropic:{self.model}] error: {e}")
                return None

    translator = AnthropicTranslator(api_key=anthropic_api_key, model=chosen_model)

elif backend == "ollama":
    from langchain_ollama import ChatOllama

    class OllamaTranslator:
        def __init__(self, model="llama3.1", temperature=0):
            self.model = model
            self.temperature = temperature
            self.client = ChatOllama(model=self.model, temperature=self.temperature)

        def prompt(self, prompt):
            try:
                resp = self.client.invoke(prompt)
                return resp.content
            except Exception as e:
                print(f"[Ollama:{self.model}] error: {e}")
                return None

    translator = OllamaTranslator(model=chosen_model)

else:
    raise ValueError(f"Unsupported backend: {backend}")

# === LOAD DATA ===
df = pd.read_csv(input_csv)#.head(max_examples)

# === RUN ORDER-BIAS EVAL ===
results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Order-bias eval ({backend})"):
    orig_correct = row["Answer"].strip()
    correct_text = row[orig_correct]
    other_labels = [l for l in ["A", "B", "C", "D"] if l != orig_correct]
    other_texts = [row[l] for l in other_labels]

    for target in ["A", "B", "C", "D"]:
        # build new choice mapping
        new = {}
        oth_iter = iter(other_texts)
        for label in ["A", "B", "C", "D"]:
            new[label] = correct_text if label == target else next(oth_iter)

        # format prompt
        prompt = QUERY_TEMPLATE.format(
            Question=row["Question"],
            A=new["A"],
            B=new["B"],
            C=new["C"],
            D=new["D"]
        )

        # get response
        out = translator.prompt(prompt)
        if out:
            last = out.strip().splitlines()[-1]
            pred = last.replace("Answer: ", "").strip() if last.startswith("Answer: ") else "Invalid"
        else:
            pred = "Error"

        # record
        results.append({
            "OriginalIndex": idx,
            "RotationPosition": target,
            "OriginalCorrect": orig_correct,
            "Question": row["Question"],
            "A": new["A"],
            "B": new["B"],
            "C": new["C"],
            "D": new["D"],
            "Predicted": pred,
            "RawOutput": out,
            "Subject": row.get("Subject", ""),         # NEW
            "Subcategory": row.get("Subcategory", "")  # NEW
        })


# === SAVE RESULTS ===
out_df = pd.DataFrame(results)
out_df.to_csv(output_csv, index=False)
print(f"Saved order-bias results to {output_csv}")
