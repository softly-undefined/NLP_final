import os
import re
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
backend = "ollama"  # options: "openai", "anthropic", "ollama"
chosen_model = "mistral-small"
open_ai_api_key = ""
anthropic_api_key = ""
input_csv = "data/new_0.2_alignment_eval_mistral-small_cn.csv"
output_csv = f"output/new_0.2_order_bias_{chosen_model.replace(':', '_').replace('/', '_')}_cot_zh.csv"
error_log_path = "output/2errors.txt"
max_examples = 2000
save_every = 200

print(input_csv)
print(chosen_model)

# === PROMPT TEMPLATE ===
QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# === EXTRACT ANSWER ===
ANSWER_PATTERN = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"

def extract_answer_from_output(text):
    if not isinstance(text, str):
        return "Error"

    # Try regex
    match = re.search(ANSWER_PATTERN, text)
    if match:
        return match.group(1).upper()

    # Fallback: check last few lines for single-letter answer
    lines = text.strip().splitlines()
    for line in reversed(lines[-5:]):
        if line.strip().upper() in ['A', 'B', 'C', 'D']:
            return line.strip().upper()

    return "Invalid"

# === TRANSLATOR CLASSES ===
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
            self.client = ChatOllama(
                            model=self.model,
                            temperature=self.temperature,
                            num_predict=512,
                        )

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

# === LOAD EXISTING PROGRESS IF AVAILABLE ===
if os.path.exists(output_csv):
    existing = pd.read_csv(output_csv)
    done_keys = set((r["OriginalIndex"], r["RotationPosition"]) for _, r in existing.iterrows())
    results = existing.to_dict("records")
    print(f"Resuming from {len(done_keys)} completed prompts...")
else:
    done_keys = set()
    results = []

# === RUN ORDER-BIAS EVAL ===
for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Order-bias eval ({backend})"):
    orig_correct = row["CorrectAnswer"].strip()
    correct_text = row[orig_correct]
    other_labels = [l for l in ["A", "B", "C", "D"] if l != orig_correct]
    other_texts = [row[l] for l in other_labels]

    for target in ["A", "B", "C", "D"]:
        key = (idx, target)
        if key in done_keys:
            continue  # Skip already processed

        new = {}
        oth_iter = iter(other_texts)
        for label in ["A", "B", "C", "D"]:
            new[label] = correct_text if label == target else next(oth_iter)

        prompt = QUERY_TEMPLATE.format(
            Question=row["Question"],
            A=new["A"],
            B=new["B"],
            C=new["C"],
            D=new["D"]
        )

        out = translator.prompt(prompt)
        pred = extract_answer_from_output(out)

        if pred in ["Invalid", "Error"]:
            os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[Failure] idx={idx}, target={target}\n")
                f.write(f"Prompt:\n{prompt}\n")
                f.write(f"Output:\n{out}\n")
                f.write("-" * 80 + "\n")

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
            "Subject": row.get("Subject", ""),
            "Subcategory": row.get("Subcategory", "")
        })

        if len(results) % save_every == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False)
            print(f"[Checkpoint] Saved {len(results)} entries so far...")

# === SAVE RESULTS ===
# Final save
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"Final save completed: {len(results)} entries written to {output_csv}")