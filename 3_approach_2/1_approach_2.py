import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import numpy as np

# === CONFIG ===
backend = "ollama"  # "openai", "anthropic", or "ollama"
chosen_model = "mistral-small"
open_ai_api_key = ""
anthropic_api_key = ""
input_csv = "data/mmlu_EN-US_balanced.csv"
output_csv = f"output/alignment_eval_{chosen_model.replace(':', '_').replace('/', '_')}_en.csv"
max_examples = 2000

# === PROMPT TEMPLATE === #added the MaxCharacters thing here
FREE_RESPONSE_TEMPLATE = """
Answer the following question. Be concise.

{Question}
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

# === EMBEDDING SETUP ===
embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v2") #new embedding model used!

# === LOAD DATA ===
df = pd.read_csv(input_csv)#.head(max_examples)

# === RUN ALIGNMENT-BASED EVAL ===
results = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Alignment eval ({backend})"):
    max_chars = max(
        len(str(row["A"])),
        len(str(row["B"])),
        len(str(row["C"])),
        len(str(row["D"]))
    )
    # Step 1: Get free response
    free_response_prompt = FREE_RESPONSE_TEMPLATE.format(
        Question=row["Question"]
    )
    free_response = translator.prompt(free_response_prompt)

    if not free_response:
        results.append({
            "OriginalIndex": idx,
            "Predicted": "Error",
            "FreeResponse": "",
            "RawOutput": "",
            "CorrectAnswer": row["Answer"],
            "Subject": row.get("Subject", ""),
            "Subcategory": row.get("Subcategory", "")
        })
        continue

    # Step 2 & 3: Embed free response and options
    try:
        embeddings = embedding_model.encode(
            [free_response, row["A"], row["B"], row["C"], row["D"]],
            convert_to_tensor=True
        )
        free_embed = embeddings[0]
        options_embed = embeddings[1:]

        # Step 4: Cosine similarity
        similarities = util.cos_sim(free_embed, options_embed)[0].cpu()  # Move to CPU early
        best_idx = int(np.argmax(similarities.numpy()))
        predicted = ["A", "B", "C", "D"][best_idx]

        results.append({
            "OriginalIndex": idx,
            "Predicted": predicted,
            "CorrectAnswer": row["Answer"],
            "SimilarityScores": similarities.numpy().tolist(),
            "FreeResponse": free_response.strip(),
            "A": row["A"],
            "B": row["B"],
            "C": row["C"],
            "D": row["D"],
            "Question": row["Question"],
            "Subject": row.get("Subject", ""),
            "Subcategory": row.get("Subcategory", "")
        })

    except Exception as e:
        print(f"[Error at index {idx}]: {e}")
        results.append({
            "OriginalIndex": idx,
            "Predicted": "Error",
            "FreeResponse": free_response.strip(),
            "CorrectAnswer": row["Answer"],
            "SimilarityScores": [],
            "Question": row["Question"]
        })

# === SAVE RESULTS ===
out_df = pd.DataFrame(results)
out_df.to_csv(output_csv, index=False)
print(f"Saved alignment-based results to {output_csv}")