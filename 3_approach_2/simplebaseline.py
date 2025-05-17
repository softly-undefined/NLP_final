import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# this code can be made as a subset of our other stuff

# === CONFIG ===
backend = "ollama"  # options: "openai", "anthropic", "ollama"
chosen_model = "llama3.2"  # or other model names depending on backend
input_csv = "data/mmlu_EN-US_balanced.csv"
output_csv = f"output/answered_{chosen_model.replace(':', '_').replace('/', '_')}.csv"
max_examples = 1000

# === PROMPT TEMPLATE ===
QUERY_TEMPLATE_MULTICHOICE = """
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
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error during prompt with OpenAI model={self.model}: {e}")
                return None

    translator = OpenAITranslator(api_key=api_key, model=chosen_model)

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
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "user", "content": [{"type": "text", "text": prompt}]}
                    ]
                )
                return response.content[0].text
            except Exception as e:
                print(f"Error during prompt with Anthropic model={self.model}: {e}")
                return None

    translator = AnthropicTranslator(api_key=api_key, model=chosen_model)

elif backend == "ollama":
    from langchain_ollama import ChatOllama

    class OllamaTranslator:
        def __init__(self, model="llama3.1", temperature=0):
            self.model = model
            self.temperature = temperature
            self.client = ChatOllama(model=self.model, temperature=self.temperature)

        def prompt(self, prompt):
            try:
                response = self.client.invoke(f"{prompt}")
                return response.content
            except Exception as e:
                print(f"Error during translation with Ollama model={self.model}: {e}")
                return None

    translator = OllamaTranslator(model=chosen_model)

else:
    raise ValueError(f"Unsupported backend: {backend}")

# === LOAD DATA ===
df = pd.read_csv(input_csv)
df = df.head(max_examples)

predictions = []
raw_outputs = []

# === RUN INFERENCE ===
for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {backend}"):
    prompt = QUERY_TEMPLATE_MULTICHOICE.format(
        Question=row["Question"],
        A=row["A"],
        B=row["B"],
        C=row["C"],
        D=row["D"]
    )
    response = translator.prompt(prompt)
    raw_outputs.append(response)

    if response:
        last_line = response.strip().split("\n")[-1]
        if last_line.startswith("Answer: "):
            prediction = last_line.replace("Answer: ", "").strip()
        else:
            prediction = "Invalid"
    else:
        prediction = "Error"

    predictions.append(prediction)

# === SAVE RESULTS ===
df["Predicted"] = predictions
df["RawOutput"] = raw_outputs
df.to_csv(output_csv, index=False)
print(f"Saved output to {output_csv}")