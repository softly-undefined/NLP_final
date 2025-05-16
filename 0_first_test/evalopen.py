from openai import OpenAI
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Model to use
chosen_model = "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo", etc.
print(f"Using model: {chosen_model}")

class OpenAITranslator:
    def __init__(self, api_key=None, model="gpt-4o-mini", temperature=0, max_tokens=1000):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is missing. Set it as an environment variable or pass it as an argument.")

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
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during prompt with OpenAI model={self.model}: {e}")
            return None

# Prompt template
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

translator = OpenAITranslator(model=chosen_model, api_key="")

# Load dataset
df = pd.read_csv('test.csv')
df = df.head(100)

predictions = []
raw_outputs = []

# Loop with tqdm progress bar
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
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

# Save results
df["Predicted"] = predictions
df["RawOutput"] = raw_outputs
df.to_csv(f'answered_{chosen_model.replace(":", "_")}.csv', index=False)
