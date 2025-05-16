from langchain_ollama import ChatOllama
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # <-- added

models = [
    'deepseek-r1:7b', 
    'llama3.2', 
    'llama3.1'
]
chosen_model = models[1]
print(chosen_model)

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
            print(f"Error during translation with Olama model={self.model}: {e}")
            return None

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

translator = OllamaTranslator(model=chosen_model)

df = pd.read_csv('test.csv')
df = df.head(100)

predictions = []
raw_outputs = []

# tqdm progress bar added here
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

df["Predicted"] = predictions
df["RawOutput"] = raw_outputs

df.to_csv(f'answered_{chosen_model.replace(":", "_")}.csv', index=False)
