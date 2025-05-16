from langchain_ollama import ChatOllama
import os

models = [
    'deepseek-r1:7b', 
    'llama3.2', 
    'llama3.1'
]
model = models[0]

class OllamaTranslator:
    def __init__(self, model="llama3.1", temperature=0):
        self.model = model
        self.temperature = temperature
        self.client = ChatOllama(model=self.model, temperature=self.temperature)

    def prompt(self, prompt):
        try:
            response = self.client.invoke(f"{prompt}")
            return response.content  # Extracts text from the response
        except Exception as e:
            print(f"Error during translation with Olama model={self.model}: {e}")
            return None
