from openai import OpenAI
import os

class OpenAITranslator:
    def __init__(self, api_key=None, model="gpt-4", temperature=0, max_tokens=1000):
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
