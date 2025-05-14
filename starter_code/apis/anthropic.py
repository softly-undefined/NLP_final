import anthropic
import os

class AnthropicTranslator:
    def __init__(self, api_key=None, model="claude-3-opus-20240229", temperature=0, max_tokens=1000):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is missing. Set it as an environment variable or pass it as an argument.")

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