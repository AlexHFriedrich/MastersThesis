import json
import ollama


class ConvManager:
    def __init__(self):
        self.model = "llama3.1"
        self.conv_hist = []

    def talk(self, prompt):
        self.conv_hist.append({"role": "user", "content": prompt})
        response = ollama.chat(model=self.model, messages=self.conv_hist, stream=False)
        self.conv_hist.append({"role": "assistant", "content": response})

        return {"response": response}
