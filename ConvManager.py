import json
from ollama import OllamaModel

class ConvManager:
    def __init__(self):
        self.model = OllamaModel(model_name = "facebook/llama3.1")
        print(self.model.name)
        self.conv_hist = []

    def talk(self, prompt):
        response = self.model.talk(prompt)

        self.conv_hist.append({"role": "user", "content": prompt})
        self.conv_hist.append({"role": "bot", "content": response})

        return {"response": response}

    def converse(self):
        convo = {"history": self.conv_hist}
        return json.dumps(convo)