import json
import ollama


class ConvManager:
    def __init__(self):
        self.model = "llama3.1"
        self.conv_hist = []

    def talk(self, prompt):
        self.conv_hist.append({"role": "user", "content": prompt})
        response = ollama.chat({"model": self.model, "messages": self.conv_hist})
        self.conv_hist.append({"role": "bot", "content": response})

        return {"response": response}

    def converse(self):
        convo = {"history": self.conv_hist}
        return json.dumps(convo)
