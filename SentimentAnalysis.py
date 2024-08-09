from transformers import pipeline


def load_model(path):
    model = pipeline(model=path)
    return model


def call_model(model, text):
    return model(text)
