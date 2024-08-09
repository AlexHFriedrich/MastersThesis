from transformers import pipeline


def load_model():
    model = pipeline("zero-shot-classification")
    return model


def call_model(model, text, labels):
    return model(text, labels)
