from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def load_model(path):
    model = AutoModelForSequenceClassification.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer


def tokenize(tokenizer, text):
    return tokenizer(text, truncation=True, max_length=512)


def call_model(model, tokenizer, text):
    tokenized_text = tokenize(tokenizer, text)
    print(tokenized_text)
    return model(**tokenized_text)
