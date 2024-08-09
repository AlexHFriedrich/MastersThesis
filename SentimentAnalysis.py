from zeroshot_classifier.models import BinaryBertCrossEncoder


def zero_shot_test(input_text, labels, aspect='intent'):
    model = BinaryBertCrossEncoder(model_name='claritylab/zero-shot-implicit-binary-bert')

    text = input_text
    labels = labels
    aspect = aspect
    aspect_sep_token = model.tokenizer.additional_special_tokens[0]
    text = f'{aspect} {aspect_sep_token} {text}'

    query = [[text, lb] for lb in labels]
    logits = model.predict(query, apply_softmax=True)
    print(logits)
