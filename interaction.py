import ollama

while True:
    prompt = input("You: ")

    if prompt == "q":
        break

    stream = ollama.chat(
        model="llama3.1", messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
