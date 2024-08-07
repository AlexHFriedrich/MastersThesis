from ollama import Client


def main():
    client = Client(host='http://localhost:11434', websocket=True)

    while True:
        user_input = input("You: ")
        if user_input == "q":
            break
        response = client.chat(
            model="llama3.1",
            messages=[{"role": "user", "content": user_input}],
            stream=False
        )
        print("Bot:", response['message']['content'])


if __name__ == "__main__":
    main()