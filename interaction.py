import ollama


def main():
    session_state = [
        {"role": "system", "content": "You are convinced that talking with a full mouth is the worst offense possible"}]

    while True:
        user_input = input("You: ")
        if user_input == "q":
            break

        session_state.append({"role": "user", "content": user_input})
        response = ollama.chat(
            model="llama3.1",
            messages=session_state,
            stream=False
        )
        print("Bot:", response['message']['content'])

        session_state.append({"role": "assistant", "content": response['message']['content']})
        print(session_state)


if __name__ == "__main__":
    main()
