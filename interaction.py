import ollama


def main():
    session_state = [
        {"role": "system", "content": "You are convinced that talking with a full mouth is the worst offense possible"}]

    session_state_second_instance = [{"role": "system", "content": "You really like to talk while eating"}]
    while len(session_state) < 10:
        first_actor = ollama.chat(
            model="llama3.1",
            messages=session_state_second_instance,
            stream=False
        )
        print("First:", first_actor['message']['content'])
        session_state.append({"role": "user", "content": first_actor['message']['content']})
        second_actor = ollama.chat(
            model="llama3.1",
            messages=session_state,
            stream=False
        )
        print("Second:", second_actor['message']['content'])

        session_state.append({"role": "assistant", "content": second_actor['message']['content']})

    print(session_state)


if __name__ == "__main__":
    main()
