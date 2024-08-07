import time

import ollama


def main():
    session_state = [
        {"role": "system", "content": "You are convinced that talking with a full mouth is the worst offense possible"}]

    session_state_second_instance = [
        {"role": "system", "content": "You really like to talk while eating and like talking about it even more"}]
    run_times = []

    while len(session_state) < 30:
        start = time.time()
        first_actor = ollama.chat(
            model="llama3.1",
            messages=session_state_second_instance,
            stream=False
        )
        # print("First:", first_actor['message']['content'], "\n")
        session_state.append({"role": "user", "content": first_actor['message']['content']})
        session_state_second_instance.append({"role": "assistant", "content": first_actor['message']['content']})
        second_actor = ollama.chat(
            model="llama3.1",
            messages=session_state,
            stream=False
        )
        # print("Second:", second_actor['message']['content'], "\n")

        session_state.append({"role": "assistant", "content": second_actor['message']['content']})
        session_state_second_instance.append({"role": "user", "content": second_actor['message']['content']})
        run_times.append((time.time() - start) / len(session_state[-1]) + len(session_state_second_instance[-2]))

    print(session_state)
    print(sum(run_times) / len(run_times))


if __name__ == "__main__":
    main()
