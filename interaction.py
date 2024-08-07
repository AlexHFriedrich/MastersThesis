import time

import ollama


def main():
    session_state = [
        {"role": "system", "content": "You are convinced that talking with a full mouth is the worst offense possible"}]

    session_state_second_instance = [
        {"role": "system", "content": "You really like to talk while eating and like talking about it even more"}]
    run_times = []

    while len(session_state) < 10:

        first_actor = ollama.chat(
            model="llama3.1",
            messages=session_state_second_instance,
            stream=False,
            options={"num_predict": 50}
        )
        # print("First:", first_actor['message']['content'], "\n")
        session_state.append({"role": "user", "content": first_actor['message']['content']})
        session_state_second_instance.append({"role": "user", "content": first_actor['message']['content']})

        # measure the time it takes to sample and choose a response
        start = time.time()
        possible_answers = []
        while len(possible_answers) < 5:
            possible_answers.append(ollama.chat(
                model="llama3.1",
                messages=session_state,
                stream=False,
                options={"num_predict": 50}
            )['message']['content'])

        choice = 1
        # here would be the RL agent choosing an optimal solution out of the sampled ones

        second_actor = possible_answers[choice]
        # print("Second:", second_actor['message']['content'], "\n")

        session_state.append({"role": "assistant", "content": second_actor})
        session_state_second_instance.append({"role": "assistant", "content": second_actor})
        run_times.append(time.time() - start)
        print(run_times[-1])

    print(session_state)
    print(run_times)
    print(sum(run_times) / len(run_times))


if __name__ == "__main__":
    main()
