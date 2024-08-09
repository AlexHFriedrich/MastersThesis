import time
import ollama

from SentimentAnalysis import load_model, call_model


def main():
    session_state = [
        {"role": "system", "content": "You are convinced that talking with a full mouth is the worst offense possible"}]

    session_state_second_instance = [
        {"role": "system", "content": "You really like to talk while eating and like talking about it even more"}]

    model, tokenizer = load_model("ft_bert_emotion")

    while len(session_state) < 10:
        start = time.time()
        first_actor = ollama.chat(
            model="llama3.1",
            messages=session_state_second_instance,
            stream=False,
            options={"num_predict": 50}
        )
        print(len(first_actor['message']['content']))
        if len(first_actor['message']['content']) > 0:
            print("First:", first_actor['message']['content'], "\n")
            # here the SA model would be called to determine the sentiment of the first actor output
            sentiment = call_model(model, tokenizer, first_actor['message']['content'])
            print(sentiment)
        # print("First:", first_actor['message']['content'], "\n")
        session_state.append({"role": "user", "content": first_actor['message']['content']})
        session_state_second_instance.append({"role": "user", "content": first_actor['message']['content']})

        second_actor = ollama.chat(
            model="llama3.1",
            messages=session_state,
            stream=False,
            options={"num_predict": 50}
        )

        session_state.append({"role": "assistant", "content": second_actor})
        session_state_second_instance.append({"role": "assistant", "content": second_actor})

        print(time.time() - start)


if __name__ == "__main__":
    main()
