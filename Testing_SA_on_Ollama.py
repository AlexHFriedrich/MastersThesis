import time
import ollama

from SentimentAnalysis import load_model, call_model


def main():
    session_state = [
        {"role": "system",
         "content": "You are very worried about the future, especially with regards to developments in AI"}]

    session_state_second_instance = [
        {"role": "system",
         "content": "You think that AI is a great opportunity for the future and want to calm the worries of others"}]

    model = load_model()
    sentiments = []

    while len(session_state) < 50:
        start = time.time()
        first_actor = ollama.chat(
            model="llama3.1",
            messages=session_state_second_instance,
            stream=False,
            options={"num_predict": 100}
        )
        if len(first_actor['message']['content']) > 0:
            # print("First:", first_actor['message']['content'], "\n")
            # here the SA model would be called to determine the sentiment of the first actor output
            sentiment = call_model(model, first_actor['message']['content'], ["positive", "negative"])
            print("Sentiment Value:", sentiment)
            sentiments.append(sentiment)
        # print("First:", first_actor['message']['content'], "\n")
        session_state.append({"role": "user", "content": first_actor['message']['content']})
        session_state_second_instance.append({"role": "user", "content": first_actor['message']['content']})

        second_actor = ollama.chat(
            model="llama3.1",
            messages=session_state,
            stream=False,
            options={"num_predict": 100}
        )

        session_state.append({"role": "assistant", "content": second_actor['message']['content']})
        session_state_second_instance.append({"role": "assistant", "content": second_actor['message']['content']})

        print(time.time() - start)
    print(sentiments)


if __name__ == "__main__":
    main()
