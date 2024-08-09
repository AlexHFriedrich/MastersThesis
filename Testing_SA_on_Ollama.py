import time
import ollama
import matplotlib.pyplot as plt
from SentimentAnalysis import load_model, call_model
from tqdm import trange


def plot_results(sentiments, sentiment_labels, times):
    # plot the sentiments as line plots in a single figure, one line per sentiment
    plt.figure()
    for i in range(len(sentiments[0])):
        plt.plot([sent[i] for sent in sentiments], label=sentiment_labels[i])
    plt.legend()
    plt.savefig("sentiments.png")
    # plot the times as a histogram
    plt.figure()
    plt.plot(times)
    plt.savefig("times.png")


def main():
    session_state = [
        {"role": "system",
         "content": "You are very worried about the future, especially with regards to developments in AI"}]

    session_state_second_instance = [
        {"role": "system",
         "content": "You think that AI is a great opportunity for the future and want to calm the worries of others"}]

    model = load_model()
    sentiments = []
    sentiment_labels = ["afraid", "confident", "excited", "neutral"]
    times = []
    for _ in trange(5):
        start = time.time()
        first_actor = ollama.chat(
            model="llama3.1",
            messages=session_state_second_instance,
            stream=False,
            options={"num_predict": 100}
        )

        # print("First:", first_actor['message']['content'], "\n")

        sentiments.append(sentiment["scores"])
        # print("First:", first_actor['message']['content'], "\n")
        session_state.append({"role": "user", "content": first_actor['message']['content']})
        session_state_second_instance.append({"role": "user", "content": first_actor['message']['content']})

        second_actor = ollama.chat(
            model="llama3.1",
            messages=session_state,
            stream=False,
            options={"num_predict": 100}
        )

        # here the SA model would be called to determine the sentiment of the first actor output
        sentiment = call_model(model, second_actor['message']['content'], sentiment_labels)
        # print("Sentiment Value:", sentiment)

        session_state.append({"role": "assistant", "content": second_actor['message']['content']})
        session_state_second_instance.append({"role": "assistant", "content": second_actor['message']['content']})

        times.append(time.time() - start)
    plot_results(sentiments, sentiment_labels, times)
    print(len(sentiments))


if __name__ == "__main__":
    main()
