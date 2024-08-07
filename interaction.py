import json
from ConvManager import ConvManager


def main():
    conv_manager = ConvManager()
    while True:
        user_input = input("You: ")
        if user_input == "q":
            break

        response = conv_manager.talk(user_input)
        print("Bot:", response)

    print(conv_manager.conv_hist)


if __name__ == "__main__":
    main()
