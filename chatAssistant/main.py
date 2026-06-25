try:
    from .chatbot.chat import start_chat
except ImportError:
    from chatbot.chat import start_chat


if __name__ == "__main__":
    start_chat()
