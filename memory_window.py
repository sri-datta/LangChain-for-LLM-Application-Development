from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(temperature=0)

# Only remember last K conversation turns (1 turn = 1 human + 1 AI message)
K = 2

history = []
system_message = SystemMessage(content="You are a helpful assistant.")

def chat_with_window_memory(user_input):
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=""))  # placeholder

    # Keep only last K turns (K*2 messages)
    recent = history[-(K * 2):]

    # Build messages: system + recent history (excluding last placeholder)
    messages = [system_message] + recent[:-1] + [HumanMessage(content=user_input)]

    response = chat.invoke(messages)

    # Replace placeholder with actual response
    history[-1] = AIMessage(content=response.content)

    return response.content

if __name__ == "__main__":
    print("=== Turn 1 ===")
    print(chat_with_window_memory("Hi, my name is Sri!"))

    print("\n=== Turn 2 ===")
    print(chat_with_window_memory("I live in New York."))

    print("\n=== Turn 3 ===")
    print(chat_with_window_memory("My favourite food is pizza."))

    # This is turn 4 — Turn 1 (name) is now outside the window
    print("\n=== Turn 4 - Does it remember my name? ===")
    print(chat_with_window_memory("What is my name?"))

    print("\n=== Turn 4 - Does it remember my city? ===")
    print(chat_with_window_memory("Which city do I live in?"))

    print("\n=== Messages in Window (last 2 turns) ===")
    for msg in history[-(K * 2):]:
        print(f"{msg.type}: {msg.content}")
