from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(temperature=0)

# Memory is just a list of messages
history = [
    SystemMessage(content="You are a helpful assistant.")
]

def chat_with_memory(user_input):
    history.append(HumanMessage(content=user_input))
    response = chat.invoke(history)
    history.append(AIMessage(content=response.content))
    return response.content

if __name__ == "__main__":
    # Turn 1
    print("=== Turn 1 ===")
    print(chat_with_memory("Hi, my name is Sri!"))

    # Turn 2
    print("\n=== Turn 2 ===")
    print(chat_with_memory("What is 1+1?"))

    # Turn 3 - test if it remembers the name
    print("\n=== Turn 3 ===")
    print(chat_with_memory("What is my name?"))

    # Print full memory
    print("\n=== Full Memory ===")
    for message in history:
        print(f"{message.type}: {message.content}")
