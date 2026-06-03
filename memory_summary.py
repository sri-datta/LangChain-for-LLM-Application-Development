from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(temperature=0)

history = []
summary = ""

def summarize_history():
    if not history:
        return ""
    conversation_text = "\n".join(
        f"{msg.type}: {msg.content}" for msg in history
    )
    summary_prompt = [
        HumanMessage(content=(
            f"Summarize this conversation in 2-3 sentences:\n{conversation_text}"
        ))
    ]
    response = chat.invoke(summary_prompt)
    return response.content

def chat_with_summary_memory(user_input, summarize_after=3):
    global summary, history

    history.append(HumanMessage(content=user_input))

    # Build messages: system + summary of past + recent human message
    messages = [SystemMessage(content="You are a helpful assistant.")]
    if summary:
        messages.append(SystemMessage(content=f"Summary of conversation so far: {summary}"))
    messages.append(HumanMessage(content=user_input))

    response = chat.invoke(messages)
    history.append(AIMessage(content=response.content))

    # Summarize old history after every N turns
    if len(history) >= summarize_after * 2:
        print("\n[Summarizing old conversation...]\n")
        summary = summarize_history()
        history = []  # clear history, summary takes its place

    return response.content

if __name__ == "__main__":
    print("=== Turn 1 ===")
    print(chat_with_summary_memory("Hi, my name is Sri!"))

    print("\n=== Turn 2 ===")
    print(chat_with_summary_memory("I live in New York."))

    print("\n=== Turn 3 ===")
    print(chat_with_summary_memory("My favourite food is pizza."))

    # History is now summarized — does it still remember?
    print("\n=== Turn 4 - Does it remember my name? ===")
    print(chat_with_summary_memory("What is my name?"))

    print("\n=== Turn 5 - Does it remember my city? ===")
    print(chat_with_summary_memory("Which city do I live in?"))

    print("\n=== Current Summary ===")
    print(summary)
