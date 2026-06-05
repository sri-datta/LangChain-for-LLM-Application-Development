from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(temperature=0)

# Define tools the agent can use
@tool
def calculator(expression: str) -> str:
    """Use this to solve math calculations. Input should be a math expression like '25 * 4' or '100 / 5'."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def word_counter(text: str) -> str:
    """Use this to count the number of words in a text."""
    count = len(text.split())
    return f"The text has {count} words."

@tool
def uppercase(text: str) -> str:
    """Use this to convert text to uppercase."""
    return text.upper()

@tool
def reverse_text(text: str) -> str:
    """Use this to reverse a text string."""
    return text[::-1]

@tool
def character_counter(text: str) -> str:
    """Use this to count the number of characters in a text."""
    return f"The text has {len(text)} characters."

@tool
def celsius_to_fahrenheit(celsius: str) -> str:
    """Use this to convert Celsius temperature to Fahrenheit."""
    c = float(celsius)
    f = (c * 9/5) + 32
    return f"{c}°C = {f}°F"

@tool
def fahrenheit_to_celsius(fahrenheit: str) -> str:
    """Use this to convert Fahrenheit temperature to Celsius."""
    f = float(fahrenheit)
    c = (f - 32) * 5/9
    return f"{f}°F = {round(c, 2)}°C"

@tool
def get_current_date(dummy: str = "") -> str:
    """Use this to get today's current date. No input needed."""
    from datetime import date
    return f"Today's date is {date.today().strftime('%B %d, %Y')}"

@tool
def get_weather(city: str) -> str:
    """Use this to get the current weather for a city. Input should be a city name like 'New York' or 'London'."""
    import urllib.request
    url = f"https://wttr.in/{city.replace(' ', '+')}?format=3"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.read().decode("utf-8").strip()
    except Exception as e:
        return f"Could not fetch weather for {city}: {e}"

tools = [calculator, word_counter, uppercase, reverse_text,
         character_counter, celsius_to_fahrenheit, fahrenheit_to_celsius,
         get_current_date, get_weather]

# Create agent
agent = create_agent(chat, tools)

if __name__ == "__main__":
    print("=== LangChain Agent ===")
    print("Available tools: calculator, word_counter, uppercase, reverse_text,")
    print("                 character_counter, celsius_to_fahrenheit, fahrenheit_to_celsius, get_current_date\n")

    while True:
        question = input("Ask me anything (or 'exit'): ")
        if question.lower() == "exit":
            break

        result = agent.invoke({"messages": [("human", question)]})

        # Print agent's thinking steps
        for message in result["messages"]:
            if message.type == "tool":
                print(f"[Tool used: {message.name} → {message.content}]")

        # Final answer is the last AI message
        final = result["messages"][-1].content
        print(f"\nFinal Answer: {final}\n")
