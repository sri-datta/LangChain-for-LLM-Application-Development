from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
import requests

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(temperature=0)

@tool
def get_weather(city: str) -> str:
    """Use this to get the current weather for a city."""
    url = f"https://wttr.in/{city.replace(' ', '+')}?format=3"
    try:
        response = requests.get(url, timeout=5)
        return response.text.strip()
    except Exception as e:
        return f"Could not fetch weather: {e}"

@tool
def get_joke(dummy: str = "") -> str:
    """Use this to get a random joke. No input needed."""
    url = "https://official-joke-api.appspot.com/random_joke"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        return f"{data['setup']} ... {data['punchline']}"
    except Exception as e:
        return f"Could not fetch joke: {e}"

@tool
def get_dog_image(dummy: str = "") -> str:
    """Use this to get a random dog image URL. No input needed."""
    url = "https://dog.ceo/api/breeds/image/random"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        return f"Random dog image: {data['message']}"
    except Exception as e:
        return f"Could not fetch dog image: {e}"

@tool
def get_country_info(country: str) -> str:
    """Use this to get information about a country like population, capital, region."""
    url = f"https://restcountries.com/v3.1/name/{country}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()[0]
        name = data["name"]["common"]
        capital = data.get("capital", ["N/A"])[0]
        population = data.get("population", "N/A")
        region = data.get("region", "N/A")
        return f"{name} — Capital: {capital}, Population: {population:,}, Region: {region}"
    except Exception as e:
        return f"Could not fetch country info: {e}"

tools = [get_weather, get_joke, get_dog_image, get_country_info]

agent = create_agent(chat, tools)

if __name__ == "__main__":
    print("=== Agent with API Tools ===")
    print("Tools: weather, joke, dog image, country info\n")

    while True:
        question = input("Ask me anything (or 'exit'): ")
        if question.lower() == "exit":
            break

        result = agent.invoke({"messages": [("human", question)]})

        for message in result["messages"]:
            if message.type == "tool":
                print(f"[API called: {message.name} → {message.content}]")

        print(f"\nAnswer: {result['messages'][-1].content}\n")
