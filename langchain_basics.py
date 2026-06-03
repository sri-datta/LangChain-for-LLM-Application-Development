import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(temperature=0)

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = "American English in a calm and respectful tone"

prompt_template = ChatPromptTemplate.from_template(
    "Translate the text delimited by triple backticks into {style}. text: ```{text}```"
)

messages = prompt_template.format_messages(style=style, text=customer_email)

print("---- Prompt ----")
print(messages[0].content)

response = chat.invoke(messages)

print("\n---- Response ----")
print(response.content)

# Reply to customer in a different language
service_reply = """
Hey there customer, the warranty does not cover \
cleaning expenses for your kitchen because it's \
your fault that you did not secure the blender lid \
before starting the blender. \
Tough luck! See ya!
"""

reply_style = "a polite tone that speaks in Spanish"

reply_messages = prompt_template.format_messages(
    style=reply_style,
    text=service_reply
)

print("\n---- Service Reply Prompt ----")
print(reply_messages[0].content)

reply_response = chat.invoke(reply_messages)

print("\n---- Translated Reply ----")
print(reply_response.content)
