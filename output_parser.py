from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(temperature=0)

customer_review = """\
This leaf blower is pretty amazing. It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's birthday.\
I think my wife liked it as a birthday gift. \
However, the price is a bit high for what you get, \
so I think it's fair value for the money.
"""

# Define the structure of the output
class ReviewInfo(BaseModel):
    gift: bool = Field(description="Was the item purchased as a gift? Answer True or False.")
    delivery_days: int = Field(description="How many days did it take to deliver?")
    price_value: str = Field(description="Extract sentences about price or value.")

# Create the parser
output_parser = PydanticOutputParser(pydantic_object=ReviewInfo)
format_instructions = output_parser.get_format_instructions()

print("---- Format Instructions ----")
print(format_instructions)

# Build prompt with format instructions
prompt_template = ChatPromptTemplate.from_template(
    "For the following review, extract the requested information.\n"
    "{format_instructions}\n"
    "review: ```{review}```"
)

messages = prompt_template.format_messages(
    format_instructions=format_instructions,
    review=customer_review
)

response = chat.invoke(messages)

print("\n---- Raw Response ----")
print(response.content)

# Parse into a structured object
output = output_parser.parse(response.content)

print("\n---- Parsed Output ----")
print(output)
print("\ngift:", output.gift)
print("delivery_days:", output.delivery_days)
print("price_value:", output.price_value)
