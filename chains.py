from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(temperature=0)
parser = StrOutputParser()

# Chain 1 - Translate review to English
translate_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to English:\n{review}"
)

# Chain 2 - Summarize the English review
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize the following review in 1 sentence:\n{english_review}"
)

# Chain 3 - Detect sentiment
sentiment_prompt = ChatPromptTemplate.from_template(
    "What is the sentiment of this review? (positive/negative/neutral)\n{summary}"
)

# Build individual chains
translate_chain = translate_prompt | chat | parser
summary_chain = summary_prompt | chat | parser
sentiment_chain = sentiment_prompt | chat | parser

# Sequential chain - output of each feeds into the next
def run_sequential_chain(review):
    print("=== Original Review ===")
    print(review)

    english = translate_chain.invoke({"review": review})
    print("\n=== Step 1: Translated to English ===")
    print(english)

    summary = summary_chain.invoke({"english_review": english})
    print("\n=== Step 2: Summary ===")
    print(summary)

    sentiment = sentiment_chain.invoke({"summary": summary})
    print("\n=== Step 3: Sentiment ===")
    print(sentiment)

if __name__ == "__main__":
    spanish_review = """
    Este soplador de hojas es increíble. Llegó en dos días y
    mi esposa quedó muy feliz con el regalo. El precio es un
    poco alto pero vale la pena.
    """
    run_sequential_chain(spanish_review)
