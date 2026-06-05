from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(temperature=0)
parser = StrOutputParser()

# Individual subject chains
math_chain = (
    ChatPromptTemplate.from_template("You are a math expert. Answer this:\n{input}")
    | chat | parser
)

history_chain = (
    ChatPromptTemplate.from_template("You are a history expert. Answer this:\n{input}")
    | chat | parser
)

science_chain = (
    ChatPromptTemplate.from_template("You are a science expert. Answer this:\n{input}")
    | chat | parser
)

general_chain = (
    ChatPromptTemplate.from_template("Answer this question:\n{input}")
    | chat | parser
)

# Router chain - detects the subject
router_prompt = ChatPromptTemplate.from_template(
    "Which subject does this question belong to? Reply with only one word: math, history, science, or general.\n"
    "Question: {input}"
)
router_chain = router_prompt | chat | parser

# Route to the right chain based on subject
def route(question):
    subject = router_chain.invoke({"input": question}).strip().lower()
    print(f"[Routing to: {subject}]")

    if subject == "math":
        return math_chain.invoke({"input": question})
    elif subject == "history":
        return history_chain.invoke({"input": question})
    elif subject == "science":
        return science_chain.invoke({"input": question})
    else:
        return general_chain.invoke({"input": question})

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        print(route(question))
