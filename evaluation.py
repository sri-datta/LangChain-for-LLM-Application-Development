from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

# Load document
with open("docs/langchain_info.txt", "r") as f:
    document = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.create_documents([document])
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Chain 1 - Q&A chain
qa_prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the context below.\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)

qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | qa_prompt
    | chat
    | StrOutputParser()
)

# Chain 2 - Eval chain
eval_prompt = ChatPromptTemplate.from_template(
    "You are an evaluator. Given a question, the correct answer, and a student's answer, "
    "decide if the student's answer is correct.\n\n"
    "Question: {question}\n"
    "Correct Answer: {expected}\n"
    "Student Answer: {answer}\n\n"
    "Reply with only: CORRECT or INCORRECT"
)

eval_chain = eval_prompt | chat | StrOutputParser()

# Full automated chain - runs Q&A then grades the answer
def run_eval(example):
    question = example["question"]
    expected = example["expected"]

    # Chain 1 - get answer
    answer = qa_chain.invoke(question)

    # Chain 2 - grade the answer
    grade = eval_chain.invoke({
        "question": question,
        "expected": expected,
        "answer": answer
    }).strip()

    return {"question": question, "expected": expected, "answer": answer, "grade": grade}

# Wrap into a runnable chain
full_eval_chain = RunnableLambda(run_eval)

if __name__ == "__main__":
    print("=== Running Evaluation ===")
    print("Type your questions and expected answers. Type 'done' to finish.\n")

    examples = []
    while True:
        question = input("Enter question (or 'done' to start grading): ")
        if question.lower() == "done":
            break
        expected = input("Enter your expected answer: ")
        examples.append({"question": question, "expected": expected})
        print()

    if not examples:
        print("No questions entered.")
    else:
        print("\n=== Grading ===\n")
        results = []

        for example in examples:
            result = full_eval_chain.invoke(example)
            results.append(result["grade"])

            print(f"Q        : {result['question']}")
            print(f"Expected : {result['expected']}")
            print(f"Got      : {result['answer']}")
            print(f"Grade    : {result['grade']}")
            print("-" * 60)

        correct = results.count("CORRECT")
        print(f"\n=== Score: {correct}/{len(results)} correct ===")
