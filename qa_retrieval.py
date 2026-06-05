from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough

_ = load_dotenv(find_dotenv())

chat = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

# Step 1 - Load document from file
file_path = input("Enter path to your document (e.g. docs/langchain_info.txt): ")
with open(file_path, "r") as f:
    document = f.read()

print(f"\nLoaded document ({len(document)} characters)")

# Step 2 - Split document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.create_documents([document])

print(f"=== Document split into {len(chunks)} chunks ===")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.page_content[:80]}...")

# Step 2 - Store chunks in vector database
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Step 3 - Build Q&A chain
prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the context below.\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)

qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat
    | StrOutputParser()
)

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question about LangChain (or 'exit'): ")
        if question.lower() == "exit":
            break
        answer = qa_chain.invoke(question)
        print(f"Answer: {answer}")
