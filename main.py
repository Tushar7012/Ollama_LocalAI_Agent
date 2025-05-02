from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectors import retriever

model = OllamaLLM(model="llama3:8b")  # Adjust this to your Ollama model name

template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews:
{reviews}

Here is the question to answer:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question.lower() == "q":
        break

    reviews = retriever.invoke(question)
    review_text = "\n".join([doc.page_content for doc in reviews])

    result = chain.invoke({"reviews": review_text, "question": question})
    print(result)
