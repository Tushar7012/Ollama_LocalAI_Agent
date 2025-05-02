from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import os
import pandas as pd

# Load dataset
df = pd.read_csv("D:/LangChain/Ollama_RAG/realistic_restaurant_reviews (1).csv")

# Define embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Vector DB path
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Load and format documents
if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        content = f"{row['cTitle']} {row['Review']}"
        metadata = {"rating": row["Rating"], "date": row["Date"]}
        document = Document(page_content=content, metadata=metadata)
        documents.append(document)
        ids.append(str(i))

# Create Chroma vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents if not already added
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Expose an invoke function for external use
def invoke(query: str):
    return retriever.invoke(query)
