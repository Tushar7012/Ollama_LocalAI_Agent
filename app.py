import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vectors import retriever

st.set_page_config(
    page_title="Pizza Restaurant Assistant",
    page_icon="🍕",
    layout="wide"
)


@st.cache_resource
def load_llm_chain():
    model = OllamaLLM(model="llama3:8b") 
    
    template = """
    You are an expert in answering questions about a pizza restaurant.
    
    Here are some relevant reviews:
    {reviews}
    
    Here is the question to answer:
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain

chain = load_llm_chain()


st.title("🍕 Pizza Restaurant Assistant")
st.write("Ask any question about our restaurant and get answers based on customer reviews!")

question = st.text_input("Ask your question:", key="question_input")


if st.button("Get Answer") or question:
    if question:
        with st.spinner("Searching for relevant information..."):
           
            reviews = retriever.invoke(question)
            review_text = "\n".join([doc.page_content for doc in reviews])
            
        
            with st.expander("View retrieved reviews", expanded=False):
                for i, doc in enumerate(reviews, 1):
                    st.markdown(f"**Review {i}:**")
                    st.write(doc.page_content)
                    st.write(f"Rating: {doc.metadata.get('rating')}, Date: {doc.metadata.get('date')}")
                    st.divider()
        
        with st.spinner("Generating answer..."):
            result = chain.invoke({"reviews": review_text, "question": question})
            
        
        st.markdown("### Answer:")
        st.markdown(result)
    else:
        st.warning("Please enter a question first.")

# App info
with st.sidebar:
    st.header("About")
    st.write("""
    This app uses Retrieval-Augmented Generation (RAG) to answer questions about our pizza restaurant.
    It retrieves relevant customer reviews and uses them to generate accurate answers to your questions.
    """)
    
    st.subheader("How it works")
    st.write("""
    1. You ask a question about the restaurant
    2. The system finds the most relevant customer reviews
    3. An AI model generates an answer based on those reviews
    """)
    
    st.markdown("---")
    st.write("Powered by Langchain, Ollama, and Streamlit")