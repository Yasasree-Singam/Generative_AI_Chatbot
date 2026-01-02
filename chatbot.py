import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Extract the text and break into chunks
if file is not None:
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text+=page.extract_text()
        #st.write(text)

    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " ", ""],
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

# generating embeddings

# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
# creating vector store - FAISS
    def build_vector_store(chunks):
        return FAISS.from_texts(chunks, embeddings)
    vector_store = build_vector_store(chunks)
    llm = ChatOllama(
        model="gemma3:4b",
        temperature=0
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions strictly based on the provided context from the PDF document."
         "Only answer questions using information from the context below."
         "If the question cannot be answered using the context, response with:'I can only answer questions related to the uploaded PDF document.'\n\n"
         "Context:\n{context}"),
         ("human","{question}")
    ])

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    #output results
    retriever = vector_store.as_retriever()

    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    user_question = st.text_input("Type your question here")
    # match = vector_store.similarity_search(user_question)

    if user_question:
        response = chain.invoke(user_question)
        st.write(response)

    


