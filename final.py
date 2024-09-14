import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.document_loaders.base import Document
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import fitz  # PyMuPDF for reading PDFs
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import PyPDFLoader
from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


st.title("RAG application for documents(pdf)")

# File uploader to accept PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split PDF using PyPDFLoader
loader = PyPDFLoader("uploaded_pdf.pdf")
pages = loader.load_and_split()

question_logger=[]

# Title for the app
st.title("Questions")

# Input box for the user to type something
user_input = st.text_input("Enter some text:")

# Temporary variable to store the input
temp_variable = user_input
question_logger.append(temp_variable)


# Display the stored input
st.write(f"Stored in temporary variable: {temp_variable}")

#initializing the model and embeddings 
MODEL="llama2"
model=Ollama(model=MODEL)
embeddings=OllamaEmbeddings()
#the vectorstore created will be stored in primary memory RAM when the program executes using DocArray
vectorstores=DocArrayInMemorySearch(pages, embedding=embeddings)
#using  the facbook ai similarity search(FAISS) for embeeding the document in a vectorstore
vectorstore = FAISS.from_documents(pages, embeddings)
retriever = vectorstore.as_retriever()
parser=StrOutputParser()

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
print(prompt.format(context="Here is some context", question="Here is a question"))

#creating the RAG chain
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
)
#displaying the response
questions = [temp_variable]
for question in questions:
    code = f'''
Question: {question}
Answer: {chain.invoke({'question': question})}
'''
    st.code(code, language="python")       
#displaying the follow up questions
response=chain.invoke("give me follow-up questions to the preevious question")
st.code(response,language="python")