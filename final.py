import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from operator import itemgetter
import os

# Configure page
st.set_page_config(page_title="RAG PDF Application", page_icon="📄")

# Title for the app
st.title("RAG Application for Documents (PDF)")

# Initialize session state for question history
if 'question_logger' not in st.session_state:
    st.session_state.question_logger = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'chain' not in st.session_state:
    st.session_state.chain = None

# File uploader to accept PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_pdf_path = "uploaded_pdf.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Check if we need to reload the document
    if st.session_state.vectorstore is None:
        with st.spinner("Processing PDF..."):
            try:
                # Load and split PDF using PyPDFLoader
                loader = PyPDFLoader(temp_pdf_path)
                pages = loader.load_and_split()
                
                if not pages:
                    st.error("No content found in the PDF file.")
                else:
                    # Initialize the model and embeddings
                    MODEL = "llama3"
                    model = Ollama(model=MODEL)
                    embeddings = OllamaEmbeddings(model=MODEL)
                    
                    # Create FAISS vectorstore
                    st.session_state.vectorstore = FAISS.from_documents(pages, embeddings)
                    retriever = st.session_state.vectorstore.as_retriever()
                    
                    # Create parser
                    parser = StrOutputParser()
                    
                    # Create prompt template
                    template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
                    prompt = PromptTemplate.from_template(template)
                    
                    # Create the RAG chain
                    st.session_state.chain = (
                        {
                            "context": itemgetter("question") | retriever,
                            "question": itemgetter("question"),
                        }
                        | prompt
                        | model
                        | parser
                    )
                    
                    st.success(f"PDF processed successfully! Found {len(pages)} pages.")
                    
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.session_state.vectorstore = None
                st.session_state.chain = None

# Question input section
if st.session_state.chain is not None:
    st.markdown("---")
    st.header("Ask Questions")
    
    # Input box for the user to type a question
    user_input = st.text_input("Enter your question:", key="question_input")
    
    # Button to submit question
    if st.button("Ask Question") or (user_input and user_input not in st.session_state.question_logger):
        if user_input:
            with st.spinner("Generating answer..."):
                try:
                    # Get answer from the chain
                    answer = st.session_state.chain.invoke({'question': user_input})
                    
                    # Store question in history
                    st.session_state.question_logger.append({
                        'question': user_input,
                        'answer': answer
                    })
                    
                    # Display the response
                    st.markdown("### Answer:")
                    st.write(answer)
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
        else:
            st.warning("Please enter a question.")
    
    # Display question history
    if st.session_state.question_logger:
        st.markdown("---")
        st.header("Question History")
        
        for idx, qa in enumerate(reversed(st.session_state.question_logger), 1):
            with st.expander(f"Q{len(st.session_state.question_logger) - idx + 1}: {qa['question']}"):
                st.markdown("**Answer:**")
                st.write(qa['answer'])
    
    # Generate follow-up questions button
    if st.session_state.question_logger:
        st.markdown("---")
        if st.button("Generate Follow-up Questions"):
            with st.spinner("Generating follow-up questions..."):
                try:
                    last_question = st.session_state.question_logger[-1]['question']
                    followup_prompt = f"Based on the question '{last_question}', generate 3 relevant follow-up questions. List them clearly."
                    followup_response = st.session_state.chain.invoke({'question': followup_prompt})
                    
                    st.markdown("### Suggested Follow-up Questions:")
                    st.write(followup_response)
                    
                except Exception as e:
                    st.error(f"Error generating follow-up questions: {str(e)}")
                    
    # Reset button
    st.markdown("---")
    if st.button("Clear History and Reset"):
        st.session_state.question_logger = []
        st.session_state.vectorstore = None
        st.session_state.chain = None
        if os.path.exists("uploaded_pdf.pdf"):
            os.remove("uploaded_pdf.pdf")
        st.rerun()
        
else:
    if uploaded_file is not None:
        st.info("Processing document... Please wait.")
    else:
        st.info("👆 Please upload a PDF file to get started.")
