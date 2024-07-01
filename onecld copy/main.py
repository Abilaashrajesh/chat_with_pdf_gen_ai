from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import streamlit as st
import re
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    # print(text)
    return text

def clean_text(text):
    # Define regex patterns
    url_pattern = re.compile(r'https?://\S+|www\.\S+|\(\S+\)')
    bracketed_number_pattern = re.compile(r'\[\d+\]')
    unwanted_symbols_pattern = re.compile(r'[^\w\s]')
    text = url_pattern.sub('', text)
    text = bracketed_number_pattern.sub('', text)
    text = unwanted_symbols_pattern.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_chain():
    prompt_template = """
    You are a chatbot.You want to behave as  a chatbot and nswer the question from the provided context.Take the user input and try to give the anwer rerlevant to it .If the answer is not innprovided context just say, "answer is not available in the context", don't provide the wrong answer.Give the output in simple formatted sentence. \n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

def main():
    pdf_path = 'pdf_data_2.pdf'  # Path to your local PDF file
    raw_text=get_text(pdf_path)
    cln_text=clean_text(raw_text)
    text_chunks = get_chunks(cln_text)
    get_vector(text_chunks)
    st.title("Q & A ChatBot")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    user_qns = st.text_input("How can i assist you ... ", key="input")
    if st.button("Send"):
        if user_input:
            # Append the user question and chatbot response to conversation
            st.session_state.conversation.append(f"You: {user_qns}")
            response = user_input(user_qns)
            st.write(response['output_text'])
            
    # Display the conversation
    # for msg in st.session_state.conversation:
    #     st.write(msg)

if __name__ == "__main__":
    main()
