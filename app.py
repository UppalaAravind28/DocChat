import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.session_state.vectorstore_loaded = True


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say: "The answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def get_response(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


def clear_input():
    user_question = st.session_state.widget_input.strip()
    if user_question:
        st.session_state.conversation.append(("user", user_question))
        with st.spinner("Thinking..."):
            response = get_response(user_question)
        st.session_state.conversation.append(("bot", response))
    st.session_state.widget_input = ""


def main():
    st.set_page_config("Chat PDF using Gemini", layout="centered")
    st.markdown("<h1 style='text-align:center;'>📄 Chat with PDF using Gemini 💬</h1>", unsafe_allow_html=True)

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title("📚 Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and click on 'Process'", accept_multiple_files=True)
        if st.button("Process PDF(s)"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Extracting text and building index..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("Could not extract text from the PDF(s). Make sure they are text-based (not scanned images).")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")

    # Main chat interface
    if st.session_state.vectorstore_loaded:
        st.markdown("### 💭 Ask a Question about your PDF")

        # Scrollable chat container
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for role, message in st.session_state.conversation:
                css_class = "user" if role == "user" else "bot"
                st.markdown(f"<div class='chat-message {css_class}'>{message}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Fixed input box at the bottom
        st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
        st.text_input("Your question:", key="widget_input", placeholder="Type your question here....", on_change=clear_input)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Please upload and process a PDF file first.")

    # --- CSS Styles ---
    st.markdown("""
    <style>
    /* Global layout */
    body {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f8f9fa;
        margin: 0;
        padding: 0;
    }

    h1 {
        color: #343a40;
    }

    /* Chat container */
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 10px 20px;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    /* Chat message bubbles */
    .chat-message {
        padding: 12px 16px;
        border-radius: 20px;
        max-width: 75%;
        word-wrap: break-word;
        line-height: 1.5;
        font-size: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .chat-message.user, .chat-message.bot{
        width: fit-content;
        padding: 1rem;
        border-radius: 12px;
        argin-bottom: 1rem;
        max-width: 75%;
        line-height: 1.5;
    }

    .chat-message.user {
        align-self: flex-end;
        background-color: #d1ecf1;
        color: #040404;
        margin-left: auto;
        text-align: right;
        border: 1px solid #bee5eb;
    }

    .chat-message.bot {
        align-self: flex-start;
        background-color: #e2f0d9;
        color: #040404;
        margin-right: auto;
        border: 1px solid #c3e6cb;
    }

    /* Input area */
    .stTextInput > div {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #ced4da;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        transition: all 0.3s ease-in-out;
    }

    .stTextInput input {
        font-size: 16px;
        padding: 6px;
        border: none;
        outline: none;
        background: none;
        width: 100%;
        color: #FCFCFC;
    }

    .stTextInput:focus-within {
        border-color: #80bdff;
        box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
    }

    /* Sticky input bar */
    .fixed-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 12px 20px;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.08);
        border-top: 1px solid #dee2e6;
        z-index: 999;
    }

    /* Responsive adjustments */
    @media screen and (max-width: 768px) {
        .chat-message {
            max-width: 90%;
        }

        .fixed-input {
            padding: 10px 16px;
        }
    }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()