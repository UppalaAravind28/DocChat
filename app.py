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


# --- NEW: Function to extract text from various file types ---
def get_text_from_file(file):
    file_extension = os.path.splitext(file.name)[1].lower()

    if file_extension == ".pdf":
        pdf_reader = PdfReader(file)
        return "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

    elif file_extension in [".xlsx", ".xls"]:
        import pandas as pd
        df_dict = pd.read_excel(file, sheet_name=None)
        full_text = ""
        for sheet_name, df in df_dict.items():
            full_text += f"\n\nSheet: {sheet_name}\n"
            full_text += df.to_string(index=False)
        return full_text

    elif file_extension == ".csv":
        import pandas as pd
        df = pd.read_csv(file)
        return df.to_string(index=False)

    elif file_extension == ".docx":
        from docx import Document
        doc = Document(file)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


# --- Modified: Accept all supported file types ---
def get_pdf_text(files):
    raw_text = ""
    for file in files:
        try:
            raw_text += get_text_from_file(file)
        except Exception as e:
            st.warning(f"Could not process file: {file.name} - {str(e)}")
    return raw_text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.session_state.vectorstore_loaded = True


def get_conversational_chain():
    prompt_template = """
    You are an AI assistant DocBot designed to answer questions based on the provided document content. 
    Respond in a natural, conversational tone — as a helpful assistant would.

    If the context does not contain enough information to answer the question, respond with:
    "The answer is not available in the context."

    Follow these rules:
    - Always base your answer strictly on the context.
    - If the question requires multiple steps, explain each clearly.
    - Use bullet points or numbered lists where appropriate.
    - Keep your tone professional but easy to understand.

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
        # Greeting Detection
        greeting_keywords = ["hi", "hello", "hey", "good morning", "good afternoon", "is anyone there"]
        is_greeting = any(word.lower() in user_question.lower() for word in greeting_keywords)

        st.session_state.conversation.append(("user", user_question))

        if is_greeting:
            bot_response = "Hello! 👋 I'm DocBot, here to help you explore your documents. How can I assist you today?"
        else:
            with st.spinner("Thinking..."):
                bot_response = get_response(user_question)

        st.session_state.conversation.append(("bot", bot_response))
    st.session_state.widget_input = ""


def main():
    st.set_page_config("Chat PDF using Gemini", layout="centered")
    st.markdown("<h1 style='text-align:center;'>📄 Chat with Docs using DocBot 💬</h1>", unsafe_allow_html=True)

    # Sidebar for uploading files
    with st.sidebar:
        st.title("📚 Menu:")
        uploaded_files = st.file_uploader(
            "Upload PDF, Excel, CSV, or Word files",
            type=["pdf", "xlsx", "xls", "csv", "docx"],
            accept_multiple_files=True
        )
        if st.button("Process Files"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Extracting text and building index..."):
                    raw_text = get_pdf_text(uploaded_files)
                    if not raw_text.strip():
                        st.error("Could not extract text from the files. Make sure they are readable.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")

    # Main chat interface
    if st.session_state.vectorstore_loaded:
        st.markdown("### 💭 Ask a Question about your Documents")

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
        st.info("👈 Please upload and process a file first.")

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
        word-wrap: break-word;
        line-height: 1.5;
        font-size: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        max-width: 75%;
    }

    .chat-message.user {
        align-self: flex-end;
        background-color: #d1ecf1;
        color: #040404;
        border: 1px solid #bee5eb;
        margin-left: auto;
        text-align: right;
        width: fit-content;
        margin-bottom: 1rem;
        
    }

    .chat-message.bot {
        align-self: flex-start;
        background-color: #e2f0d9;
        color: #040404;
        border: 1px solid #c3e6cb;
        width: fit-content;
        margin-bottom: 1rem;
        margin-right: auto;
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
        background-color: #39547E;
        padding: 12px 20px;
        z-index: 999;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        border-top: 1px solid #dee2e6;
    }

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