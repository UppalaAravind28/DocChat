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
from docx import Document
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader
from io import StringIO


# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False
if "history" not in st.session_state:
    st.session_state.history = []

def get_text_from_file(file):
    file_extension = os.path.splitext(file.name)[1].lower()

    if file_extension == ".pdf":
        pdf_reader = PdfReader(file)
        return "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

    elif file_extension in [".xlsx", ".xls"]:
        df_dict = pd.read_excel(file, sheet_name=None)
        full_text = ""
        for sheet_name, df in df_dict.items():
            full_text += f"\n\nSheet: {sheet_name}\n"
            full_text += df.to_string(index=False)
        return full_text

    elif file_extension == ".csv":
        df = pd.read_csv(file)
        return df.to_string(index=False)

    elif file_extension == ".docx":
        doc = Document(file)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def get_web_text(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        st.error(f"Error fetching content from URL: {str(e)}")
        return ""


def get_raw_text(files=None, url=None):
    raw_text = ""

    if files:
        for file in files:
            try:
                raw_text += get_text_from_file(file)
            except Exception as e:
                st.warning(f"Could not process file: {file.name} - {str(e)}")

    if url.strip():
        with st.spinner("Fetching content from website..."):
            web_text = get_web_text(url)
            raw_text += web_text

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

    Use the following rules:
    - Always base your answer strictly on the context.
    - Format answers with bullet points, numbered lists, tables, and clear paragraphs.
    - When appropriate, use emojis to make the response more engaging.
    - Avoid markdown syntax if possible, but use it when needed for structure.
    - Keep your tone professional but friendly.
    - If the output contains data, organize it into table format for better presentation.

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
            bot_response = get_response(user_question)
        st.session_state.conversation.append(("bot", bot_response))
    st.session_state.widget_input = ""


def main():
    st.set_page_config("Chat PDF using Gemini", layout="centered")
    st.markdown("<h1 style='text-align:center;'>📄 Chat with Docs using DocBot 💬</h1>", unsafe_allow_html=True)

    # Sidebar Navigation Panel
    with st.sidebar:
        st.title("📚 Menu:")

        # --- File Upload Section ---
        st.markdown("### 📁 Upload Files")
        uploaded_files = st.file_uploader(
            "Upload PDF, Excel, CSV, or Word files",
            type=["pdf", "xlsx", "xls", "csv", "docx"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        # --- Web Link Input ---
        st.markdown("### 🔗 Paste a Website Link")
        url_input = st.text_input("Enter any public link:", placeholder="https://example.com",  key="url_input")

        # --- Process Button ---
        if st.button("Process Files & Link"):
            if not uploaded_files and not url_input.strip():
                st.warning("Please upload at least one file OR enter a link.")
            else:
                with st.spinner("Extracting text and building index..."):
                    raw_text = get_raw_text(uploaded_files, url_input)
                if not raw_text.strip():
                    st.error("No usable content found in files or link.")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Done! Ready to answer questions.")


    # Main Chat Interface
    if st.session_state.vectorstore_loaded:
        st.markdown("### 💭 Ask a Question about your Documents or Content")

        # Scrollable chat container
        with st.container():
            for role, message in st.session_state.conversation:
                with st.chat_message(role):
                    if role == "user":
                        st.markdown(f"🧑‍💻 {message}")
                    else:
                        st.markdown(message.replace("Answer:", "🤖 **DocBot:**"))

        # Fixed input box + buttons below it
        st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
        st.text_input("Your question:", key="widget_input", placeholder="Type your question here....", on_change=clear_input)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📥 Download Chat", use_container_width=True):
                if st.session_state.conversation:
                    conversation_str = "\n\n".join([f"{role}: {msg}" for role, msg in st.session_state.conversation])
                    st.download_button("Download", data=conversation_str, file_name="chat_history.txt", mime="text/plain")
                else:
                    st.info("No conversation history to download yet.")

        with col2:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.conversation = []
                st.session_state.vectorstore_loaded = False

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Please upload files or paste a link and click 'Process Files & Link'")
        # --- How to Use Guide ---
        st.markdown("---")
        with st.expander("❓ How to Use This Bot"):
            st.markdown("""
            - Click on the side menu. Having a symbol like this  >  top of the left side.
            - Upload PDFs, Excel, Word docs, or CSV files
            - Or paste a public website link
            - Click **Process Files & Link**
            - Ask any question!

            ### Supported Formats
            - ✅ PDF, ✅ Excel (.xlsx/.xls), ✅ CSV, ✅ Word (.docx)
            - 🔗 Public websites via link
            """)


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
        text-align: right;
    }

    .chat-message.bot {
        align-self: flex-start;
        background-color: #e2f0d9;
        color: #040404;
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
        background-color: #39547E;
        padding: 12px 20px;
        z-index: 999;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        border-top: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)


def get_processed_text():
    return getattr(st.session_state, "processed_text", "No text processed yet")


if __name__ == "__main__":
    main()