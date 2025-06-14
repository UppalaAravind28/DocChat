

---

# 📄 DocBot: Chat with PDFs, Excel, Word Docs & Web Links

DocBot is an **AI-powered document assistant** that lets you upload documents or paste web links and ask questions — it will find answers using natural language processing and semantic search.

Built using:
- 🔐 Google Gemini LLM (via LangChain)
- 🧠 FAISS vector store for fast similarity search
- 🌐 Streamlit UI with sticky input bar
- 📁 Support for multiple file types + web scraping

Perfect for students, researchers, professionals, and developers who want to get insights from complex documents quickly and accurately.
- DocChat Live link : (https://docchatt.streamlit.app/)

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| ✅ Upload multiple file types | PDF, Excel, Word, CSV |
| ✅ Paste any public link | Scrapes and indexes content |
| 🤖 Gemini-powered Q&A | Uses context from documents |
| 🧠 FAISS vector store | Fast semantic search |
| 💬 Streamlit chat UI | Scrollable chat + fixed input bar |
| 🌐 Supports modern websites | Wikipedia, Indiabix, news pages |

---

## 📦 Supported File Types

| Format | Description |
|--------|-------------|
| ✅ PDF | Extract text from multi-page PDFs |
| ✅ Excel (.xlsx / .xls) | Read full spreadsheets |
| ✅ CSV | View tabular data as plain text |
| ✅ Word (.docx) | Extract paragraphs from Word docs |
| ✅ Web Pages | Scrape content from websites using link |

---

## 🛠️ How It Works

### 1. Upload Files or Paste a Link  
Users can upload one or more files (PDF/Excel/Word/CSV) or paste a **public website URL**

### 2. Process Text  
All content is:
- Split into chunks
- Converted into embeddings using Gemini
- Stored in FAISS vector store

### 3. Ask Questions  
Ask anything related to the uploaded files or pasted links

### 4. Get Answers  
Gemini reads from the most relevant parts of the content and gives a detailed, accurate response.

If the answer isn't found → bot says:
> "The answer is not available in the context."

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|--------|
| [Streamlit](https://streamlit.io) | For building the interactive UI |
| [Google Gemini API](https://makersuite.google.com/) | For Q&A and embeddings |
| [FAISS](https://github.com/facebookresearch/faiss) | Fast similarity search |
| [LangChain](https://www.langchain.com) | For prompt templates and chains |
| [PyPDF2](https://pypdf2.readthedocs.io) | PDF text extraction |
| [Pandas](https://pandas.pydata.org/) | Excel & CSV processing |
| [python-docx](https://python-docx.readthedocs.io) | Word document support |
| [WebBaseLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/web_base) | Public web scraping |
| [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) | HTML parsing |
| [FAISS](https://github.com/facebookresearch/faiss) | Efficient vector storage and search |

---

## 📁 Folder Structure

```
DocChat/
│
├── app.py                  # Main application logic and Streamlit UI
├── requirements.txt        # List of Python dependencies
├── .env                    # Stores API keys securely
├── README.md               # Project documentation (this file)
└── LICENSE                 # MIT License information
```

---

## 📦 Dependencies

The project uses the following Python libraries:

### 🔧 Required Libraries

| Library | Purpose |
|--------|---------|
| [`streamlit`](https://streamlit.io) | For the interactive web UI |
| [`PyPDF2`](https://pypi.org/project/PyPDF2/) | Extract text from PDF files |
| [`langchain`](https://python.langchain.com/) | LLM framework & prompt management |
| [`langchain-google-genai`](https://python.langchain.com/docs/integrations/chat_models/google_generative_ai) | Integration with Google Gemini models |
| [`google-generativeai`](https://pypi.org/project/google-generativeai/) | Google Gemini API access |
| [`faiss-cpu`](https://github.com/facebookresearch/faiss) | Vector similarity search |
| [`python-dotenv`](https://pypi.org/project/python-dotenv/) | Load environment variables from `.env` file |
| [`pandas`](https://pandas.pydata.org/) | Read and process Excel & CSV files |
| [`openpyxl`](https://openpyxl.readthedocs.io/) | Support for `.xlsx` files |
| [`python-docx`](https://python-docx.readthedocs.io/) | Read Word (.docx) documents |
| [`beautifulsoup4`](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) | Web scraping HTML parsing |
| [`requests`](https://docs.python-requests.org/en/latest/) | HTTP requests for web scraping |
| [`langchain-community`](https://pypi.org/project/langchain-community/) | Extra LangChain tools and loaders |

### 🧩 Optional (for advanced features)

| Library | Use Case |
|--------|----------|
| `playwright` | For JavaScript-heavy websites |
| `weasyprint` | Generate PDFs from conversation |
| `streamlit-extras` | Additional Streamlit components |
| `pillow` | Image handling in UI |
| `tkinter` or `PyInstaller` | Build desktop app |

---

## 💾 Install Dependencies

Make sure you're inside the project folder:

```bash
cd DocChat
```

Then install all required packages:

```bash
pip install -r requirements.txt
```

If you’re using a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### ✅ Sample `requirements.txt`

```txt
streamlit==1.29.0
PyPDF2==3.0.1
langchain==0.1.16
langchain-google-genai==1.0.10
google-generativeai==0.3.1
faiss-cpu==1.7.0
python-dotenv==1.0.1
pandas==2.1.4
openpyxl==3.1.2
python-docx==0.8.11
beautifulsoup4==4.12.0
requests==2.31.0
langchain-community==0.0.15
```

---

## 🌐 Deployment Guide

You can deploy this app online using any of the following platforms:

### 1. [Streamlit Cloud](https://streamlit.io/cloud)

- Push your code to a public GitHub repo
- Go to [Streamlit Cloud Dashboard](https://share.streamlit.io/)
- Click **"New App"** → link your repo
- Set environment variables in **Settings > Secrets**

> ✅ Fastest way to host a Streamlit app for free
- DocChat Live link : (https://docchatt.streamlit.app/)

---

### 2. [Hugging Face Spaces](https://huggingface.co/spaces)

- Create a new Space with SDK: `Gradio` or `Streamlit`
- Upload your files:
  - `app.py`
  - `requirements.txt`
  - `.env` (optional)
- Hugging Face will auto-deploy your app

> 🧠 You can use `pip install -r requirements.txt` in your app's setup

---


```bash
docker build -t docbot .
docker run -p 8501:8501 docbot
```

Access your bot at:  
🔗 `http://localhost:8501`

---


> ⚠️ Tip: Test it locally before distributing

---

## 📄 License

This project is open-source under the **MIT License**.

See the full license in the `LICENSE` file.

---

## 🧑‍💻 Author

**Uppala Aravind**  
📧 Email: uppalaaravind28@gmail.com  
💼 LinkedIn: [https://www.linkedin.com/in/uppala-aravind-28-lin/](https://www.linkedin.com/in/uppala-aravind-28-lin/)  
🐙 GitHub: [UppalaAravind28](https://github.com/UppalaAravind28)

---

## 🤝 Contributing

Contributions are welcome! If you'd like to help improve DocBot, feel free to fork the repo and submit pull requests.

### Steps to Contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to your branch (`git push origin feature/new-feature`)
5. Open a Pull Request

Let’s keep improving this together!

---

## 📬 Contact

For questions, suggestions, or collaboration opportunities, feel free to reach out:

- 👨‍💻 GitHub: [https://github.com/your-username/DocChat](https://github.com/your-username/DocChat)
- 💼 LinkedIn: [https://www.linkedin.com/in/uppala-aravind-28-lin/](https://www.linkedin.com/in/uppala-aravind-28-lin/)
- 📧 Email: uppalaaravind28@gmail.com

---
### Thank You for Visiting! 🌟

Let’s work together to create impactful solutions and push the boundaries of technology. Explore my repositories, star them if you like, and don’t hesitate to reach out. I look forward to connecting with you! 😊

---
