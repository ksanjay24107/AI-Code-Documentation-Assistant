import os
import gradio as gr
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini Model
model = genai.GenerativeModel("gemini-2.5-flash")

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def explain_code(files, question):
    if not files:
        return "Please upload at least one code file."

    code = ""

    # Read uploaded files
    for file in files:
        with open(file.name, "r", encoding="utf-8", errors="ignore") as f:
            code += "\n\n"
            code += f.read()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.create_documents([code])

    # Create FAISS Vector DB
    db = FAISS.from_documents(docs, embedding_model)

    # Retrieve relevant chunks
    retrieved_docs = db.similarity_search(question, k=4)

    context = ""

    for doc in retrieved_docs:
        context += doc.page_content + "\n\n"

    prompt = f"""
You are an expert software engineer.

Answer ONLY from the given code.

Code:
{context}

Question:
{question}
"""

    response = model.generate_content(prompt)

    return response.text


demo = gr.Interface(
    fn=explain_code,
    inputs=[
        gr.File(file_count="multiple", label="Upload Code Files"),
        gr.Textbox(label="Ask a Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="AI Code Documentation Assistant (RAG)"
)

demo.launch()