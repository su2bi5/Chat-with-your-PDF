import os
import shutil
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox

import boto3
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_aws import BedrockLLM

# Initialize AWS Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Titan Embedding Model
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

# Data ingestion function
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Create and save FAISS vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# Initialize Llama 3 model
def get_llama3_llm():
    llm = Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock,
        model_kwargs={"max_gen_len": 512}
    )
    return llm

# Define prompt template
prompt_template = """
Human: Use the following context to answer the question. 
Provide a concise yet detailed response of at least 250 words. 
If you don't know the answer, say that you don't know.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Retrieve response from Llama 3
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']


# GUI Class
class RAGApp:
    def __init__(self, master):
        self.master = master
        master.title("Chat with PDF (AWS Bedrock - Llama 3)")

        self.llm = None
        self.faiss_index = None

        # GUI Layout
        self.frame = tk.Frame(master)
        self.frame.pack(padx=10, pady=10)

        self.select_btn = tk.Button(self.frame, text="Select PDF File(s)", command=self.select_pdfs)
        self.select_btn.grid(row=0, column=0, sticky='w', pady=5)

        self.ingest_btn = tk.Button(self.frame, text="Create/Update Vector Store", command=self.create_vector_store)
        self.ingest_btn.grid(row=0, column=1, sticky='w', padx=10, pady=5)

        self.query_label = tk.Label(self.frame, text="Enter your question:")
        self.query_label.grid(row=1, column=0, columnspan=2, sticky='w')

        self.query_entry = tk.Entry(self.frame, width=80)
        self.query_entry.grid(row=2, column=0, columnspan=2, pady=5)

        self.ask_btn = tk.Button(self.frame, text="Ask", command=self.process_query)
        self.ask_btn.grid(row=3, column=0, columnspan=2, pady=5)

        self.output_box = scrolledtext.ScrolledText(self.frame, width=100, height=25, wrap=tk.WORD)
        self.output_box.grid(row=4, column=0, columnspan=2, pady=10)

        self.load_faiss_index()

    def log(self, message):
        self.output_box.insert(tk.END, message + "\n")
        self.output_box.see(tk.END)

    def select_pdfs(self):
        files = filedialog.askopenfilenames(
            title="Select PDF file(s)",
            filetypes=[("PDF files", "*.pdf")]
        )
        if not files:
            return

        os.makedirs("data", exist_ok=True)

        for file_path in files:
            filename = os.path.basename(file_path)
            destination = os.path.join("data", filename)
            try:
                shutil.copy(file_path, destination)
                self.log(f"Added: {filename}")
            except Exception as e:
                self.log(f"Error adding {filename}: {e}")

    def create_vector_store(self):
        def run_ingestion():
            try:
                self.log("Processing selected PDFs...")
                docs = data_ingestion()
                get_vector_store(docs)
                self.log("Vector store created/updated successfully.")
                self.load_faiss_index()
            except Exception as e:
                messagebox.showerror("Error", str(e))

        threading.Thread(target=run_ingestion).start()

    def load_faiss_index(self):
        try:
            if not os.path.exists("faiss_index/index.faiss"):
                self.log("Vector store not found. Please create it first.")
                return
            self.log("Loading vector store...")
            self.faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            self.llm = get_llama3_llm()
            self.log("Vector store and LLM loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load vector store: {e}")

    def process_query(self):
        query = self.query_entry.get()
        if not query:
            messagebox.showwarning("Input Required", "Please enter a question.")
            return

        def run_query():
            try:
                self.log(f"\nUser: {query}")
                self.log("Processing your query...")
                response = get_response_llm(self.llm, self.faiss_index, query)
                self.log("\nLlama 3 Response:\n" + response + "\n")
            except Exception as e:
                messagebox.showerror("Error", f"Query failed: {e}")

        threading.Thread(target=run_query).start()


# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()
