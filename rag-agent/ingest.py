import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

DATA_DIR = "data"
DB_DIR = "chroma_db"

def load_documents():
    documents = []
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            continue

        if filename.endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8") as f:
                documents.append(
                    Document(
                        page_content=f.read(),
                        metadata={"source": filename}
                    )
                )
    return documents

def main():
    documents = load_documents()
    if not documents:
        raise RuntimeError("No .txt or .md files found in data/")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="knowledge_base",
    )

    print(f"Indexed {len(documents)} documents into {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
