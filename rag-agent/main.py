import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

load_dotenv()

DB_DIR = "chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory=DB_DIR,
    collection_name="knowledge_base",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class AgentState(TypedDict):
    question: str
    needs_retrieval: bool
    context: str
    sources: List[str]
    answer: str

def route(state: AgentState):
    question = state["question"].lower()

    # Simple, transparent routing policy:
    # knowledge-base questions trigger retrieval; greetings/general
    # conversational questions can be answered directly.
    retrieval_terms = [
        "according to", "document", "file", "policy", "project",
        "what does", "what is", "explain", "describe", "summarize"
    ]

    needs = any(term in question for term in retrieval_terms)
    return {"needs_retrieval": needs}

def retrieve(state: AgentState):
    docs = retriever.invoke(state["question"])

    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

    sources = sorted({
        doc.metadata.get("source", "unknown")
        for doc in docs
    })

    return {"context": context, "sources": sources}

def answer(state: AgentState):
    if state["needs_retrieval"]:
        prompt = f"""Answer the question using only the retrieved context below.
If the context does not contain enough information, say that clearly.

Question:
{state['question']}

Retrieved context:
{state.get('context', '')}
"""
    else:
        prompt = state["question"]

    response = llm.invoke(prompt)
    return {"answer": response.content}

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda state: "retrieve" if state["needs_retrieval"] else "answer",
        {"retrieve": "retrieve", "answer": "answer"},
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()

def main():
    if not os.path.exists(DB_DIR):
        print("Vector database not found. Run: python ingest.py")
        return

    app = build_graph()

    print("Agentic RAG Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        result = app.invoke({
            "question": question,
            "needs_retrieval": False,
            "context": "",
            "sources": [],
            "answer": "",
        })

        print(f"\nAssistant: {result['answer']}")

        if result.get("sources"):
            print("Sources:", ", ".join(result["sources"]))

        print()

if __name__ == "__main__":
    main()
