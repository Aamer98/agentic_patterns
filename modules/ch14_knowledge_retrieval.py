import requests
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import weaviate
from weaviate.embedded import EmbeddedOptions

load_dotenv()

url = "https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/state_of_the_union.txt"
res = requests.get(url)

with open("sotu.txt", "w") as f:
    f.write(res.text)

text_loader = TextLoader("sotu.txt")
text = text_loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=5)
chunks = splitter.split_documents(text)

client = weaviate.Client(embedded_options = EmbeddedOptions)
vectorstore = Weaviate.from_documents(client=client, documents=chunks, embedding=OpenAIEmbeddings(), by_text=False )

retriever = vectorstore.as_retriever()

llm = ChatOpenAI(temperature=0)

class RAGGraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str


def retrieve_docs_node(state: RAGGraphState):
    """Retrieves documents relevant to the question from the vector store."""

    question = state['question']

    documents = retriever.invoke(question)

    return {'question': question, 'documents':documents, 'generation':""}


def generate_response_node(state: RAGGraphState):
    """Generates response based on question and context."""

    question = state['question']
    documents = state['documents']

    prompt_template = ChatPromptTemplate.from_template("""
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer: 
        """)
    
    rag_chain = prompt_template | llm | StrOutputParser

    context = "\n\n".join([doc.page_content for doc in documents])

    generation = rag_chain.invoke({'question': question, 'context': context})

    return {'question': question, 'documents': documents, 'generation': generation}


# Create Graph

workflow = StateGraph(RAGGraphState)

workflow.add_node("retreive", retrieve_docs_node)
workflow.add_node("generate", generate_response_node)

workflow.set_entry_point("retreive")

workflow.add_edge("retreive", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# --- 5. Run the RAG Application ---
if __name__ == "__main__":
    print("\n--- Running RAG Query ---")
    query = "What did the president say about Justice Breyer"
    inputs = {"question": query}
    for s in app.stream(inputs):
        print(s)
    print("\n--- Running another RAG Query ---")
    query_2 = "What did the president say about the economy?"
    inputs_2 = {"question": query_2}
    for s in app.stream(inputs_2):
        print(s)