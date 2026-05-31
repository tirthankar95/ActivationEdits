from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langsmith import traceable
from typing import List
from uuid import uuid4
import pandas as pd 
import argparse
import logging
import dotenv
import os

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)
print(os.getenv("OPENAI_API_KEY"))
# -------------------------
# Utils
# -------------------------
def extract_text(document_path: str) -> pd.DataFrame:
    df = pd.read_csv(document_path)
    return df
# -------------------------
# Vector DB Base
# -------------------------
class VectorDB:
    def __init__(self, embedding_model, collection_name=None):
        self.e_model = embedding_model
        self.collection_name = collection_name
    def add_documents(self, documents: List[str]):
        raise NotImplementedError
    def similarity_search(self, query: str, topK: int = 3) -> List[str]:
        raise NotImplementedError
# -------------------------
# Chroma Implementation
# -------------------------
class ChromaDB(VectorDB):
    def __init__(self, embedding_model=None, collection_name=None):
        super().__init__(embedding_model, collection_name)
        # Initialize ChromaDB client here
        self.chroma_client = Chroma(collection_name=self.collection_name,
                                    embedding_function=self.e_model,
                                    persist_directory="./chroma_persistence")
    def add_documents(self, documents: List[str]):
        # Code to add documents to ChromaDB
        ids = [uuid4().hex for _ in documents]
        self.chroma_client.add_texts(texts=documents, ids=ids)
    @traceable(name="vector_search")
    def similarity_search(self, query: str, topK: int = 3) -> List[str]:
        # Code to query ChromaDB and return topK results
        results = self.chroma_client.similarity_search(query=query, k=topK)
        return [doc.page_content for doc in results]
# -------------------------
# RAG Implementation
# -------------------------
class RAG():
    def __init__(self, vectorDB: VectorDB, knowledge_csv: str, refresh: bool = True):
        self.llm = ChatOpenAI(base_url='http://localhost:8080/v1')
        self.vector_db = vectorDB
        self.documents  = [knowledge_csv]
        self.refresh = refresh
        if self.refresh:
            self._insert()
    
    def _insert(self):
        for doc in self.documents:
            df = extract_text(doc)
            df['text'] = df['CWE-ID'].astype(str) + ": " + df['DESCRIPTION']
            self.vector_db.add_documents(df['text'].tolist())
    
    @traceable(name="rag_based_answer")
    def answer(self, query: str, topK: int = 3) -> str:
        relevant_chunks = self.vector_db.similarity_search(query, topK)
        logger.info(f"\033[1;34mRetrieved: {len(relevant_chunks)}.\nContext: {relevant_chunks}\033[0m")
        context = "\n\n".join(relevant_chunks)
        prompt = [
            SystemMessage(content="You are a helpful assistant that answers questions based on the provided context."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ]
        response = self.llm.invoke(prompt)
        return response.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Language analytics and sampling utility')
    parser.add_argument('--refresh', action='store_true', help='Refresh and reinsert CWE descriptions into vector DB')
    args = parser.parse_args()
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    vectorDB = ChromaDB(embedding_model = embedding_model, collection_name = "chroma_collection")
    rag = RAG(vectorDB = vectorDB, knowledge_csv = "data/CVEFixes/CWE_descriptions.csv", 
            refresh = args.refresh)
    # TC 1
    q1 = "What is CWE-89? SQL Injection"
    logger.info(f'\033[1;37mQuestion: {q1}\033[0m \n\033[1;32mAnswer: {rag.answer(q1)}\033[0m')
    