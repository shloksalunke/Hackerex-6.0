import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from langchain_unstructured import UnstructuredLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from mistralai.client import MistralClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from cachetools import LRUCache
from tenacity import retry, stop_after_attempt, wait_exponential

class QueryProcessor:
    """
    Final, robust RAG pipeline designed to handle both known (pre-cached)
    and unknown (on-the-fly) documents with high performance and accuracy.
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Mistral API key is required.")
        self.client = MistralClient(api_key=api_key)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Cache for vector stores, keyed by document URL
        self.vectorstore_cache = LRUCache(maxsize=10)

    def _create_vectorstore_from_url(self, doc_url: str) -> FAISS:
        """Downloads a document from a URL and creates a FAISS vector store."""
        response = requests.get(doc_url, timeout=60)
        response.raise_for_status()
        
        # Use a temporary file to handle the document content
        with open("temp_document.pdf", "wb") as f:
            f.write(response.content)
        
        try:
            loader = UnstructuredLoader(file_path="temp_document.pdf")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
            docs = text_splitter.split_documents(documents)
            
            if not docs:
                raise ValueError("Document could not be split into chunks.")

            vectorstore = FAISS.from_documents(docs, self.embeddings)
            print(f"✅ Vector store created for '{doc_url}'. Chunks: {len(docs)}")
            return vectorstore
        finally:
            # Clean up the temporary file
            if os.path.exists("temp_document.pdf"):
                os.remove("temp_document.pdf")

    def get_or_create_vectorstore(self, doc_url: str) -> FAISS:
        """
        The core logic: gets a vector store from cache if available,
        otherwise creates it on the fly. This handles UNKNOWN documents.
        """
        if doc_url in self.vectorstore_cache:
            print(f"✅ Cache hit for document: {doc_url}")
            return self.vectorstore_cache[doc_url]
        
        print(f"⏳ Cache miss for UNKNOWN document: {doc_url}. Processing on the fly...")
        vectorstore = self._create_vectorstore_from_url(doc_url)
        self.vectorstore_cache[doc_url] = vectorstore
        return vectorstore

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _get_single_answer(self, question: str, vectorstore: FAISS) -> str:
        """Processes a single question against a given vector store."""
        matched_docs = vectorstore.similarity_search(question, k=4)
        context = "\n\n---\n\n".join([doc.page_content for doc in matched_docs])

        prompt = f"""
Based ONLY on the following context, answer the user's question in a detailed and helpful manner, as if you are a policy expert explaining it to a customer.
Your response MUST be only the answer text. Do not add any preamble like "The answer is...".
If the information is not in the context, you MUST respond with the exact phrase: "The answer to this question could not be found in the provided document."

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat(model="mistral-large-latest", messages=messages)
        return response.choices[0].message.content.strip()

    def get_answers(self, doc_url: str, questions: list[str]) -> list[str]:
        """Main public method that drives the process for any document."""
        try:
            # This one line makes the system robust for any document
            vectorstore = self.get_or_create_vectorstore(doc_url)
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Pass the dynamically acquired vectorstore to each thread
                answers = list(executor.map(lambda q: self._get_single_answer(q, vectorstore), questions))
            return answers
        except Exception as e:
            print(f"🚨 A critical error occurred in get_answers: {e}")
            return [f"A system error occurred while processing {doc_url}: {str(e)}"] * len(questions)