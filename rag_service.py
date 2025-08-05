# --- START OF FILE rag_service.py (with your new advanced prompt) ---
import os
import requests
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from mistralai.client import MistralClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from cachetools import LRUCache
from tenacity import retry, stop_after_attempt, wait_exponential
import traceback

class QueryProcessor:
    """
    Elite-performance RAG pipeline, re-engineered for sub-29-second performance
    and maximum accuracy on complex, unknown documents.
    """
    def __init__(self, api_key: str):
        self.client = MistralClient(api_key=api_key)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.in_memory_cache = LRUCache(maxsize=10)
        self.disk_cache_path = "document_indexes"
        os.makedirs(self.disk_cache_path, exist_ok=True)

    def _get_url_hash(self, doc_url: str) -> str:
        return hashlib.md5(doc_url.encode()).hexdigest()

    def _create_and_save_vectorstore(self, doc_url: str) -> FAISS:
        print(f"⏳ Processing new document from URL: {doc_url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(doc_url, timeout=60, headers=headers)
        response.raise_for_status()
        
        temp_path = f"temp_{self._get_url_hash(doc_url)}.pdf"
        with open(temp_path, "wb") as f:
            f.write(response.content)

        try:
            loader = PyMuPDFLoader(file_path=temp_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            
            if not docs: raise ValueError("Document could not be split into text chunks.")

            vectorstore = FAISS.from_documents(docs, self.embeddings)
            
            index_path = os.path.join(self.disk_cache_path, self._get_url_hash(doc_url))
            vectorstore.save_local(index_path)
            print(f"✅ Vector store created and saved to disk at: {index_path}")
            return vectorstore
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)

    def get_vectorstore(self, doc_url: str) -> FAISS:
        if doc_url in self.in_memory_cache:
            print(f"✅ Cache hit (In-Memory) for document.")
            return self.in_memory_cache[doc_url]

        index_path = os.path.join(self.disk_cache_path, self._get_url_hash(doc_url))
        if os.path.exists(index_path):
            print(f"✅ Cache hit (Disk) for document. Loading now...")
            vectorstore = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            self.in_memory_cache[doc_url] = vectorstore
            return vectorstore
        
        vectorstore = self._create_and_save_vectorstore(doc_url)
        self.in_memory_cache[doc_url] = vectorstore
        return vectorstore

    def _format_answer(self, json_string: str) -> str:
        try:
            start_index = json_string.find('{')
            end_index = json_string.rfind('}') + 1
            if start_index == -1 or end_index == 0:
                print(f"🚨 Could not find JSON object in LLM response: {json_string}")
                return "Error: Could not parse AI model response."

            clean_json_string = json_string[start_index:end_index]
            data = json.loads(clean_json_string)
            
            decision = data.get("decision", "Not specified")
            justification = data.get("justification", "No justification provided.")
            
            if decision == "Not specified":
                return "The answer to this question could not be found in the provided document."
            
            return justification

        except (json.JSONDecodeError, KeyError) as e:
            print(f"🚨 Error parsing LLM JSON response: {e}\nRaw response was: {json_string}")
            return "A system error occurred while formatting the answer."

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _get_single_answer(self, question: str, vectorstore: FAISS) -> str:
        """Processes a single question using the new, advanced prompt."""
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
        matched_docs = retriever.invoke(question)
        context = "\n\n---\n\n".join([doc.page_content for doc in matched_docs])

        # --- ADVANCED PROMPT INTEGRATION (with f-string fix) ---
        prompt = f"""
        <RULES>
        You are an elite AI insurance analyst working for Bajaj Finserv. Your sole responsibility is to evaluate insurance-related user queries with utmost precision, based **exclusively** on the information provided in the <CONTEXT> block.

        Follow these strict instructions:

        1. 📌 **Strict Context Boundaries**  
           Your response must be based solely on the text within the <CONTEXT> section. You must NOT use external knowledge, assumptions, or interpretations beyond what is explicitly mentioned in the document.

        2. 🧠 **Analyze and Decide**  
           Examine the <CONTEXT> thoroughly to determine whether the <USER_QUESTION> can be answered clearly based on the clauses present. Your final decision must be one of the following:
           - `"Approved"`
           - `"Rejected"`
           - `"Not specified"`

        3. ✍️ **Justify Your Decision**  
           Provide a short, clear, and professional justification for your decision. It must reference the **specific conditions, inclusions, or exclusions** that led to the outcome.

        4. 📚 **Reference Clause(s)**  
           Point to the exact **clause number(s)** and/or **unique excerpt(s)** from the context that support your answer. If multiple clauses apply, list all of them in a **comma-separated format**.

        5. 🧾 **Format Output as Valid JSON**  
           You MUST respond with a strict and syntactically correct JSON block that conforms to the following structure. Do not add any text before or after the JSON block.

        ```json
        {{
          "decision": "Approved" | "Rejected" | "Not specified",
          "amount": "<numeric amount or 'N/A'>",
          "justification": "<A concise, professional explanation referencing matching conditions or exclusions.>",
          "clause_reference": "<Exact clause number(s) and/or short unique excerpt(s) from the document.>"
        }}
        ```

        6. ❓ **"Not Specified" Case**
           If the document does not provide sufficient information to determine an answer, you MUST return this exact JSON object:
        ```json
        {{
          "decision": "Not specified",
          "amount": "N/A",
          "justification": "The answer to this question could not be found in the provided document.",
          "clause_reference": ""
        }}
        ```
        </RULES>

        <CONTEXT>
        {context}
        </CONTEXT>

        <USER_QUESTION>
        {question}
        </USER_QUESTION>
        
        JSON_OUTPUT:
        """
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.chat(
            model="mistral-large-latest", 
            messages=messages,
            response_format={"type": "json_object"}
        )
        raw_json_answer = response.choices[0].message.content.strip()
        
        return self._format_answer(raw_json_answer)

    def get_answers(self, doc_url: str, questions: list[str]) -> list[str]:
        """Main public method that drives the process."""
        try:
            vectorstore = self.get_vectorstore(doc_url)
            with ThreadPoolExecutor(max_workers=len(questions) or 1) as executor:
                answers = list(executor.map(lambda q: self._get_single_answer(q, vectorstore), questions))
            return answers
        except Exception as e:
            print(f"🚨 A critical error occurred in get_answers. Exception type: {type(e)}, Message: {e}")
            traceback.print_exc()
            
            return [f"A system error occurred while processing {doc_url}: {str(e)}"] * len(questions)

# --- END OF FILE rag_service.py ---