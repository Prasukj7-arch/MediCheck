import requests
from langchain_core.embeddings import Embeddings
from langchain.docstore.document import Document
from vector_store_pinecone import PineconeVectorStore
import uuid

class OllamaEmbeddings(Embeddings):
    def __init__(self, base_url="http://localhost:11434/v1/embeddings", model="mxbai-embed-large:latest"):
        self.base_url = base_url
        self.model = model
        self.headers = {"Content-Type": "application/json"}
    
    def embed_documents(self, texts):
        payload = {"model": self.model, "input": texts}
        try:
            response = requests.post(self.base_url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except requests.RequestException as e:
            raise ValueError(f"Error generating embeddings: {e}")
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]

class RAGQnA:
    def __init__(self):
        self.vector_store = PineconeVectorStore()
        self.embedding_model = OllamaEmbeddings()
        self.llm_url = "http://localhost:11434/v1/chat/completions"
        self.llm_model = "qwen2.5-coder:0.5b"
    
    def ingest_data(self, texts):
        """Ingest text chunks into Pinecone."""
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            print(f"Generated {len(embeddings)} embeddings. First embedding length: {len(embeddings[0])}")
            self.vector_store.store_embeddings(texts, embeddings)
            print(f"Successfully stored {len(texts)} chunks in Pinecone.")
        except Exception as e:
            print(f"Error storing in Pinecone: {e}")

    def answer_question(self, question):
        """Answer a question using RAG."""
        if not question.strip():
            return "Error: Query cannot be empty."
        
        # Embed query
        try:
            query_embedding = self.embedding_model.embed_query(question)
            print(f"Query embedding generated (length: {len(query_embedding)})")
        except Exception as e:
            print(f"Error embedding query: {e}")
            return "Failed to embed query."
        
        # Retrieve relevant chunks
        try:
            contexts = self.vector_store.query_embeddings(query_embedding, top_k=3)
            context = "\n\n".join(contexts)
            print(f"Retrieved {len(contexts)} chunks for context.")
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return "Failed to retrieve relevant chunks."
        
        # Prepare LLM input
        llm_input = f"User query: {question}\n\nContext from documents:\n{context}\n\nAnswer the user's query based on the provided context."
        
        # Call LLM
        try:
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": llm_input}
                ],
                "stream": False
            }
            response = requests.post(self.llm_url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            llm_output = response.json()["choices"][0]["message"]["content"]
            return llm_output
        except requests.RequestException as e:
            print(f"Error calling LLM: {e}")
            return "Failed to generate response from LLM."