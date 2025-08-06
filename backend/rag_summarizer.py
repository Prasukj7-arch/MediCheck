import requests
from langchain_core.embeddings import Embeddings

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

class RAGSummarizer:
    def __init__(self):
        self.embedding_model = OllamaEmbeddings()
        self.llm_url = "http://localhost:11434/v1/chat/completions"
        self.llm_model = "qwen2.5-coder:0.5b"
    
    def ingest_data(self, texts):
        """Ingest text chunks into Weaviate."""
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            print(f"Generated {len(embeddings)} embeddings for summarizer.")
            self.vector_store.store_embeddings(texts, embeddings)
            print(f"Successfully stored {len(texts)} chunks in Weaviate.")
        except Exception as e:
            print(f"Error storing in Weaviate: {e}")

    def summarize(self, query):
        """Generate a summary for the query."""
        if not query.strip():
            return "Error: Query cannot be empty."
        
        # Embed query
        try:
            query_embedding = self.embedding_model.embed_query(query)
            print(f"Query embedding generated for summarization (length: {len(query_embedding)})")
        except Exception as e:
            print(f"Error embedding query: {e}")
            return "Failed to embed query."
        
        # Retrieve relevant texts
        try:
            texts = self.vector_store.query_embeddings(query_embedding, limit=3)
            combined_text = " ".join(texts)
            print(f"Retrieved {len(texts)} chunks for summarization.")
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return "Failed to retrieve relevant chunks."
        
        # Prepare LLM input
        llm_input = f"Summarize the following text:\n{combined_text}\nSummary:"
        
        # Call LLM
        try:
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant specialized in summarization."},
                    {"role": "user", "content": llm_input}
                ],
                "stream": False
            }
            response = requests.post(self.llm_url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            summary = response.json()["choices"][0]["message"]["content"]
            return summary
        except requests.RequestException as e:
            print(f"Error calling LLM: {e}")
            return "Failed to generate summary."