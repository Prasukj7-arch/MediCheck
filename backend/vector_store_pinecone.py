import uuid
import os
import pickle
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class PineconeVectorStore:
    def __init__(self, index_name="rag-chatbot"):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name

        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=1024,  # mxbai-embed-large output dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(index_name)

    def store_embeddings(self, texts, embeddings, batch_size=500, save_to_file=True):
        """Store text chunks and their embeddings in Pinecone in batches."""
        vectors = [(str(uuid.uuid4()), embedding, {"text": text}) for text, embedding in zip(texts, embeddings)]

        # Optionally save to file for retry/debug
        if save_to_file:
            with open("embeddings.pkl", "wb") as f:
                pickle.dump(vectors, f)
            print("✅ Saved embeddings to embeddings.pkl")

        total = len(vectors)
        print(f"🔢 Total vectors to upload: {total}")

        for i in range(0, total, batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
                print(f"✅ Uploaded batch {i + 1} to {i + len(batch)}")
            except Exception as e:
                print(f"❌ Failed to upload batch {i + 1} to {i + len(batch)}: {e}")

    def query_embeddings(self, query_embedding, top_k=5):
        """Query Pinecone for similar texts."""
        results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        return [match["metadata"]["text"] for match in results["matches"]]
