from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_qna import RAGQnA
from utils import scrape_website, chunk_text, get_nhs_condition_urls
import os

BATCH_SIZE = 40
PROGRESS_FILE = "scrape_index.txt"

app = FastAPI(title="NHS Disease RAG")

class Query(BaseModel):
    text: str

# Initialize RAG Q&A
rag_qna = RAGQnA()

# Step 1: Fetch all condition URLs
condition_urls = get_nhs_condition_urls()
total_urls = len(condition_urls)

# Step 2: Read last scraped index
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        start_index = int(f.read().strip())
else:
    start_index = 0

end_index = min(start_index + BATCH_SIZE, total_urls)
batch_urls = condition_urls[start_index:end_index]

print(f"📦 Scraping batch {start_index + 1} to {end_index} of {total_urls} URLs")

all_chunks = []

for i, url in enumerate(batch_urls, start=start_index + 1):
    print(f"🔍 Scraping URL {i}/{total_urls}: {url}")
    try:
        scraped = scrape_website(url)
        if scraped:
            chunks = chunk_text(scraped)
            all_chunks.extend(chunks)
            print(f"✅ Ingested {len(chunks)} chunks from: {url}")
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")

# Step 3: Ingest into RAG
if all_chunks:
    rag_qna.ingest_data(all_chunks)
    print(f"✅ Finished batch {start_index + 1} to {end_index}")
    # Save new index for next run
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(end_index))
else:
    print("⚠️ No data was ingested in this batch.")

@app.post("/qna")
async def qna(query: Query):
    try:
        answer = rag_qna.answer_question(query.text)
        return {"question": query.text, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))