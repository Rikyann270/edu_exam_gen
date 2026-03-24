import os
import json

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("Warning: chromadb not found. Install with: pip install chromadb openai")
    chromadb = None

from dotenv import load_dotenv

load_dotenv() # Load variables from .env if present

# Set up API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

BASE_DIR = "/Users/luke/Downloads/docs/Filemail.com - eduquest"
INPUT_JSON = os.path.join(BASE_DIR, "extracted_syllabus_data.json")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

def chunk_text(text, chunk_size=1500):
    """
    Splits the text into smaller chunks. 
    1500 characters is roughly 300 words, perfect for finding specific exam questions.
    """
    chunks = []
    # Basic character split (ensure overlapping or cleaner boundary splitting in production)
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def build_vector_db():
    print(f"Loading data from {INPUT_JSON}...")
    if not os.path.exists(INPUT_JSON):
        print("Data file not found. Have you completed Step 1 (extract_data.py)?")
        return

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if not chromadb:
        return

    print("Initializing ChromaDB and OpenAI Embeddings...")
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    
    # We use OpenAI's text-embedding-3-small (much cheaper and highly efficient)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

    # Recreate the collection to clear out old data if ran twice
    try:
        chroma_client.delete_collection(name="exam_syllabus_collection")
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name="exam_syllabus_collection", 
        embedding_function=openai_ef
    )

    documents = []
    metadatas = []
    ids = []

    print("Chunking documents...")
    chunk_id_counter = 0
    # Add a safety limit if testing, or process everything
    for doc in dataset:
        if "content" not in doc or not doc["content"]:
            continue
            
        chunks = chunk_text(doc["content"])
        
        for chunk in chunks:
            # Metadata allows us to filter later (e.g. "Only search P6 Science")
            safe_metadata = {
                "filename": str(doc.get("filename", "Unknown")),
                "level": str(doc.get("level", "Unknown")),
                "subject": str(doc.get("subject", "Unknown")),
                "term": str(doc.get("term", "Unknown")),
                "doc_type": str(doc.get("doc_type", "Unknown"))
            }

            documents.append(chunk)
            metadatas.append(safe_metadata)
            ids.append(f"chunk_{chunk_id_counter}")
            chunk_id_counter += 1

    print(f"Adding {len(documents)} document chunks to Vector Database (calling OpenAI API)...")
    
    # Upload in batches of 100 to avoid memory / API limits
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        print(f"Uploading batch {i} to {end} / {len(documents)}...")
        try:
            collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
        except Exception as e:
            print(f"Error uploading batch {i}-{end}: {e}")

    print(f"\n✅ Successfully built Vector Database at '{DB_DIR}'!")

if __name__ == "__main__":
    build_vector_db()
