import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    import time
    from openai import RateLimitError

    BATCH_SIZE = 100  # You can adjust this
    RETRY_DELAY = 20  # seconds

    def _persist_db(d):
        """Call any available persist method on the Chroma object or its internals.

        Returns True if a persist method was found and called, False otherwise.
        """
        # Preferred: top-level persist
        persist_fn = getattr(d, "persist", None)
        if callable(persist_fn):
            persist_fn()
            return True

        # Try client-level persist (different package versions expose internals differently)
        client = getattr(d, "_client", None) or getattr(d, "client", None)
        if client:
            client_persist = getattr(client, "persist", None)
            if callable(client_persist):
                client_persist()
                return True

        # Try collection-level persist
        collection = getattr(d, "_collection", None) or getattr(d, "collection", None)
        if collection:
            coll_persist = getattr(collection, "persist", None)
            if callable(coll_persist):
                coll_persist()
                return True

        return False

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        for i in range(0, len(new_chunks), BATCH_SIZE):
            batch = new_chunks[i:i+BATCH_SIZE]
            batch_ids = [chunk.metadata["id"] for chunk in batch]
            while True:
                try:
                    db.add_documents(batch, ids=batch_ids)
                    # Persist if available (safe across different chroma/langchain versions)
                    persisted = _persist_db(db)
                    if persisted:
                        print(f"✅ Persisted batch {i//BATCH_SIZE+1}")
                    else:
                        print(f"✅ Added batch {i//BATCH_SIZE+1} ({len(batch)} docs) (no persist method detected)")
                    break
                except RateLimitError as e:
                    err_text = str(e)
                    # If it's an exhausted quota error, retrying won't help. Try to fallback to local embeddings.
                    if "insufficient_quota" in err_text or "You exceeded your current quota" in err_text:
                        print(f"❌ Insufficient quota detected: {e}")
                        print("Attempting to switch to local embeddings (sentence-transformers). If not installed, the script will exit with instructions.")
                        try:
                            import sys
                            os.environ["USE_LOCAL_EMBEDDINGS"] = "1"
                            new_embedding = get_embedding_function()
                        except Exception as fallback_err:
                            print("⚠️ Could not create local embeddings:", fallback_err)
                            print("Next steps:")
                            print(" 1) Add billing or increase quota for your OpenAI account, or")
                            print(" 2) Install sentence-transformers and set USE_LOCAL_EMBEDDINGS=1 to use local embeddings:")
                            print("    pip install sentence-transformers");
                            print("    setx USE_LOCAL_EMBEDDINGS 1  # (Windows) then re-open the terminal"
                                  )
                            sys.exit(1)

                        # Recreate the Chroma DB with the local embedding function and refresh existing IDs.
                        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=new_embedding)
                        existing_items = db.get(include=[])
                        existing_ids = set(existing_items["ids"]) if existing_items and "ids" in existing_items else set()
                        print("✅ Switched to local embeddings and refreshed DB. Resuming ingestion.")
                        # After switching embeddings, try the same batch again.
                        continue
                    else:
                        print(f"⚠️ Rate limit hit, waiting {RETRY_DELAY}s: {e}")
                        time.sleep(RETRY_DELAY)
        print("✅ All new documents added.")
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/The Sorcerer's Stone - J. K. Rowling.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()