# data_indexing.py — clean rebuild-safe version
   
import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from config import CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL, POLICY_DOCS_PATH
from pdf_extractor import PDFExtractor

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
except Exception:
    pass

# ======== Setup ======== #
embedder = SentenceTransformer(EMBED_MODEL)
extractor = PDFExtractor()

def build_index(force_rebuild: bool = False):
    """
    Build or reuse an index in ChromaDB.
    Ensures embedding dimensions always match EMBED_MODEL.
    """
    # Always nuke collection if forced OR if embedding dims mismatch
    if force_rebuild and os.path.exists(CHROMA_DIR):
        print(f"🗑️ Removing old Chroma DB at {CHROMA_DIR} ...")
        shutil.rmtree(CHROMA_DIR)

    chroma = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = chroma.get_collection(CHROMA_COLLECTION)
        count = collection.count()
        meta = collection.metadata or {}
        stored_dim = meta.get("embedding_dimension")

        if count > 0 and stored_dim == embedder.get_sentence_embedding_dimension():
            print(f"✅ Reusing existing collection '{CHROMA_COLLECTION}' with {count} chunks "
                  f"(dim={stored_dim}).")
            return
        else:
            print("⚠️ Embedding mismatch or empty collection → rebuilding index.")
            chroma.delete_collection(CHROMA_COLLECTION)
    except Exception:
        # Collection doesn't exist yet
        pass

    print(f"Building new collection '{CHROMA_COLLECTION}' from PDFs in {POLICY_DOCS_PATH}...")
    collection = chroma.get_or_create_collection(
        CHROMA_COLLECTION,
        metadata={"embedding_dimension": embedder.get_sentence_embedding_dimension()}
    )

    docs = extractor.load_all_pdfs_from_folder(POLICY_DOCS_PATH)
    if not docs:
        print("⚠️ No documents found in PDF folder.")
        return

    print(f"Encoding {len(docs)} chunks with {EMBED_MODEL} "
          f"(dim={embedder.get_sentence_embedding_dimension()}) ...")
    texts = [d.text for d in docs]
    embeddings = embedder.encode(texts, convert_to_numpy=True).tolist()

    print("Inserting into ChromaDB...")
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(texts))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"id": f"chunk_{i}"} for i in range(len(texts))]
    )

    print(f"✅ Indexed {len(texts)} chunks into '{CHROMA_COLLECTION}' at {CHROMA_DIR}")

if __name__ == "__main__":
    # Force rebuild recommended when switching embed models
    build_index(force_rebuild=True)
