import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
dim = 384

EXCLUDED_DIRS = {"node_modules", ".git", "venv", "__pycache__"}
#MAX_FILE_SIZE = 1_000_000  # 1MB limit

def chunk(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def index_codebase(path: str, index_path="data/code_index.faiss", chunks_path="data/chunks.pkl"):
    chunks = []
    index = faiss.IndexFlatL2(dim)

    print(f"📁 Scanning codebase at: {path}")

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file in files:
            filepath = os.path.join(root, file)
            if not file.endswith(('.py', '.js', '.ts', '.md', '.txt')):
                continue
            #if os.path.getsize(filepath) > MAX_FILE_SIZE:
            #    print(f"⚠️ Skipping large file: {filepath}")
            #    continue
            try:
                with open(filepath, 'r', errors='ignore') as f:
                    text = f.read()
                    parts = chunk(text)
                    print(f"📄 Indexing {file} ({len(parts)} chunks)")
                    vecs = model.encode(parts, show_progress_bar=False)
                    index.add(vecs)
                    chunks.extend(parts)
            except Exception as e:
                print(f"❌ Failed to process {filepath}: {e}")

    if not chunks:
        raise ValueError("🚫 No valid code files were indexed.")

    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\n✅ Indexed {len(chunks)} chunks from {path}")
