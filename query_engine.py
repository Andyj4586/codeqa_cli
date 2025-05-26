import faiss
import pickle
import subprocess
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
index_path = "data/code_index.faiss"
chunks_path = "data/chunks.pkl"

def answer_question(question, top_k=3):
    try:
        index = faiss.read_index(index_path)
        chunks = pickle.load(open(chunks_path, "rb"))
    except:
        return "❌ Error: No index found. Please index a codebase first."

    q_vec = model.encode([question])
    D, I = index.search(q_vec, top_k)
    context = "\n\n".join([chunks[i] for i in I[0]])
    prompt = f"{context}\n\n### Question: {question}\n### Answer:"

    result = subprocess.run(
        ["ollama", "run", "codellama", prompt],
        capture_output=True, text=True
    )
    return result.stdout.strip()
