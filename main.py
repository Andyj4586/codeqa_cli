import os
from indexer import index_codebase
from query_engine import answer_question

def main():
    os.makedirs("data", exist_ok=True)

    print("🔍 Welcome to CodeQA CLI\n")
    if not os.path.exists("data/code_index.faiss"):
        repo_path = input("📁 Enter path to your codebase: ").strip()
        print("📦 Indexing codebase...")
        index_codebase(repo_path)
        print("✅ Indexing complete!\n")

    while True:
        question = input("❓ Ask a question (or type 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        answer = answer_question(question)
        print(f"\n🧠 Answer:\n{answer}\n")

if __name__ == "__main__":
    main()
