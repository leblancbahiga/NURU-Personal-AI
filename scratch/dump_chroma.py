import chromadb
from pathlib import Path
import sys

# Path to ChromaDB
db_path = "/Users/leblancbahiga/Downloads/Assistant IA/data/chroma_db"

client = chromadb.PersistentClient(path=db_path)

def dump_collection(name):
    print(f"\n--- Collection: {name} ---")
    try:
        col = client.get_collection(name)
        res = col.get()
        for i in range(len(res['ids'])):
            print(f"ID: {res['ids'][i]}")
            print(f"Doc: {res['documents'][i]}")
            if res['metadatas']:
                print(f"Meta: {res['metadatas'][i]}")
            print("-" * 20)
    except Exception as e:
        print(f"Error reading {name}: {e}")

dump_collection("documents")
dump_collection("conversations")
dump_collection("corrections_prioritaires")
