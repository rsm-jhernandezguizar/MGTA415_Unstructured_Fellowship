import gradio as gr
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import faiss
import numpy as np
import torch
import json
from pathlib import Path
from load_documents_v2 import load_docs



# ------------------------ Load documents ------------------------
docs = load_docs()
print(f"📚 Loaded {len(docs)} documents.")


# model = "sentence-transformers/all-MiniLM-L6-v2"
# model = 'bert-base-nli-mean-tokens'
model ='all-mpnet-base-v2'
embedder = SentenceTransformer(model)


question_cache = []


# Load or compute embeddings
emb_path = Path("embeddings.npy")

if emb_path.exists():
    print("Loading cached embeddings...")
    embs = np.load(emb_path)
else:
    print("Computing embeddings...")
    embs = embedder.encode(docs, batch_size=64, convert_to_numpy=True)
    np.save(emb_path, embs)
    print("Saved embeddings to disk.")

# Build the FAISS index
print("Building FAISS index...")
index = faiss.IndexFlatL2(embs.shape[1])  # shape[1] = embedding dimension
index.add(embs)

print(f"Index is trained: {index.is_trained}")
print(f"Number of vectors in index: {index.ntotal}")


DATA_DIR = Path("Data")
DATA_DIR.mkdir(exist_ok=True)

QA_TEXT_FILE = DATA_DIR / "question_texts.jsonl"
QA_EMB_FILE = DATA_DIR / "question_embs.npy"

def load_question_cache():
    if QA_TEXT_FILE.exists() and QA_EMB_FILE.exists():
        with open(QA_TEXT_FILE, "r") as f:
            questions = [json.loads(line) for line in f]
        embeddings = np.load(QA_EMB_FILE)
        return questions, torch.tensor(embeddings)
    return [], torch.empty(0)

def save_question_entry(question, answer, embedding_tensor):
    with open(QA_TEXT_FILE, "a") as f:
        f.write(json.dumps({"question": question, "answer": answer}) + "\n")

    embedding = embedding_tensor.cpu().numpy()
    if QA_EMB_FILE.exists():
        existing = np.load(QA_EMB_FILE)
        updated = np.vstack([existing, embedding])
    else:
        updated = np.expand_dims(embedding, axis=0)

    np.save(QA_EMB_FILE, updated)

def build_question_index():
    if not QA_EMB_FILE.exists() or not QA_TEXT_FILE.exists():
        return None, []

    embs = np.load(QA_EMB_FILE).astype("float32")
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)

    questions = []
    with open(QA_TEXT_FILE) as f:
        for line in f:
            questions.append(json.loads(line))

    return index, questions



def retrieve(question: str, k: int = 2):
    q_emb = embedder.encode([question]).astype("float32")
    D, I = index.search(q_emb, k)                # distances & indices
    hits = [docs[i] for i in I[0]]
    return "\n\n".join(hits)





from transformers import pipeline, AutoTokenizer

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # any small instruct model works
tok  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llm  = pipeline("text-generation",
                model=model_name,
                tokenizer=tok,
                device="cpu",           
                max_new_tokens=200,
                temperature=.01)        


# Load functions from previous steps (assumed available)
# load_question_cache(), save_question_entry()

# Load cache from disk
question_cache, question_embs = load_question_cache()

def l2_distance(a, b):
    return torch.norm(a - b)



def get_cached_answer(query, threshold=0.90):  # adjust threshold higher for stricter match
    q_emb = embedder.encode(query, convert_to_tensor=True)
    for i, item in enumerate(question_cache):
        if item["question"].strip().lower() == query.strip().lower():
            print(f"✅ Exact match for: {item['question']}")
            return item["answer"], "exact_match"

        sim = cos_sim(q_emb, question_embs[i]).item()
        if sim >= threshold:
            print(f"🤝 Similar match (cosine={sim:.3f}) for: {item['question']}")
            return item["answer"], "similar_match"

    return None, None



def answer(question: str, history=None):
    cached, reason = get_cached_answer(question)

    if cached:
        if reason == "exact_match":
            print("✅ Used exact cached question (no embedding needed).")
        elif reason == "similar_match":
            print("🤝 Used similar cached question (Cosine similarity).")
        return cached

    print("🧠 No cached match. Generating answer with LLM...")
    context = retrieve(question, k=1)
    prompt = f"""You are a helpful assistant and Dungeon Master in a high-fantasy tabletop world. 
Your task is to classify and respond to the player's query using only the context provided below. 
If the answer is not in the context, say you don't know. Do not invent facts. Avoid duplicate information.

**Step 1 — Classify the query intent.** Choose the most likely category for this question:
- "spell" — related to magic, casting, effects, durations, components
- "weapon" — related to attacks, damage types, martial equipment
- "feat" — character abilities, enhancements, bonuses
- "class" — features of D&D character classes
- "lore" — world-building, places, quests, NPCs, hidden knowledge

**Step 2 — Answer using only the retrieved context.**
- Structure the output in JSON with the required fields.
- Use vivid sensory detail in `narration`.
- Keep `player_options` grounded in the situation.
- Use `hidden_logic` for dice outcomes, passive checks, and event triggers.
- Store meta or branching logic in `dm_notes`.

**Always begin by clearly stating the question type.**

    ### Context
    {context}

    ### Question
    {question}

    ### Answer
    """
    resp = llm(prompt)[0]["generated_text"]
    final_answer = resp.split("### Answer", 1)[-1].strip()

    # Store in disk-based cache
    embedding = embedder.encode(question, convert_to_tensor=True)
    save_question_entry(question, final_answer, embedding)
    print(prompt)
    return final_answer

# ------------------------ Gradio Interface ------------------------
chat = gr.ChatInterface(fn=answer, title="🦙 🛡️🧙 ⚔️ TinyLlama Chatbot ⚔️🧙🛡️  🦙")
chat.launch(server_port=7860)




