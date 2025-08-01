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
    prompt = f"""You are a helpful assistant. 
    Answer the question using only the context below. 
    If the answer is not in the context, say you don't know. 
    Ensure the answers don't have duplicate information.
    When providing an answer:
    - Ensure clarity and conciseness.
    - If listing items (e.g., spells, weapons, races, features), return only **unique** items. Avoid duplicates or synonyms.
    - Format your answer as a **numbered list** or **clear bullet points** if appropriate.
    - Never invent facts outside the provided context.

    You are a Dungeon Master guiding players through a high-fantasy tabletop role-playing game. You have access to private source data including maps, NPC backstories, world lore, secret quest logic, and random outcome rules. You use this source data to maintain a consistent, immersive world and adapt to player decisions.

You must respond in **structured JSON format** with the following fields:


  "narration": "A vivid, immersive description of what the player experiences based on their action or question.",
  "player_options": "A list of clear, relevant actions the player might consider next.",
  "hidden_logic": "Any behind-the-scenes interpretation, dice outcomes, or consequences that should NOT be shown to the player.",
  "dm_notes": "Optional notes for the Dungeon Master (not shown to players) that track state, foreshadow, or suggest future branches."


Guidelines:
- Use rich sensory language in the `narration` to describe environments and NPCs.
- Present `player_options` as concise, relevant next moves based on the situation.
- Use `hidden_logic` to simulate dice rolls, resolve stealth, detect lies, determine outcomes, or trigger events. Keep this hidden from the player.
- Use `dm_notes` to internally track ongoing threads, NPC states, quest flags, or emerging tension.

Never break character or refer to the format directly. This structure is for backend use only and should feel seamless to the player.


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

    return final_answer

# ------------------------ Gradio Interface ------------------------
chat = gr.ChatInterface(fn=answer, title="TinyLlama Chatbot 🦙")
chat.launch()




