import gradio as gr
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from load_documents import load_docs
docs = load_docs()

query_chache = {}


embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embs = embedder.encode(docs, batch_size=64, convert_to_numpy=True)
index = faiss.IndexFlatL2(embs.shape[1])
index.add(embs)
question_cache = []


def retrieve(question: str, k: int = 5):
    q_emb = embedder.encode([question]).astype("float32")
    D, I = index.search(q_emb, k)                # distances & indices
    hits = [docs[i] for i in I[0]]
    return "\n\n".join(hits)




model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # any small instruct model works
tok  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llm  = pipeline("text-generation",
                model=model_name,
                tokenizer=tok,
                device="cpu",            # change to 0 if you have a GPU
                max_new_tokens=200,
                temperature=.01)         # deterministic

from sentence_transformers.util import cos_sim
import torch


def get_cached_answer(query, threshold=0.90):
    q_emb = embedder.encode(query, convert_to_tensor=True)
    for item in question_cache:
        sim = cos_sim(q_emb, item['embedding']).item()
        if sim >= threshold:
            print(f"⚡ Using cached answer (similarity={sim:.2f}) for: {item['question']}")
            return item['answer']
    return None

def answer(question: str,history=None):
    cached = get_cached_answer(question)
    if cached:
        return cached

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

    # Store in cache
    question_cache.append({
        'question': question,
        'embedding': embedder.encode(question, convert_to_tensor=True),
        'answer': final_answer
    })

    return final_answer




# Launch Gradio chat UI
chat = gr.ChatInterface(fn=answer, title="TinyLlama Chatbot 🦙")
chat.launch()
