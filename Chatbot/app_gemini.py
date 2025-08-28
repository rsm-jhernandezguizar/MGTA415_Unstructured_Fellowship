# app_gemini.py
import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from Chatbot.gemini_council import run_council

# ---------- Page ----------
st.set_page_config(page_title="DnD Gemini Chatbot", page_icon="✨", layout="wide")

# Load local .env (no-op on Spaces, but helpful locally)
load_dotenv()

# ---------- Config ----------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

st.sidebar.header("Settings")
st.sidebar.write("Backend: Google Gemini API")
st.sidebar.text_input("Model", value=GEMINI_MODEL, key="gemini_model_preview", disabled=True)
# st.sidebar.caption("Set GEMINI_API_KEY (or GOOGLE_API_KEY) — in local .env or Space → Settings → Variables & secrets.")

# ---------- Client (cached) ----------
@st.cache_resource(show_spinner=False)
def get_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

client = get_client()

if client is None:
    st.warning("API key missing. Set GEMINI_API_KEY (or GOOGLE_API_KEY) and refresh.")
    
# ---------- Chat state ----------
if "messages" not in st.session_state:
    # keep your existing tuple format for backward-compat
    st.session_state.messages = []

st.title("Gemini Chatbot")

# Render chat history
for role, content in st.session_state.messages:
    with st.chat_message(role):
        if isinstance(content, (dict, list)):
            st.json(content)
        else:
            st.markdown(content)

# ---------- Generation ----------
# def generate_reply(user_msg: str) -> str:
#     if client is None:
#         return "⚠️ Gemini error: missing API key (set GEMINI_API_KEY or GOOGLE_API_KEY)."

#     try:
        # Build a simple conversational context from your stored (role, content) tuples
        # context = "\n".join(f"{r.title()}: {c}" for r, c in st.session_state.messages)
        # system_prompt = """"""
#         system_prompt = """
# You are a helpful assistant and Dungeon Master in a high-fantasy tabletop world. 
# Your task is to classify and respond to the player's query using only the context provided below. 
# If the answer is not in the context, say you don't know. Do not invent facts. Avoid duplicate information.

# **Step 1 — Classify the query intent.** Choose the most likely category for this question:
# - "spell" — related to magic, casting, effects, durations, components
# - "weapon" — related to attacks, damage types, martial equipment
# - "feat" — character abilities, enhancements, bonuses
# - "class" — features of D&D character classes
# - "lore" — world-building, places, quests, NPCs, hidden knowledge

# **Step 2 — Answer using only the retrieved context.**
# - Structure the output in JSON with the required fields.
# - Use vivid sensory detail in `narration`.
# - Keep `player_options` grounded in the situation.
# - Use `hidden_logic` for dice outcomes, passive checks, and event triggers.
# - Store meta or branching logic in `dm_notes`.

# **Always begin by clearly stating the question type.**

#                         """


        # prompt = f"{system_prompt}\n {context}\nUser: {user_msg}\nAssistant:\n{run_council(user_msg)}"
        
    #     prompt = ''
    #     resp = client.models.generate_content(
    #         model=GEMINI_MODEL,
    #         contents=prompt
    #     )
    #     # Prefer .text; fall back defensively
    #     return getattr(resp, "text", "") or str(resp)
    # except Exception as e:
    #     return f"⚠️ Gemini error: {e}"

# ---------- UI ----------
## assings and makes an if conditon for the chat input
if user_msg := st.chat_input("Type your message"):
    #append the user's message log with the message
    st.session_state.messages.append(("user", user_msg))

    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = run_council(user_msg)
            st.json(reply)

            # reply = generate_reply(user_msg)
            # st.markdown(reply)
            

    st.session_state.messages.append(("assistant", reply))
