"""
Chat with Multiple PDFs + D&D Council (Streamlit + LangChain + Hugging Face + Gemini)
-------------------------------------------------------------------------------
This app now supports two complementary modes:

A) PDF RAG (retrieval-augmented generation)
   - Upload one or more PDFs, we index them with FAISS, and a small HF model
     (FLAN-T5) answers questions using retrieved context.

B) D&D 5e Council (Gemini-based multi-persona adjudication)
   - Ask a D&D rules/lore question and receive five persona proposals (RAW, RAI,
     Balance DM, Narrative DM, System Lawyer) + a judge synthesis.

The code is heavily commented to explain each piece and safe defaults. Use the
sidebar to select a mode and configure parameters.

Security & keys:
- The Gemini council requires a Google API key (GEMINI_API_KEY or GOOGLE_API_KEY).
  You can also paste it in the sidebar at runtime (kept only in memory).
- The PDF RAG mode does not require external API keys.

"""

# ================================
# Imports
# ================================

import json
from io import BytesIO
from typing import List, Dict

import streamlit as st
import torch
import fitz  # PyMuPDF for robust PDF parsing

# LangChain bits
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document

# Hugging Face Transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from pathlib import Path
from dotenv import load_dotenv
import os

# Load env from both CWD and the script’s folder
load_dotenv()  # tries current working directory
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")  # Chatbot/.env


# Gemini (Google GenAI); import is optional so the PDF RAG mode still works
try:
    import os
    import google.genai as genai
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False


# ================================
# Hardware / device selection
# ================================
# If CUDA is available we use it for the generator (faster inference);
# otherwise we fall back to CPU. Embeddings are kept on CPU for memory safety.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================
# Cached heavyweight resources
# ================================
# Streamlit reruns the script on every widget interaction. To prevent reloading
# models repeatedly (slow!), we cache the tokenizer/model/embeddings/pipeline.

@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load a compact sentence-transformer for embeddings on CPU.

    all-MiniLM-L6-v2 is a strong speed/quality baseline (~384-d vectors).
    """
    embedding_model = "all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},  # keep embeddings on CPU
    )


@st.cache_resource(show_spinner=False)
def get_generator(model_name: str = "google/flan-t5-small", max_length: int = 256, temperature: float = 0.7):
    """Create a text2text-generation pipeline on the chosen device.

    FLAN-T5 small is lightweight and fine for demos; you can swap in a larger
    model (e.g., flan-t5-base) if you have the VRAM. For decoder-only models,
    use a "text-generation" pipeline instead.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    generator = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1,  # -1 = CPU
        model_kwargs={
            # max_length is an upper bound for total generated sequence length.
            # We'll still also pass max_new_tokens at call-time to be explicit.
            "max_length": max_length,
            # Temperature > 1.0 → more random; < 1.0 → more deterministic.
            "temperature": temperature,
        },
    )
    return generator


# ================================
# PDF loading and text prep
# ================================

def load_pdfs(uploaded_files) -> List[Document]:
    """Read PDFs with PyMuPDF and return LangChain Documents.

    Each Document holds the entire PDF's extracted text in .page_content and
    stores the original filename in .metadata for traceability.
    """
    documents: List[Document] = []
    for uploaded_file in uploaded_files:
        pdf_content = BytesIO(uploaded_file.read())
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        try:
            text = []
            for page in doc:
                # get_text() usually returns a reasonable layout-preserving text.
                # If your PDFs are scans, you'll need OCR (e.g., pytesseract).
                text.append(page.get_text())
            full_text = "".join(text)
        finally:
            doc.close()

        documents.append(
            Document(page_content=full_text, metadata={"file_name": uploaded_file.name})
        )
    return documents


def chunk_documents(documents: List[Document], *, chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """Split long docs into overlapping chunks for better recall.

    Overlap helps keep context continuity across chunk boundaries.
    Tune sizes based on your model's context window and typical query length.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


# ================================
# Retrieval-augmented generation (RAG) helpers
# ================================

def build_vector_store(chunks: List[Document]) -> FAISS:
    """Embed chunks and build an in-memory FAISS index."""
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def retrieve_context(index: FAISS, query: str, k: int = 3) -> List[Document]:
    """Return top-k similar chunks for a user query."""
    return index.similarity_search(query, k=k)


def build_prompt(retrieved_docs: List[Document], user_query: str) -> str:
    """Concatenate retrieved text + the user's question into a simple prompt.

    For stronger results, consider an explicit instruction template like:
    "Answer using only the CONTEXT below. If unsure, say you don't know.

"
    """
    context = "\n".join(doc.page_content.strip() for doc in retrieved_docs)
    prompt = (
        f"You are a helpful assistant. Use the CONTEXT to answer the QUESTION. "
        f"CONTEXT: {context} QUESTION: {user_query} ANSWER:"
    )
    return prompt


# ================================
# D&D Council (Gemini) helpers
# ================================
# We mirror the structure from your `Dnd_Council_AW.py` with extra comments
# and Streamlit-friendly surfaces. The purpose is to:
#   1) Generate 5 persona proposals
#   2) Synthesize a final judge ruling
#   3) Provide a short brief summary
# If google.genai isn't installed, we show a helpful error message.

_PERSONAS: List[str] = [
    "RAW Literalist (rules-as-written, cite page numbers when sure)",
    "RAI Interpreter (rules-as-intended, weigh designer intent and Sage Advice)",
    "Table Balance DM (fairness and pacing; minimize swingy outcomes)",
    "Narrative DM (cinematic flow; reward setup like hiding/ambush)",
    "System Lawyer (edge cases, timing windows, conditions, and advantage rules)",
]

_SYSTEM_PREAMBLE = """
    "You are an expert D&D 5e adjudication panel.",
    "Answer precisely and avoid homebrew unless asked. Distinguish RAW vs RAI when relevant.",
    "If you are uncertain on an exact page reference, say so explicitly rather than guessing."
)
                    """


def _persona_prompt(persona: str, question: str) -> str:
    """Return a compact JSON-structured prompt for a single persona."""
    return f"""{_SYSTEM_PREAMBLE}

Persona: {persona}

Task: Propose a short ruling (<= 180 words) to the player's question below.
- Be decisive.
- Note RAW vs RAI if applicable.
- If relevant, mention core sources (PHB, DMG, Basic Rules) and page refs **only if sure**.
- Include a 1–10 confidence score.

Question: {question}

Return JSON with keys:
  "persona": str,
  "ruling": str,
  "confidence": int
"""


def _final_prompt(question: str, proposals: List[Dict]) -> str:
    """Prompt the presiding judge to synthesize a final decision."""
    proposals_json = json.dumps(proposals, ensure_ascii=False)
    return f"""{_SYSTEM_PREAMBLE}

You are the presiding Judge. You have 5 proposals from different personas.

Question:
{question}

Proposals (JSON array):
{proposals_json}

Instructions:
- Synthesize a single final ruling.
- Briefly explain why you chose it vs alternatives (2–4 bullets).
- If relevant, clarify timing (surprise, hidden, advantage, sneak attack conditions).
- Add a 1–10 confidence score.
- If you are not 100% certain about a specific citation, do not invent it.

Return JSON with keys:
  "ruling": str,
  "reasoning": list[str],
  "confidence": int
"""


def _gemini_client(api_key: str | None = None):
    """Create a Gemini client from an explicit key or environment.

    Accepts GEMINI_API_KEY or GOOGLE_API_KEY from the environment if api_key
    is not provided. Raises a RuntimeError with a helpful message if missing.
    """
    if not _HAS_GENAI:
        raise RuntimeError("google.genai is not installed. Run: pip install google-genai")
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing Gemini API key: set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment, "
            "or paste it into the sidebar."
        )
    return genai.Client(api_key=key)


def run_council(question: str, *, api_key: str | None = None, model: str = "gemini-1.5-flash", temperature: float = 0.4) -> Dict[str, object]:
    """
    Runs a 5-member 'council' using Google Gemini and returns a dict:
      {
        "brief": {...},         # 1-2 sentence summary
        "proposals": [...],     # 5 persona proposals
        "final": {...}          # judge synthesis
      }

    Does NOT require OPENAI_API_KEY and does NOT use Serper.
    """
    client = _gemini_client(api_key=api_key)

    # 1) Generate proposals (5 personas)
    proposals: List[Dict] = []
    for persona in _PERSONAS:
        resp = client.models.generate_content(
            model=model,
            contents=_persona_prompt(persona, question),
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=512
            )
        )
        text = getattr(resp, "text", None) or (
            resp.candidates[0].content.parts[0].text if resp.candidates else ""
        )
        try:
            proposal = json.loads(text)
            if not isinstance(proposal, dict) or "persona" not in proposal:
                raise ValueError("Unexpected structure")
        except Exception:
            proposal = {
                "persona": persona,
                "ruling": text.strip(),
                "confidence": 6
            }
        proposals.append(proposal)

    # 2) Judge synthesis
    judge_resp = client.models.generate_content(
        model=model,
        contents=_final_prompt(question, proposals),
        config=genai.types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=512
        )
    )
    judge_text = getattr(judge_resp, "text", None) or (
        judge_resp.candidates[0].content.parts[0].text if judge_resp.candidates else ""
    )
    try:
        final = json.loads(judge_text)
        if not isinstance(final, dict) or "ruling" not in final:
            raise ValueError("Unexpected structure")
    except Exception:
        final = {
            "ruling": judge_text.strip(),
            "reasoning": ["Synthesis returned non-JSON; using raw text."],
            "confidence": 6
        }

    # 3) Brief summary for quick display
    brief_prompt = f"""Summarize the final ruling in 1–2 sentences for a DM. 
Question: {question}
Final ruling: {final.get('ruling','')}
Return plain text only."""
    brief_resp = client.models.generate_content(
        model=model,
        contents=brief_prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=120
        )
    )
    brief_text = getattr(brief_resp, "text", None) or (
        brief_resp.candidates[0].content.parts[0].text if brief_resp.candidates else ""
    )

    return {
        "brief": {"summary": brief_text.strip()},
        "proposals": proposals,
        "final": final
    }


# ================================
# Streamlit UI
# ================================

def main():
    st.title("PDF Q&A + D&D Council")
    st.write("Choose a mode in the sidebar to either chat with your PDFs or ask the D&D 5e council.")

    # ---------------------------
    # Sidebar controls
    # ---------------------------
    with st.sidebar:
        st.header("Mode & Settings")
        mode = st.radio("Answer mode", ["PDF RAG (FLAN-T5)", "D&D Council (Gemini)"])

        # Common generation controls (where applicable)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)

        if mode == "PDF RAG (FLAN-T5)":
            st.subheader("Chunking & Retrieval")
            chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=500, step=50)
            chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=100, step=25)
            top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=3)
            max_new_tokens = st.slider("Max new tokens", min_value=16, max_value=512, value=128, step=16)
        else:
            st.subheader("Gemini Settings")
            gemini_key = st.text_input("Gemini API key (optional if set in env)", type="password")
            gemini_model = st.text_input("Model", value="gemini-1.5-flash")
            # Council typically uses a slightly cooler temperature; we reuse slider value but cap below.
            council_temp = min(temperature, 0.9)

    # ---------------------------
    # Mode A: PDF RAG
    # ---------------------------
    if mode == "PDF RAG (FLAN-T5)":
        uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

        if uploaded_files:
            # Fingerprint so we only rebuild when inputs change
            file_keys = tuple((f.name, f.size, f.type) for f in uploaded_files)
            if "_index_key" not in st.session_state or st.session_state.get("_index_key") != (file_keys, chunk_size, chunk_overlap):
                with st.spinner("Reading and indexing PDFs…"):
                    documents = load_pdfs(uploaded_files)
                    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    st.session_state.vector_store = build_vector_store(chunks)
                    st.session_state._index_key = (file_keys, chunk_size, chunk_overlap)
                st.success("PDFs indexed. You can start chatting!")

            # Build (or fetch cached) generator pipeline with current temperature
            generator = get_generator(temperature=temperature)
            llm = HuggingFacePipeline(pipeline=generator)

            user_input = st.text_input("Ask a question about your PDFs:")
            if user_input:
                docs = retrieve_context(st.session_state.vector_store, user_input, k=top_k)

                # Optionally show sources
                with st.expander("Show retrieved context (sources)"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(
    f"**{i}. {d.metadata.get('file_name','(unknown)')}**\n\n"
    f"{d.page_content[:700]}…"
)

                prompt = build_prompt(docs, user_input)

                try:
                    outputs = generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)
                    response = outputs[0]["generated_text"]
                    st.write(response)
                except torch.cuda.OutOfMemoryError:
                    st.error("Out of memory. Try a smaller model, fewer PDFs, or lower max_new_tokens.")
        else:
            st.info("Upload one or more PDFs to begin.")

    # ---------------------------
    # Mode B: D&D Council (Gemini)
    # ---------------------------
    else:
        st.subheader("D&D 5e Council")
        question = st.text_area("Ask a D&D rules/lore question:")
        go = st.button("Run Council")

        if go and question.strip():
            try:
                with st.spinner("Consulting the council…"):
                    result = run_council(question, api_key=gemini_key if gemini_key else None, model=gemini_model, temperature=council_temp)
            except Exception as e:
                st.error(f"Council failed: {e}")
            else:
                # Brief summary
                st.markdown("### Brief")
                st.write(result.get("brief", {}).get("summary", ""))

                # Proposals
                st.markdown("### Council Proposals (5 personas)")
                for p in result.get("proposals", []):
                    with st.expander(f"{p.get('persona','(persona)')} — confidence {p.get('confidence','?')}"):
                        st.write(p.get("ruling", ""))

                # Final ruling
                st.markdown("### Final Ruling")
                final = result.get("final", {})
                st.write(final.get("ruling", ""))
                reasons = final.get("reasoning", [])
                if reasons:
                    st.markdown("**Why this ruling**")
                    for r in reasons:
                        st.markdown(f"- {r}")
                st.markdown(f"**Confidence:** {final.get('confidence','?')}")
        else:
            st.info("Enter a question and press *Run Council*.")


if __name__ == "__main__":
    main()
