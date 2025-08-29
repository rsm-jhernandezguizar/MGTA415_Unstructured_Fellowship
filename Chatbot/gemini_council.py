# %%
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
# assign API key above

# %%
import google.genai as genai
import os
import json
from typing import List, Dict

# %%
# ------------- Gemini client helper -----------------
def _gemini_client(api_key: str | None = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing Gemini API key: set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment, "
            "or pass api_key=... to run_council()."
        )
    return genai.Client(api_key=key)

# %%
# ------------- Core prompting helpers ----------------
_PERSONAS: List[str] = [
    "RAW Literalist (rules-as-written, cite page numbers when sure)",
    "RAI Interpreter (rules-as-intended, weigh designer intent and Sage Advice)",
    "Table Balance DM (fairness and pacing; minimize swingy outcomes)",
    "Narrative DM (cinematic flow; reward setup like hiding/ambush)",
    "System Lawyer (edge cases, timing windows, conditions, and advantage rules)"
]

_SYSTEM_PREAMBLE = """You are an expert D&D 5e adjudication panel. 
Answer precisely and avoid homebrew unless asked. Distinguish RAW vs RAI when relevant.
If you are uncertain on an exact page reference, say so explicitly rather than guessing."""

def _persona_prompt(persona: str, question: str) -> str:
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

# ------------- Public API ----------------------------
def run_council(
    question: str,
    *,
    api_key: str | None = None,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.4
) -> Dict[str, object]:
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



