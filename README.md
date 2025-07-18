🧙‍♂️ D&D Chatbot Assistant

Your AI‑powered Game Master Sidekick

Welcome to the Dungeons & Dragons Chatbot Assistant — a Python‑powered project that helps Dungeon Masters run smoother, faster, and more immersive games. From dice rolls 🎲 to spell look‑ups 🔮, character & NPC generation 👤, and full combat/initiative tracking ⚔️, this repo has you covered.

⸻

🚀  Quick Start

1. Clone the repo

$ git clone https://github.com/yourname/dnd-chatbot-assistant.git $ cd dnd-chatbot-assistant

2. Create virtual environment (optional but recommended)

$ python -m venv venv && source venv/bin/activate

3. Install dependencies

$ pip install -r requirements.txt

4. Run the CLI prototype

$ python main.py

Note ⬇️  Add your OpenAI key (or other model credentials) to .env:

OPENAI_API_KEY=sk-...

⸻

✨  Key Features

Module What it Does Dice Roller 🎲 Parses expressions like 2d6+3, handles advantage/disadvantage Spell Lookup 🔮 Fast SRD search, fuzzy‑match, rich formatting Character Generator 🧑‍🚀 Random or guided PC sheets (stats, gear, spells) NPC Generator 🧑‍🤝‍🧑 Flavorful NPCs with quirks, motivations & mini stat‑blocks Combat Tracker ⚔️ Initiative order, HP, status effects, round logs Map/Token Tools 🗺️ Lightweight grid map & token manager (Streamlit canvas) VTT Integration 🔄 Optional hooks for FoundryVTT / Roll20 APIs

⸻

🏗️  Project Structure

📦 dnd-chatbot-assistant ├─ agent/ # LLM wrappers & prompt templates ├─ modules/ # Dice, spells, characters, NPCs, combat, map ├─ data/ # JSON/YAML rule data (SRD) ├─ ui/ # Streamlit or Textual interfaces ├─ tests/ # Unit & integration tests ├─ requirements.txt # Third‑party libraries └─ README.md # ← you’re here!

⸻

⚡  Installation Details 1. Python 3.10+ is recommended. 2. The default LLM backend is OpenAI GPT‑4o; swap in a local model (e.g., Ollama) by editing agent/llm_interface.py. 3. Map features use Streamlit Elements. Install Node (≥ 18) if you plan to hack on the frontend bundle.

⸻

🎮  Usage Examples

roll 2d6+3 🎲 You rolled 11 (4 + 4 + 3)
!spell Misty Step 🔮 Misty Step — 2nd‑level Conjuration (Bonus Action, 30 ft teleport, self‑only)
generate npc grumpy elf librarian 👤 Delra Moonwhisper — AC 12 HP 9 | Muttering about overdue tomes 📚
start combat goblin 4, thane ⚔️ Initiative order: Thane (18), Goblin #1 (14), Goblin #3 (14 DEX tie‑break), ...
⸻

🗺️  Roadmap

Phase Milestone Status 1 Core foundation (env, data load, dice roller, basic LLM) ✅ Done 2 Game modules (spells, PC/NPC gen, combat helper) ✅ Done 3 Memory, persistence, Streamlit UI 🔄 In progress 4 Initiative tracker, map tools, VTT integration 🛠️ Planned

See docs/roadmap.md for detailed tasks & timelines.

⸻

🛠  Tech Stack • Python 3.10 🐍 • LangChain 🤖 — prompt orchestration & memory • OpenAI / LLaMA‑compatible LLMs 🧠 • Pydantic ✅ — data validation • Streamlit 🌐 — web UI • TinyDB / SQLite 💾 — lightweight persistence • pytest 🧪 — testing

⸻

🤝  Contributing 1. Fork the repo & create your branch: git checkout -b feature/my-new-feature 2. Commit your changes: git commit -am 'Add some feature' 3. Push to the branch: git push origin feature/my-new-feature 4. Open a Pull Request.

Please run pytest and black . before pushing. We love clean code! 🧼

⸻

📄  License

This project is released under the MIT License — see LICENSE for details.

⸻

🌟  Acknowledgements • Wizards of the Coast for the D&D 5e SRD 📚 • OpenAI & the open‑source LLM community 🤗 • Streamlit & Textual maintainers for awesome UI tooling 🎨
