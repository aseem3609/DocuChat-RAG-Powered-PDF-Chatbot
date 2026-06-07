# 🤖 RAG Chatbot (LangChain + Chroma + Streamlit)

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot. Upload your
PDFs, and ask questions about them in a clean chat interface. Answers are
grounded in your documents and come with **source citations** (file name + page).

Switch between **Claude 3.5 Sonnet** and **GPT-4o** with a single toggle.

---

## ✨ Features

- 📄 **Multiple PDF upload** from the sidebar.
- 🚀 **One-click processing** — chunk → embed → store in a persistent Chroma DB.
- 💬 **Streaming chat** responses (live typing effect).
- 🧠 **Conversation memory** — supports follow-up questions during the session.
- 📚 **Source citations** — a "Sources" expander with chunks + page numbers.
- 🔀 **LLM toggle** — Claude 3.5 Sonnet ⇄ GPT-4o.
- 🧮 **Embeddings toggle** — local HuggingFace (free) ⇄ OpenAI.
- 🗑️ **Clear chat** button.
- 🛡️ **Error handling** throughout the UI.

---

## 🗂️ Project Structure

```
chatbot/
├── app.py            # Streamlit chat UI
├── utils.py          # Loading, chunking, embedding, vector store, RAG chain
├── requirements.txt  # Python dependencies
├── .env.example      # Template for API keys
└── README.md         # This file
```

---

## ⚙️ Tech Stack

| Layer            | Choice                                            |
| ---------------- | ------------------------------------------------- |
| UI               | Streamlit                                         |
| Orchestration    | LangChain (LCEL)                                   |
| Vector DB        | Chroma (persistent, local)                        |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` (default)|
| LLM              | Claude 3.5 Sonnet **or** GPT-4o                    |
| PDF parsing      | `pypdf` via `PyPDFLoader`                          |

---

## 🚀 Installation & Run

> Run all commands from inside the `chatbot/` folder.

### 1. Create & activate a virtual environment

**Windows (PowerShell):**
```powershell
cd chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
cd chatbot
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API keys

Copy the template and fill in your key(s):

**Windows:**
```powershell
copy .env.example .env
```

**macOS/Linux:**
```bash
cp .env.example .env
```

Then edit `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...     # needed for Claude
OPENAI_API_KEY=sk-...            # needed for GPT-4o / OpenAI embeddings
```

> You only need the key for the provider(s) you actually use. If you stick with
> the default **HuggingFace embeddings + Claude**, you only need
> `ANTHROPIC_API_KEY`.

- Get an Anthropic key: https://console.anthropic.com/
- Get an OpenAI key: https://platform.openai.com/api-keys

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🧭 How to Use

1. In the **sidebar**, pick your **LLM** and **embeddings** provider.
2. **Upload** one or more PDFs.
3. Click **🚀 Process Documents** and wait for the success message.
4. Ask questions in the chat box at the bottom.
5. Expand **📚 Sources** under any answer to see the exact chunks + pages used.
6. Use **🗑️ Clear chat** to reset the conversation.

> The Chroma index is persisted to `chatbot/chroma_db/`, so processed documents
> are reloaded automatically the next time you launch the app.

---

## 📸 Screenshots

> Add your own screenshots here after running the app.

| Chat interface | Sources expander |
| -------------- | ---------------- |
| _add screenshot_ | _add screenshot_ |

---

## 🛠️ Configuration Notes

- **Chunking:** `RecursiveCharacterTextSplitter`, `chunk_size=1000`,
  `chunk_overlap=200` (tune in `utils.py`).
- **Temperature:** `0.3` for factual, grounded answers.
- **Retrieval `k`:** adjustable from the sidebar (default `4`).

---

## 🚧 CV-Boosting Improvements (Suggested Next Steps)

Want to make this project stand out? Here are four high-impact extensions:

1. **🧩 Agentic RAG (tool-using agent).** Wrap retrieval as a LangGraph agent
   that can decide *when* to search, call multiple tools (web search,
   calculator, SQL), and self-correct with a "grade documents → rewrite query →
   retry" loop. Demonstrates agent design, not just a static chain.

2. **📊 RAG Evaluation pipeline.** Add automatic evaluation with **RAGAS** or
   **DeepEval** measuring faithfulness, answer relevancy, and context precision/
   recall. Log results to a dashboard. Shows you can *measure* quality, which
   employers love.

3. **🔁 Advanced retrieval.** Implement **hybrid search** (BM25 + dense vectors)
   with a **re-ranker** (e.g. Cohere Rerank or a cross-encoder) and
   **query expansion / multi-query retrieval**. Big accuracy wins and signals
   depth in information retrieval.

4. **🏗️ Productionisation.** Containerise with **Docker**, add a **FastAPI**
   backend separating the API from the UI, stream over WebSockets, add
   per-user document collections + auth, and deploy to a cloud host with CI/CD.
   Demonstrates end-to-end engineering maturity.

Bonus ideas: conversation summarisation memory, citation highlighting in the
original PDF, support for `.docx`/`.txt`/web pages, and caching of embeddings.

---

