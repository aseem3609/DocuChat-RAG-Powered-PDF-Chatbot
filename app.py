"""
app.py
------
Streamlit chat interface for the RAG chatbot.

Features:
- Upload one or multiple PDFs from the sidebar.
- "Process Documents" button: chunk -> embed -> store in Chroma.
- Chat interface with streaming responses.
- Conversation memory that persists for the session.
- "Clear chat" button.
- Toggle between Claude (Anthropic) and GPT-4o (OpenAI).
- "Sources" expander showing the chunks + page numbers behind each answer.
"""

import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from utils import (
    build_rag_chain,
    build_vectorstore,
    get_llm,
    load_pdfs,
    load_vectorstore,
    reset_vectorstore,
    retrieve_sources,
    split_documents,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Set up all the session-state keys we rely on (once per session)."""
    defaults = {
        "messages": [],          # list of {"role", "content"} dicts for display
        "chat_history": [],      # list of LangChain message objects for the LLM
        "vectorstore": None,     # the Chroma instance
        "documents_processed": False,
        "processed_files": [],   # names of files indexed (for sidebar display)
        "num_chunks": 0,         # how many chunks are in the index
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------------------------
# Sidebar: settings + document upload/processing
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    # --- LLM provider toggle (easy switch between Claude and GPT-4o) --------
    llm_choice = st.radio(
        "Choose your LLM",
        options=["Claude 3.5 Sonnet", "GPT-4o"],
        index=0,
        help="Switch between Anthropic Claude and OpenAI GPT-4o.",
    )
    llm_provider = "anthropic" if llm_choice.startswith("Claude") else "openai"

    # --- Embedding provider toggle -----------------------------------------
    embed_choice = st.radio(
        "Embeddings",
        options=["HuggingFace (local, free)", "OpenAI"],
        index=0,
        help="HuggingFace runs locally with no API cost. OpenAI needs a key.",
    )
    embed_provider = "huggingface" if embed_choice.startswith("HuggingFace") else "openai"

    # --- Retrieval setting --------------------------------------------------
    top_k = st.slider(
        "Chunks to retrieve (k)",
        min_value=2,
        max_value=10,
        value=4,
        help="How many document chunks to feed the model as context.",
    )

    # --- Generation setting -------------------------------------------------
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more factual/deterministic. Higher = more creative.",
    )

    # --- Advanced chunking controls ----------------------------------------
    with st.expander("🔧 Advanced chunking"):
        chunk_size = st.slider(
            "Chunk size",
            min_value=300,
            max_value=2000,
            value=1000,
            step=100,
            help="Characters per chunk. Larger keeps more context.",
        )
        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Overlap between consecutive chunks.",
        )

    st.divider()
    st.header("📄 Documents")

    # --- Multi-file PDF upload ---------------------------------------------
    uploaded_files = st.file_uploader(
        "Upload PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # --- Process documents button ------------------------------------------
    if st.button("🚀 Process Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            try:
                with st.spinner("Reading, chunking and embedding your documents..."):
                    # 1. Load PDFs -> 2. Chunk -> 3. Embed + store in Chroma.
                    raw_docs = load_pdfs(uploaded_files)
                    chunks = split_documents(
                        raw_docs,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    st.session_state.vectorstore = build_vectorstore(
                        chunks, embed_provider=embed_provider
                    )
                    st.session_state.documents_processed = True
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.session_state.num_chunks = len(chunks)
                st.success(
                    f"Processed {len(uploaded_files)} file(s) into "
                    f"{len(chunks)} chunks. You can start chatting!"
                )
            except Exception as exc:  # noqa: BLE001 - surface any error to the UI
                st.error(f"Failed to process documents: {exc}")

    # --- Try to reuse a previously persisted store -------------------------
    if not st.session_state.documents_processed:
        existing = load_vectorstore(embed_provider=embed_provider)
        if existing is not None:
            st.session_state.vectorstore = existing
            st.session_state.documents_processed = True
            st.info("Loaded an existing document index from disk.")

    # --- Knowledge base status ---------------------------------------------
    if st.session_state.documents_processed:
        st.success("✅ Knowledge base ready")
        if st.session_state.processed_files:
            with st.expander(
                f"📑 Indexed files ({len(st.session_state.processed_files)})"
            ):
                for name in st.session_state.processed_files:
                    st.markdown(f"- {name}")
                if st.session_state.num_chunks:
                    st.caption(f"{st.session_state.num_chunks} chunks indexed.")

    st.divider()

    # --- Chat actions: export + clear + reset ------------------------------
    # Export the current conversation as a downloadable Markdown transcript.
    if st.session_state.messages:
        transcript = "\n\n".join(
            f"**{m['role'].capitalize()}:** {m['content']}"
            for m in st.session_state.messages
        )
        st.download_button(
            "⬇️ Export chat",
            data=transcript,
            file_name="chat_history.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # --- Clear chat button --------------------------------------------------
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    # --- Reset knowledge base (delete the persisted Chroma store) ----------
    if st.button("💣 Reset knowledge base", use_container_width=True):
        reset_vectorstore()
        st.session_state.vectorstore = None
        st.session_state.documents_processed = False
        st.session_state.processed_files = []
        st.session_state.num_chunks = 0
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.success("Knowledge base cleared.")
        st.rerun()


# ---------------------------------------------------------------------------
# Main area: header + chat
# ---------------------------------------------------------------------------
st.title("🤖 RAG Chatbot")
st.caption(
    "Ask questions about your uploaded PDFs. Answers are grounded in your "
    "documents, with sources shown for every response."
)

# Render the existing conversation so it persists across reruns.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Re-render the sources expander for past assistant answers.
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                for i, src in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**[{i}] {src['source']} — page {src['page']}**\n\n"
                        f"> {src['snippet']}"
                    )


# ---------------------------------------------------------------------------
# Chat input + response generation
# ---------------------------------------------------------------------------
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    # Guard: make sure documents have been processed.
    if not st.session_state.documents_processed or st.session_state.vectorstore is None:
        st.warning("Please upload and process documents before chatting.")
        st.stop()

    # 1. Show the user's message immediately.
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 2. Generate and stream the assistant's answer.
    with st.chat_message("assistant"):
        try:
            llm = get_llm(provider=llm_provider, temperature=temperature)
            chain, retriever = build_rag_chain(
                st.session_state.vectorstore, llm, k=top_k
            )

            # Retrieve sources first so we can show citations.
            source_docs = retrieve_sources(retriever, user_question)

            # Stream the response token-by-token for a live typing effect.
            response = st.write_stream(
                chain.stream(
                    {
                        "question": user_question,
                        "chat_history": st.session_state.chat_history,
                    }
                )
            )

            # 3. Build a compact, serialisable list of sources for the UI.
            sources = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "snippet": doc.page_content[:300].strip() + "...",
                }
                for doc in source_docs
            ]

            # 4. Show the sources expander.
            if sources:
                with st.expander("📚 Sources"):
                    for i, src in enumerate(sources, start=1):
                        st.markdown(
                            f"**[{i}] {src['source']} — page {src['page']}**\n\n"
                            f"> {src['snippet']}"
                        )

            # 5. Persist the assistant message + sources for re-rendering.
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": sources}
            )

            # 6. Update the LLM-facing chat history (for follow-up questions).
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=response))

        except Exception as exc:  # noqa: BLE001 - surface any error to the UI
            st.error(f"Something went wrong while generating the answer: {exc}")
