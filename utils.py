"""
utils.py
--------
Helper functions for the RAG chatbot:
- Loading and parsing PDF documents
- Splitting documents into chunks
- Building embeddings (HuggingFace by default, OpenAI optional)
- Creating / loading a persistent Chroma vector store
- Building the conversational RAG chain (Claude or OpenAI)

The functions here are intentionally framework-agnostic so the Streamlit
layer in `app.py` stays thin and readable.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import List, Literal

from dotenv import load_dotenv

# --- LangChain core imports -------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Vector store -----------------------------------------------------------
from langchain_chroma import Chroma

# Load environment variables from a local .env file (API keys, etc.)
load_dotenv()

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------
# Local directory where Chroma persists its data between runs.
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
# Default open-source embedding model (fast + good quality, runs locally).
DEFAULT_HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Collection name inside Chroma.
COLLECTION_NAME = "rag_documents"

LLMProvider = Literal["anthropic", "openai"]
EmbedProvider = Literal["huggingface", "openai"]


# ---------------------------------------------------------------------------
# 1. Document loading
# ---------------------------------------------------------------------------
def load_pdfs(uploaded_files) -> List[Document]:
    """Load one or more uploaded PDF files into LangChain `Document` objects.

    Streamlit gives us in-memory file buffers, so we write each one to a
    temporary file on disk (PyPDFLoader needs a real path) and then load it.

    Args:
        uploaded_files: list of Streamlit `UploadedFile` objects.

    Returns:
        A flat list of `Document` objects, one per PDF page, with metadata
        ("source" = original filename, "page" = page number).
    """
    documents: List[Document] = []

    for uploaded_file in uploaded_files:
        # Persist the upload to a temp file so PyPDFLoader can read it.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()

            # Overwrite the "source" metadata with the friendly original name
            # so citations show "report.pdf" instead of a temp path.
            for page in pages:
                page.metadata["source"] = uploaded_file.name
            documents.extend(pages)
        finally:
            # Always clean up the temp file.
            os.remove(tmp_path)

    return documents


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into overlapping chunks for better retrieval.

    A chunk size of ~1000 chars with 200 char overlap is a solid default:
    large enough to keep context, small enough for precise retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split on natural boundaries first, falling back gradually.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# 3. Embeddings
# ---------------------------------------------------------------------------
def get_embeddings(provider: EmbedProvider = "huggingface"):
    """Return an embedding model instance.

    - "huggingface": local sentence-transformers model (no API cost).
    - "openai": OpenAI's text-embedding-3-small (requires OPENAI_API_KEY).
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")

    # Default: local HuggingFace embeddings.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=DEFAULT_HF_EMBED_MODEL,
        # Normalised embeddings work better with cosine similarity.
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# 4. Vector store (Chroma, persistent)
# ---------------------------------------------------------------------------
def build_vectorstore(
    chunks: List[Document],
    embed_provider: EmbedProvider = "huggingface",
) -> Chroma:
    """Create (or overwrite) a persistent Chroma vector store from chunks."""
    embeddings = get_embeddings(embed_provider)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def load_vectorstore(embed_provider: EmbedProvider = "huggingface") -> Chroma | None:
    """Load an existing persistent Chroma store, if one exists on disk."""
    if not os.path.isdir(CHROMA_DIR):
        return None

    embeddings = get_embeddings(embed_provider)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def reset_vectorstore() -> bool:
    """Delete the persisted Chroma store from disk.

    Returns True if a store existed and was removed, False otherwise.
    Used by the "Reset knowledge base" button in the UI.
    """
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        return True
    return False


# ---------------------------------------------------------------------------
# 5. LLM factory
# ---------------------------------------------------------------------------
def get_llm(provider: LLMProvider = "anthropic", temperature: float = 0.3):
    """Return a chat LLM instance for the chosen provider.

    Temperature 0.3 keeps answers factual and grounded in the documents.
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            streaming=True,
        )

    # Default: Anthropic Claude 3.5 Sonnet.
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        temperature=temperature,
        streaming=True,
    )


# ---------------------------------------------------------------------------
# 6. RAG chain
# ---------------------------------------------------------------------------
# System prompt that grounds the model in the retrieved context and tells it
# how to behave (cite, stay factual, admit when it doesn't know).
SYSTEM_PROMPT = """You are a helpful, precise assistant that answers questions \
using ONLY the provided context from the user's documents.

Guidelines:
- Base your answer strictly on the context below. Do not invent facts.
- If the answer is not in the context, say clearly that you don't know based \
on the provided documents.
- Be concise but complete. Use bullet points when it improves clarity.
- When helpful, refer to the source document names.

Context:
{context}
"""


def _format_docs(docs: List[Document]) -> str:
    """Join retrieved chunks into a single context string for the prompt."""
    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[{i}] (source: {source}, page: {page})\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(vectorstore: Chroma, llm, k: int = 4):
    """Build a conversational RAG chain.

    The chain:
      1. Retrieves the top-k relevant chunks for the question.
      2. Formats them into the system prompt context.
      3. Includes prior chat history for follow-up questions.
      4. Streams the LLM answer.

    Returns a tuple of (chain, retriever) so the caller can also fetch the
    source documents separately for the citations UI.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # LCEL chain: retrieve -> format context -> prompt -> llm -> string.
    chain = (
        {
            "context": lambda x: _format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def retrieve_sources(retriever, question: str) -> List[Document]:
    """Fetch the source documents used to answer a question (for citations)."""
    return retriever.invoke(question)
