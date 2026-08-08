

from __future__ import annotations

import os
import shutil
import tempfile
from typing import List, Literal

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


from langchain_chroma import Chroma


load_dotenv()


CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

DEFAULT_HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Collection name inside Chroma.
COLLECTION_NAME = "rag_documents"

LLMProvider = Literal["anthropic", "openai"]
EmbedProvider = Literal["huggingface", "openai"]



def load_pdfs(uploaded_files) -> List[Document]:

    documents: List[Document] = []

    for uploaded_file in uploaded_files:
        
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



def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split on natural boundaries first, falling back gradually.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)



def get_embeddings(provider: EmbedProvider = "huggingface"):

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


def build_vectorstore(
    chunks: List[Document],
    embed_provider: EmbedProvider = "huggingface",
) -> Chroma:
    embeddings = get_embeddings(embed_provider)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def load_vectorstore(embed_provider: EmbedProvider = "huggingface") -> Chroma | None:
   
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



def get_llm(provider: LLMProvider = "anthropic", temperature: float = 0.3):

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            streaming=True,
        )


    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        temperature=temperature,
        streaming=True,
    )



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

    formatted = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[{i}] (source: {source}, page: {page})\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(vectorstore: Chroma, llm, k: int = 4):

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

   
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
    return retriever.invoke(question)
