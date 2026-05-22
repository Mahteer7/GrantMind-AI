"""
GrantMind AI - Streamlit Web Application.
Main frontend coordinating PDF processor, local embedding model, ChromaDB, and Claude.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from core import pdf_processor, embedder, vector_store, ai_engine

def inject_custom_css() -> None:
    """
    Injects custom responsive CSS to design an elegant dark-theme UI.
    """
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.8rem; font-weight: 700;
            background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subtitle { font-size: 1.1rem; color: #9ca3af; margin-bottom: 1.8rem; }
        .answer-card {
            background-color: #1e293b; border-left: 5px solid #3b82f6;
            border-radius: 8px; padding: 20px; margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .citation-badge {
            background-color: #334155; color: #60a5fa; border: 1px solid #475569;
            border-radius: 9999px; padding: 4px 12px; font-size: 0.8rem;
            font-weight: 500; margin-right: 8px; display: inline-block;
        }
        .chat-bubble-user {
            background-color: #475569; padding: 12px 16px; border-radius: 12px 12px 0 12px;
            margin-bottom: 0.8rem; width: fit-content; max-width: 80%; margin-left: auto;
        }
        .chat-bubble-bot {
            background-color: #1e293b; padding: 12px 16px; border-radius: 12px 12px 12px 0;
            margin-bottom: 0.8rem; width: fit-content; max-width: 80%;
            border: 1px solid #334155;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def sync_chroma_state() -> None:
    """
    Synchronizes local session state with the persistent ChromaDB collection.
    """
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = []
    if "docs_metadata" not in st.session_state:
        st.session_state.docs_metadata = {}
    try:
        col = vector_store.get_collection()
        data = col.get(include=["metadatas", "documents"])
        if data and data["metadatas"]:
            reconstructed: Dict[str, List[Dict[str, Any]]] = {}
            for doc, meta in zip(data["documents"], data["metadatas"]):
                fname = meta["filename"]
                if fname not in reconstructed:
                    reconstructed[fname] = []
                reconstructed[fname].append({"text": doc, "metadata": meta})
            st.session_state.docs_metadata = reconstructed
            st.session_state.processed_docs = list(reconstructed.keys())
    except Exception:
        pass

def init_session_state() -> None:
    """
    Initializes required chat history and setup states.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    sync_chroma_state()

def clear_all_documents() -> None:
    """
    Resets the ChromaDB collection and clears the local session state.
    """
    with st.spinner("Clearing local vector database..."):
        try:
            vector_store.clear_collection()
            st.session_state.processed_docs = []
            st.session_state.docs_metadata = {}
            st.session_state.chat_history = []
            st.toast("🗑️ Database successfully reset!", icon="✅")
        except Exception as exc:
            st.error(f"Failed to clear database: {exc}")

def handle_file_processing(uploaded_files: List[Any]) -> None:
    """
    Processes the uploaded PDF documents and inserts chunks to database.
    """
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_docs]
    if not new_files:
        st.info("Uploaded files have already been processed.")
        return
        
    p_bar = st.progress(0.0)
    for idx, f_obj in enumerate(new_files):
        with st.spinner(f"Extracting & chunking {f_obj.name}..."):
            chunks = pdf_processor.process_pdf(f_obj, f_obj.name)
        if not chunks:
            st.warning(f"⚠️ {f_obj.name} contains no readable text (could be a scanned image PDF).")
            continue
            
        with st.spinner(f"Generating dense local embeddings for {f_obj.name}..."):
            texts = [c["text"] for c in chunks]
            embeddings = embedder.embed_texts(texts)
            vector_store.insert_chunks(chunks, embeddings)
            st.session_state.docs_metadata[f_obj.name] = chunks
            st.session_state.processed_docs.append(f_obj.name)
        p_bar.progress((idx + 1) / len(new_files))
        
    tot_chunks = sum(len(c) for c in st.session_state.docs_metadata.values())
    st.success(f"Processed {len(new_files)} documents | {tot_chunks} chunks | Ready to query!")

def render_sidebar() -> None:
    """
    Renders sidebar elements including logo, upload forms, and loaded states.
    """
    with st.sidebar:
        st.markdown("## 🧠 GrantMind AI")
        st.markdown("*Humanitarian Document RAG Engine*")
        st.divider()
        
        uploaded = st.file_uploader(
            "Upload PDFs (Proposals, Reports, Policies)", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        if uploaded and st.button("🚀 Process Documents", use_container_width=True):
            handle_file_processing(uploaded)
            
        st.divider()
        if st.session_state.processed_docs:
            st.markdown("### 📂 Indexed Documents:")
            for doc in st.session_state.processed_docs:
                st.markdown(f"- 📄 {doc}")
            if st.button("🗑️ Reset Database", type="primary", use_container_width=True):
                clear_all_documents()
        else:
            st.info("No documents indexed. Upload PDFs to begin.")

def render_chips() -> str:
    """
    Displays horizontal clickable sample question chips.
    """
    clicked = ""
    st.markdown("💡 **Sample Queries:**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("💰 Budget requested?", use_container_width=True):
            clicked = "What is the total budget requested?"
    with c2:
        if st.button("👥 Target Beneficiaries?", use_container_width=True):
            clicked = "Who are the target beneficiaries?"
    with c3:
        if st.button("📦 Deliverables?", use_container_width=True):
            clicked = "What are the key deliverables?"
    with c4:
        if st.button("🌍 Geographic Areas?", use_container_width=True):
            clicked = "Which geographic areas are covered?"
    return clicked

def run_query(query_str: str) -> None:
    """
    Executes similarity search and calls Claude for answers.
    """
    if not st.session_state.processed_docs:
        st.warning("⚠️ No documents indexed yet. Please upload and process documents in the sidebar first.")
        return
    with st.spinner("Searching database & generating cited answer..."):
        try:
            q_emb = embedder.embed_query(query_str)
            hits = vector_store.search_similar(q_emb, top_k=5)
            if not hits:
                st.warning("No matching context found. Try a different query.")
                return
            ans, cites = ai_engine.answer_question(query_str, hits)
            st.session_state.chat_history.append({
                "question": query_str, "answer": ans, "citations": cites, "sources": hits
            })
        except Exception as exc:
            st.error(f"Engine Failure: {exc}")
            if st.button("🔄 Retry Query"):
                run_query(query_str)

def render_chat_history() -> None:
    """
    Renders scrolling chat log with custom dark bubble layouts.
    """
    for chat in st.session_state.chat_history:
        st.markdown(f'<div class="chat-bubble-user">👤 <b>You:</b> {chat["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble-bot">🤖 <b>GrantMind:</b> {chat["answer"]}</div>', unsafe_allow_html=True)
        
        # Sources expander
        with st.expander("📚 Sources Used"):
            for cite in chat["citations"]:
                st.markdown(f'<span class="citation-badge">📄 {cite["filename"]} (Page {cite["page_number"]})</span>', unsafe_allow_html=True)
            st.divider()
            for idx, src in enumerate(chat["sources"]):
                m = src["metadata"]
                st.markdown(f"**Chunk {idx+1} - {m['filename']} (Page {m['page_number']})**")
                st.caption(src["text"])

def render_qna_tab() -> None:
    """
    Drafts QA interactive panels and prompt submissions.
    """
    st.markdown("### 💬 Ask Documents Anything")
    render_chat_history()
    
    selected_chip = render_chips()
    input_query = st.chat_input("Ask anything about your documents...")
    
    final_query = selected_chip or input_query
    if final_query:
        run_query(final_query)
        st.rerun()

def render_summary_tab() -> None:
    """
    Tab 2: Generates bulleted single file summaries.
    """
    st.markdown("### 📑 Document Summaries")
    if not st.session_state.processed_docs:
        st.info("Upload and process documents to generate summaries.")
        return
        
    doc_sel = st.selectbox("Select document to summarize", st.session_state.processed_docs)
    if st.button("⚡ Generate Summary", type="primary"):
        with st.spinner("Extracting themes and generating structured summary..."):
            try:
                chunks = st.session_state.docs_metadata[doc_sel]
                summary = ai_engine.summarize_document(chunks)
                st.markdown('<div class="answer-card">', unsafe_allow_html=True)
                st.markdown(summary)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as exc:
                st.error(f"Summary failed: {exc}")

def render_comparison_tab() -> None:
    """
    Tab 3: Generates side-by-side matrices comparing files.
    """
    st.markdown("### ⚖️ Document Comparison")
    if len(st.session_state.processed_docs) < 2:
        st.info("Please process at least 2 documents to enable comparison mode.")
        return
        
    dim = st.text_input("Dimension to compare (e.g. 'budget', 'beneficiaries', 'risk factor')", value="budget")
    if st.button("🔍 Compare Documents", type="primary"):
        with st.spinner(f"Comparing documents based on '{dim}'..."):
            try:
                data = ai_engine.compare_documents(st.session_state.docs_metadata, dim)
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

def main() -> None:
    """
    Entry point of the Streamlit application.
    """
    st.set_page_config(page_title="GrantMind AI", page_icon="🧠", layout="wide")
    inject_custom_css()
    init_session_state()
    
    st.markdown('<div class="main-title">GrantMind AI 🧠</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Intelligent document intelligence for NGOs and humanitarian teams</div>', unsafe_allow_html=True)
    
    render_sidebar()
    
    t1, t2, t3 = st.tabs(["💬 Ask Questions", "📑 Document Summaries", "⚖️ Document Comparison"])
    with t1:
        render_qna_tab()
    with t2:
        render_summary_tab()
    with t3:
        render_comparison_tab()

if __name__ == "__main__":
    main()
