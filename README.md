# GrantMind AI 

An intelligent document analysis assistant specifically designed for NGOs, humanitarian organizations, and research bodies to analyze grant proposals, funding reports, and policy documents using locally embedded Retrieval-Augmented Generation (RAG) and Anthropic's Claude AI.

## Why GrantMind AI Matters
NGOs and humanitarian organizations handle vast volumes of donor reports, grant applications, field assessments, and policy guidelines. Sifting through hundreds of pages to find specific budgets, targets, beneficiary counts, or geographic regions is time-consuming. 

**GrantMind AI** enables organizations to upload a corpus of documents and interact with them in real-time, receiving accurate, fully-cited answers, comprehensive document summaries, and multi-document comparisons.

## Tech Stack
- **Web UI**: [Streamlit](https://streamlit.io/) (A beautiful, reactive Python web UI)
- **PDF Extraction**: [pdfplumber](https://github.com/jsvine/pdfplumber) (High-fidelity text extraction per page)
- **Local Embeddings**: [sentence-transformers](https://www.sbert.net/) (Generating dense vector embeddings locally using the free, fast `all-MiniLM-L6-v2` model)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (A local, serverless persistent vector store)
- **AI Reasoning**: [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) (State-of-the-art responses using Claude `claude-sonnet-4-20250514`)
- **Tabular Data**: [Pandas](https://pandas.pydata.org/) (Data structures for side-by-side document comparison matrices)

---

## File Structure
```
grantmind/
├── app.py                  # Streamlit application UI & frontend logic
├── core/
│   ├── __init__.py         # Package marker
│   ├── pdf_processor.py    # PDF page-by-page reading and chunking logic
│   ├── embedder.py         # Local sentence embeddings generator
│   ├── vector_store.py     # ChromaDB persistence, collection management, & search
│   └── ai_engine.py        # Anthropic Claude client, prompts, and summaries
├── .env                    # Environment file for API Keys
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

---

## Setup Instructions

### 1. Prerequisites
Ensure you have **Python 3.8 to 3.11** installed on your machine.

### 2. Clone and Install Dependencies
Install all package requirements:
```bash
pip install -r requirements.txt
```

### 3. Add your Anthropic API Key
Create a `.env` file in the root `grantmind/` directory (or modify the provided one) and input your key:
```env
ANTHROPIC_API_KEY=your-actual-api-key-here
```

### 4. Run the Web Application
Launch the dev server:
```bash
streamlit run app.py
```
This will automatically spin up the browser-based UI in your default browser (usually at `http://localhost:8501`).

---

## How It Works (RAG Pipeline)
1. **Document Upload**: You upload one or multiple PDF documents in the sidebar.
2. **Text Extraction & Cleaning**: `pdfplumber` extracts text from each page, cleans formatting, and removes consecutive spaces.
3. **Overlapping Chunking**: The text is split into chunks of ~500 words with a 50-word overlap to ensure boundaries don't split crucial contextual lines.
4. **Local Embedding**: Each text chunk is converted into a 384-dimensional vector using `all-MiniLM-L6-v2` running entirely on your local CPU. No document text is sent to third-party embedding APIs.
5. **ChromaDB Storage**: Chunks and their embeddings, alongside rich metadata (source filename, page number, chunk ID), are stored in the local `./chroma_db` directory.
6. **Query & Retrieval**: When you ask a question, the question is locally embedded and a vector cosine-similarity search retrieves the 5 most relevant document chunks.
7. **Claude Generation**: The retrieved text chunks are formatted as XML-like context blocks and passed with a highly tailored NGO system prompt to Anthropic Claude to formulate an accurate, cited answer.

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

## 👤 Developer
Built with ❤️ by Antigravity (AI Coding Assistant by Google DeepMind)
GitHub: [Antigravity-AI](https://github.com/Antigravity-AI)
