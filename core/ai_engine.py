"""
AI Engine for GrantMind AI.
Interfaces with Anthropic's Claude to answer questions, summarize, and compare.
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLAUDE_MODEL = "claude-sonnet-4-20250514"
FALLBACK_MODEL = "claude-3-5-sonnet-20241022"

def get_client() -> Anthropic:
    """
    Initializes and returns the Anthropic client.

    Returns:
        Anthropic: The configured Anthropic client.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise ValueError("ANTHROPIC_API_KEY is not configured in your .env file.")
    return Anthropic(api_key=api_key)

def call_claude(prompt: str, system: str, max_tokens: int = 1500) -> str:
    """
    Helper function to send a message to Claude with model fallback support.
    """
    client = get_client()
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as exc:
        # Fallback to standard Claude 3.5 Sonnet if newer version has access limits
        if "model_not_found" in str(exc) or "permission" in str(exc):
            response = client.messages.create(
                model=FALLBACK_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        raise exc

def format_context(relevant_chunks: List[Dict[str, Any]]) -> str:
    """
    Formats the search results into a clean XML-styled context block.
    """
    blocks = []
    for chunk in relevant_chunks:
        meta = chunk["metadata"]
        header = f"[Source: {meta['filename']}, Page {meta['page_number']}]"
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)

def extract_citations(relevant_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts a unique, deduplicated list of source files and pages.
    """
    seen = set()
    citations = []
    for chunk in relevant_chunks:
        meta = chunk["metadata"]
        cite_key = (meta["filename"], meta["page_number"])
        if cite_key not in seen:
            seen.add(cite_key)
            citations.append({
                "filename": meta["filename"],
                "page_number": meta["page_number"]
            })
    return citations

def answer_question(question: str, relevant_chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Queries Claude to answer a question based strictly on context.
    """
    context_str = format_context(relevant_chunks)
    system_prompt = (
        "You are GrantMind, an expert AI assistant for NGOs and humanitarian organizations. "
        "You analyze grant proposals, donor reports, and policy documents. "
        "Answer questions accurately based ONLY on the provided document context. "
        "Always cite which document and page your answer comes from in your response. "
        "If the answer is not in the documents, say so clearly."
    )
    prompt_content = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"
    answer_text = call_claude(prompt_content, system_prompt)
    citations = extract_citations(relevant_chunks)
    return answer_text, citations

def summarize_document(chunks: List[Dict[str, Any]]) -> str:
    """
    Generates a structured bullet-point summary of a single document.
    """
    text_content = "\n\n".join([c["text"] for c in chunks[:12]]) # Limit to first 12 chunks (~6000 words) for safety
    system_prompt = (
        "You are GrantMind, an expert AI assistant for NGOs. "
        "Summarize the provided document context using exact Markdown bullet points."
    )
    prompt_content = (
        "Analyze this document and summarize it. Extract and output exactly these sections:\n"
        "- **Main Objective**:\n"
        "- **Budget Mentioned**:\n"
        "- **Target Beneficiaries**:\n"
        "- **Key Deliverables**:\n"
        "- **Geographic Focus**:\n"
        "- **Key Risks** (if mentioned, otherwise 'None mentioned'):\n\n"
        f"DOCUMENT TEXT:\n{text_content}"
    )
    return call_claude(prompt_content, system_prompt, max_tokens=1000)

def clean_json_response(raw_text: str) -> List[Dict[str, Any]]:
    """
    Regex cleans and parses a JSON array from Claude's response.
    """
    match = re.search(r"\[\s*\{.*\}\s*\]", raw_text, re.DOTALL)
    if not match:
        raise ValueError("Could not extract clean JSON array from response.")
    return json.loads(match.group(0))

def compare_documents(docs_data: Dict[str, List[Dict[str, Any]]], dimension: str) -> List[Dict[str, Any]]:
    """
    Compares multiple documents along a single user-specified dimension.
    """
    doc_summaries = []
    for filename, chunks in docs_data.items():
        doc_text = "\n\n".join([c["text"] for c in chunks[:6]]) # Sample beginning of doc
        doc_summaries.append(f"--- Document: {filename} ---\n{doc_text}")
    
    docs_block = "\n\n".join(doc_summaries)
    system_prompt = "You are a data extraction assistant that returns clean JSON only."
    prompt_content = (
        f"Compare these documents on the dimension: '{dimension}'.\n"
        "Return a raw JSON array of objects (no markdown fences, no conversational text) "
        "where each object represents a document and contains exactly two keys:\n"
        '- "Document": the filename\n'
        f'- "Comparison Details": a concise 2-sentence extraction of the document\'s position on {dimension}.\n\n'
        f"DOCUMENTS:\n{docs_block}"
    )
    raw_response = call_claude(prompt_content, system_prompt, max_tokens=1000)
    return clean_json_response(raw_response)
