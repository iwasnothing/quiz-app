#!/usr/bin/env python

# Avoid OpenMP conflict when FAISS + NumPy/sklearn load multiple libomp copies (macOS)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import subprocess
import shutil
import re
import math
from collections import Counter
from pathlib import Path
from database import query_all_topics
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
import random
# Optional import for RetrievalQA (legacy, may not be available in newer langchain versions)
# RetrievalQA is deprecated and moved to langchain-classic, but the import path remains the same
RETRIEVAL_QA_AVAILABLE = False
RetrievalQA = None

try:
    # Try importing from langchain.chains (works if langchain-classic is installed)
    from langchain.chains import RetrievalQA
    RETRIEVAL_QA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # RetrievalQA is not available - this is OK, it's only used for optional testing
    pass
from markitdown import MarkItDown

# Import from refactored modules
from config import settings
from llm_prompts import (
    llm,
    topic_parser,
    TOPIC_EXTRACTION_PROMPT,
    NODE_RESOLUTION_PROMPT
)
from database import (
    get_faiss_vectorstore,
    save_faiss_index,
    verify_sqlite_saved,
    init_sqlite_db,
    insert_chunks_to_sqlite,
    query_all_topics,
    query_all_concepts,
    query_all_topics_with_counts
)
from models import ResolvedTopicResponse

# Initialize MarkItDown converter
markdown_converter = MarkItDown()


def clean_markdown(content: str) -> str:
    """
    Clean markdown content by removing unnecessary HTML tags, special characters, and symbols.
    Keeps only essential markdown formatting and text content.
    """
    if not content:
        return ""
    
    # Remove HTML tags (including style, script, etc.)
    content = re.sub(r'<[^>]+>', '', content)
    
    # Remove HTML entities but keep common ones that are readable
    # Replace common HTML entities with their text equivalents
    html_entities = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&apos;': "'",
        '&#39;': "'",
        '&hellip;': '...',
        '&mdash;': '—',
        '&ndash;': '–',
    }
    for entity, replacement in html_entities.items():
        content = content.replace(entity, replacement)
    
    # Remove remaining HTML entities (numeric and named)
    content = re.sub(r'&#?\w+;', '', content)
    
    # Remove excessive whitespace (more than 2 consecutive spaces)
    content = re.sub(r' {3,}', ' ', content)
    
    # Remove excessive newlines (more than 2 consecutive)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Remove special Unicode characters that are not essential (keep common punctuation)
    # Keep: letters, numbers, common punctuation, whitespace, and basic markdown symbols
    # Remove: special symbols, emojis, and other non-essential characters
    # This regex keeps: alphanumeric, spaces, newlines, and common punctuation/markdown
    content = re.sub(r'[^\w\s\n\.\,\;\:\!\?\-\(\)\[\]\{\}\'\"\/\\\*\_\#\=\+\>\<\|`~]', '', content)
    
    # Clean up markdown links - keep the text, remove the URL if it's too long
    # Convert [text](url) to just "text" if URL is very long
    def clean_link(match):
        text = match.group(1)
        url = match.group(2)
        if len(url) > 100:  # If URL is too long, just keep the text
            return text
        return match.group(0)  # Keep original if URL is reasonable
    
    content = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', clean_link, content)
    
    # Remove markdown images (keep alt text if available)
    content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', content)
    
    # Remove excessive markdown formatting symbols (more than 3 consecutive)
    content = re.sub(r'[*_#]{4,}', '', content)
    
    # Remove control characters except newline and tab
    content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', content)
    
    # Trim whitespace from start and end
    content = content.strip()
    
    return content


def find_libreoffice():
    """Find LibreOffice executable path."""
    # Try common paths for LibreOffice
    possible_paths = [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",  # macOS
        "/usr/bin/libreoffice",  # Linux
        "/usr/local/bin/libreoffice",  # Linux alternative
        "libreoffice",  # In PATH
        "soffice",  # Alternative name
    ]
    
    for path in possible_paths:
        if path in ["libreoffice", "soffice"]:
            # Check if it's in PATH
            if shutil.which(path):
                return path
        elif os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return None


def convert_old_formats_to_new(data_dir: str) -> int:
    """Convert old .ppt and .doc files to .pptx and .docx using LibreOffice."""
    data_path = Path(data_dir)
    
    # Find LibreOffice
    libreoffice_path = find_libreoffice()
    if not libreoffice_path:
        error_msg = "ERROR: LibreOffice not found. Please install it with: brew install --cask libreoffice"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    print(f"Using LibreOffice at: {libreoffice_path}")
    
    # Find old format files
    old_doc_files = list(data_path.rglob("*.doc"))
    old_ppt_files = list(data_path.rglob("*.ppt"))
    old_files = old_doc_files + old_ppt_files
    
    # Filter out temporary files (Microsoft Office lock files starting with ~$)
    old_files = [f for f in old_files if not f.name.startswith("~$")]
    
    if not old_files:
        print("No old format files (.doc/.ppt) found to convert")
        return 0
    
    print(f"\n=== Converting old formats to new formats ===")
    print(f"Found {len(old_files)} old format files:")
    print(f"  - {len([f for f in old_doc_files if not f.name.startswith('~$')])} .doc files")
    print(f"  - {len([f for f in old_ppt_files if not f.name.startswith('~$')])} .ppt files")
    
    converted_count = 0
    failed_files = []
    
    for file_path in old_files:
        file_ext = file_path.suffix.lower()
        
        # Determine target format
        if file_ext == '.doc':
            target_format = 'docx'
        elif file_ext == '.ppt':
            target_format = 'pptx'
        else:
            continue
        
        # Check if converted file already exists
        converted_path = file_path.with_suffix(f'.{target_format}')
        if converted_path.exists():
            print(f"SKIP: {file_path.name} -> {converted_path.name} (already exists)")
            converted_count += 1
            continue
        
        try:
            # Use absolute paths to handle spaces and special characters properly
            abs_file_path = file_path.resolve()
            abs_outdir = file_path.parent.resolve()
            
            # Use LibreOffice to convert
            # --headless: run without GUI
            # --convert-to: specify output format
            # --outdir: output directory (same as source)
            cmd = [
                libreoffice_path,
                '--headless',
                '--convert-to', target_format,
                '--outdir', str(abs_outdir),
                str(abs_file_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout per file
                check=False,  # Don't raise on non-zero exit
                encoding='utf-8',  # Ensure proper encoding for error messages
                errors='replace'  # Replace encoding errors instead of failing
            )
            
            # Check if conversion was successful
            if converted_path.exists():
                converted_count += 1
                print(f"✓ Converted: {file_path.name} -> {converted_path.name}")
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                print(f"ERROR: Failed to convert {file_path.name}: {error_msg[:100]}")
                failed_files.append((file_path.name, error_msg))
                
        except subprocess.TimeoutExpired:
            print(f"ERROR: Timeout converting {file_path.name}")
            failed_files.append((file_path.name, "Conversion timeout"))
        except Exception as e:
            print(f"ERROR: Failed to convert {file_path.name}: {str(e)}")
            failed_files.append((file_path.name, str(e)))
    
    # Print summary
    print(f"\n=== Conversion Summary ===")
    print(f"Successfully converted: {converted_count} files")
    if failed_files:
        print(f"Failed: {len(failed_files)} files")
        for fname, error in failed_files[:5]:
            print(f"  - {fname}: {error[:100]}")
    
    return converted_count


def convert_to_markdown(data_dir: str) -> int:
    """Recursively convert .doc/.docx/.ppt/.pptx files to markdown format."""
    data_path = Path(data_dir)
    
    # Find all office document files (now focusing on new formats after conversion)
    doc_files = list(data_path.rglob("*.doc"))  # Old format, should be converted first
    docx_files = list(data_path.rglob("*.docx"))
    ppt_files = list(data_path.rglob("*.ppt"))  # Old format, should be converted first
    pptx_files = list(data_path.rglob("*.pptx"))
    
    # Filter out temporary files (Microsoft Office lock files starting with ~$)
    doc_files = [f for f in doc_files if not f.name.startswith("~$")]
    docx_files = [f for f in docx_files if not f.name.startswith("~$")]
    ppt_files = [f for f in ppt_files if not f.name.startswith("~$")]
    pptx_files = [f for f in pptx_files if not f.name.startswith("~$")]
    
    # Prioritize new formats, but also try old formats if they exist
    all_files = docx_files + pptx_files + doc_files + ppt_files
    
    if not all_files:
        error_msg = f"ERROR: No .doc, .docx, .ppt or .pptx files found in directory: {data_dir}"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Warn if old formats still exist (should have been converted)
    if doc_files or ppt_files:
        print(f"WARNING: Found {len(doc_files)} .doc and {len(ppt_files)} .ppt files.")
        print("  These should have been converted to .docx/.pptx. Attempting conversion anyway...")
    
    print(f"Found {len(all_files)} office document files to convert to markdown")
    print(f"  - {len(doc_files)} Word files (.doc)")
    print(f"  - {len(docx_files)} Word files (.docx)")
    print(f"  - {len(ppt_files)} PowerPoint files (.ppt)")
    print(f"  - {len(pptx_files)} PowerPoint files (.pptx)")
    
    converted_count = 0
    failed_files = []
    skipped_files = []
    
    for file_path in all_files:
        file_ext = file_path.suffix.lower()
        
        # Skip old formats - they should have been converted already
        if file_ext in ['.ppt', '.doc']:
            print(f"SKIP: {file_path.name} - Old format (.{file_ext[1:]}). Please convert to .{file_ext[1:]}x first using LibreOffice.")
            skipped_files.append(file_path.name)
            continue
        
        try:
            # Use absolute path to handle spaces and special characters properly
            abs_file_path = file_path.resolve()
            # Convert to markdown
            result = markdown_converter.convert(str(abs_file_path))
            markdown_content = result.text_content
            
            if not markdown_content or not markdown_content.strip():
                print(f"WARNING: Empty markdown content for {file_path.name}, skipping...")
                skipped_files.append(file_path.name)
                continue
            
            # Clean the markdown content (remove unnecessary tags and special characters)
            markdown_content = clean_markdown(markdown_content)
            
            if not markdown_content or not markdown_content.strip():
                print(f"WARNING: Markdown content became empty after cleaning for {file_path.name}, skipping...")
                skipped_files.append(file_path.name)
                continue
            
            # Save markdown file next to the original file with .md extension
            md_path = file_path.with_suffix('.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            converted_count += 1
            print(f"✓ Converted: {file_path.name} -> {md_path.name}")
            
        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages
            if "No converter attempted" in error_msg or "not supported" in error_msg.lower():
                print(f"SKIP: {file_path.name} - Format not supported by MarkItDown")
                skipped_files.append(file_path.name)
            else:
                print(f"ERROR: Failed to convert {file_path.name}: {error_msg}")
                failed_files.append((file_path.name, error_msg))
            continue
    
    # Print summary
    print(f"\n=== Conversion Summary ===")
    print(f"Successfully converted: {converted_count} files")
    if skipped_files:
        print(f"Skipped (unsupported format): {len(skipped_files)} files")
        if len(skipped_files) <= 10:
            for fname in skipped_files:
                print(f"  - {fname}")
        else:
            print(f"  (showing first 10 of {len(skipped_files)} skipped files)")
            for fname in skipped_files[:10]:
                print(f"  - {fname}")
    if failed_files:
        print(f"Failed (errors): {len(failed_files)} files")
        for fname, error in failed_files[:5]:  # Show first 5 errors
            print(f"  - {fname}: {error[:100]}")
    
    if converted_count == 0:
        error_msg = f"ERROR: No files were successfully converted to markdown"
        if skipped_files:
            error_msg += f"\n  {len(skipped_files)} files were skipped due to unsupported formats"
        if failed_files:
            error_msg += f"\n  {len(failed_files)} files failed to convert"
        print(f"\n{error_msg}")
        raise ValueError(error_msg)
    
    return converted_count


def load_documents(data_dir: str):
    """Load markdown files from the directory using DirectoryLoader."""
    data_path = Path(data_dir)

    # Check if directory exists
    if not data_path.exists():
        error_msg = f"ERROR: Directory does not exist: {data_dir}"
        print(error_msg)
        raise FileNotFoundError(error_msg)

    if not data_path.is_dir():
        error_msg = f"ERROR: Path is not a directory: {data_dir}"
        print(error_msg)
        raise NotADirectoryError(error_msg)

    # Check for markdown files
    md_files = list(data_path.rglob("*.md"))
    
    if not md_files:
        error_msg = f"ERROR: No .md files found in directory: {data_dir}"
        print(error_msg)
        print("  Hint: Run convert_to_markdown() first to convert office documents to markdown")
        raise FileNotFoundError(error_msg)

    # Load markdown files using DirectoryLoader with TextLoader
    md_loader = DirectoryLoader(
        str(data_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    docs = md_loader.load()

    if not docs or len(docs) == 0:
        error_msg = f"ERROR: No documents were successfully loaded from directory: {data_dir}"
        print(error_msg)
        print(f"  - Found {len(md_files)} .md files, loaded {len(docs)} documents")
        raise ValueError(error_msg)

    print(f"Loaded {len(docs)} markdown documents")
    return docs


def extract_json_from_text(text: str) -> dict:
    """
    Extract JSON from text that may contain reasoning/thinking before the JSON.
    Handles thinking models that output reasoning before JSON.
    """
    import json
    import re
    
    # First, try to find JSON code blocks (```json ... ```)
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    json_block_matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in reversed(json_block_matches):
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON object by finding matching braces (handles nested JSON)
    # Find all complete JSON objects by tracking brace depth
    candidates = []
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                # Found a complete JSON object
                json_str = text[start_idx:i+1]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        candidates.append(parsed)
                except json.JSONDecodeError:
                    pass
                start_idx = -1
    
    # Return the last (most likely final answer) JSON object found
    if candidates:
        return candidates[-1]
    
    # Fallback: try to find any JSON-like structure with regex
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    # Try each match from the end (most likely to be the final answer)
    for match in reversed(matches):
        try:
            cleaned = match.strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    
    # If no JSON found, try parsing the whole text
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    
    return None


def tokenize(text: str) -> list:
    """Simple tokenization: split by whitespace and convert to lowercase."""
    if not text:
        return []
    return text.lower().split()


def bm25_search(query: str, documents: list, k: int = 50, k1: float = 1.5, b: float = 0.75) -> list:
    """
    BM25 search to find the most similar documents to a query.
    
    Args:
        query: The search query string
        documents: List of document strings to search in
        k: Number of top results to return
        k1: BM25 parameter (term frequency saturation)
        b: BM25 parameter (length normalization)
    
    Returns:
        List of top k document indices sorted by relevance
    """
    if not query or not documents:
        return []
    
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    
    # Tokenize all documents
    doc_tokens = [tokenize(doc) for doc in documents]
    
    # Calculate document frequencies (how many documents contain each term)
    doc_freq = Counter()
    for tokens in doc_tokens:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
    
    # Calculate average document length
    avg_doc_length = sum(len(tokens) for tokens in doc_tokens) / len(doc_tokens) if doc_tokens else 1
    
    # Calculate BM25 scores
    scores = []
    N = len(documents)  # Total number of documents
    
    for i, doc_token_list in enumerate(doc_tokens):
        score = 0.0
        doc_length = len(doc_token_list)
        doc_token_counts = Counter(doc_token_list)
        
        for term in query_tokens:
            if term not in doc_freq:
                continue
            
            # Term frequency in current document
            tf = doc_token_counts.get(term, 0)
            
            # Document frequency (how many documents contain this term)
            df = doc_freq[term]
            
            # Inverse document frequency
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            
            # BM25 score component for this term
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            score += idf * (numerator / denominator)
        
        scores.append((score, i))
    
    # Sort by score (descending) and return top k indices
    scores.sort(reverse=True, key=lambda x: x[0])
    return [idx for _, idx in scores[:k]]


def extract_topic_metadata(chunk_content: str) -> dict:
    """
    Extract topic and sub-concepts from a document chunk using LLM.
    Returns a dictionary with topic_name and sub_concepts.
    """
    try:
        # Clean the chunk content before sending to LLM to reduce token usage
        cleaned_content = clean_markdown(chunk_content)
        
        # Limit chunk content length to prevent token overflow (safety check)
        # Since chunk_size is 2000 characters, we allow up to 2000 chars for topic extraction
        # Estimate: ~4 characters per token, so 2000 chars ≈ 500 tokens, well under the 32768 token limit
        MAX_CHUNK_CHARS = 2000  # Match the chunk_size to avoid truncating valid chunks
        if len(cleaned_content) > MAX_CHUNK_CHARS:
            cleaned_content = cleaned_content[:MAX_CHUNK_CHARS] + "..."
            print(f"Warning: Truncated chunk content from {len(chunk_content)} to {MAX_CHUNK_CHARS} characters")
        
        # First, try with the parser chain
        chain = TOPIC_EXTRACTION_PROMPT | llm
        llm_response = chain.invoke({
            "chunk_content": cleaned_content,
            "format_instructions": topic_parser.get_format_instructions()
        })
        
        # Get the text content from the response
        response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        # Try to parse with the parser first
        try:
            result = topic_parser.parse(response_text)
        except Exception as parse_error:
            # If parsing fails, try to extract JSON from the text (for thinking models)
            # This handles cases where enable_thinking: False doesn't work or model ignores it
            print(f"Parser failed (this is expected with thinking models), extracting JSON from response...")
            extracted_json = extract_json_from_text(response_text)
            if extracted_json:
                print(f"✅ Successfully extracted JSON from response")
                result = extracted_json
            else:
                # If extraction also fails, try to parse with Pydantic directly
                try:
                    result = topic_parser.pydantic_object.parse_raw(response_text)
                except:
                    # Log the response for debugging
                    print(f"Failed to extract JSON. Response preview (last 500 chars):\n{response_text[-500:]}")
                    raise parse_error
        
        # Validate result
        if not result or not isinstance(result, dict):
            print(f"Warning: LLM returned invalid result type: {type(result)}")
            return {
                "topic_name": "Unknown",
                "sub_concepts": []
            }
        
        topic_name = result.get("topic_name", "Unknown")
        sub_concepts = result.get("sub_concepts", [])
        
        # Log if we got Unknown or empty concepts
        if topic_name == "Unknown" or not topic_name or topic_name.strip() == "":
            print(f"Warning: LLM returned 'Unknown' or empty topic_name. Result: {result}")
        
        if not sub_concepts or len(sub_concepts) == 0:
            print(f"Warning: LLM returned no concepts. Topic: {topic_name}, Result: {result}")
        
        return result
    except Exception as e:
        # Fallback if extraction fails - but log the full error
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: Failed to extract topic metadata: {e}")
        print(f"Error details:\n{error_details}")
        print(f"Chunk preview (first 200 chars): {chunk_content[:200]}...")
        return {
            "topic_name": "Unknown",
            "sub_concepts": []
        }


def resolve_topic_and_concepts(extracted_topic: str, extracted_concepts: list, 
                               existing_topics: list, existing_concepts: list) -> dict:
    """
    Resolve extracted topic and concepts by matching to existing ones in database.
    Returns resolved topic and concepts (using existing names if similar, new if different).
    """
    # If database is empty, just return extracted values
    # If extracted_concepts is empty, return empty list
    if not existing_topics and not existing_concepts:
        resolved_concepts = [] if (not extracted_concepts or (isinstance(extracted_concepts, list) and len(extracted_concepts) == 0)) else extracted_concepts
        return {
            "resolved_topic": extracted_topic,
            "resolved_concepts": resolved_concepts
        }
    
    try:
        # Use BM25 search to find similar topics/concepts instead of sending all
        # This prevents token overflow while keeping the most relevant items
        MAX_SIMILAR_ITEMS = 50  # Maximum number of similar items to include
        
        # Find similar topics using BM25 search
        if existing_topics and len(existing_topics) > 0:
            similar_topic_indices = bm25_search(extracted_topic, existing_topics, k=MAX_SIMILAR_ITEMS)
            limited_topics = [existing_topics[i] for i in similar_topic_indices]
            if len(existing_topics) > MAX_SIMILAR_ITEMS:
                print(f"Using BM25 to select {len(limited_topics)} most similar topics from {len(existing_topics)} total topics")
        else:
            limited_topics = []
        
        # Find similar concepts using BM25 search
        # Search for each extracted concept and combine results
        if existing_concepts and len(existing_concepts) > 0 and extracted_concepts:
            # Combine all extracted concepts into a single query for better matching
            concepts_query = " ".join(extracted_concepts) if extracted_concepts else ""
            similar_concept_indices = bm25_search(concepts_query, existing_concepts, k=MAX_SIMILAR_ITEMS)
            limited_concepts = [existing_concepts[i] for i in similar_concept_indices]
            if len(existing_concepts) > MAX_SIMILAR_ITEMS:
                print(f"Using BM25 to select {len(limited_concepts)} most similar concepts from {len(existing_concepts)} total concepts")
        else:
            limited_concepts = []
        
        # Format existing topics and concepts for the prompt
        existing_topics_str = "\n".join([f"- {topic}" for topic in limited_topics]) if limited_topics else "None"
        existing_concepts_str = "\n".join([f"- {concept}" for concept in limited_concepts]) if limited_concepts else "None"
        extracted_concepts_str = ", ".join(extracted_concepts) if extracted_concepts else "None"
        
        # Use ResolvedTopicResponse parser for resolution
        resolution_parser = JsonOutputParser(pydantic_object=ResolvedTopicResponse)
        
        # Get LLM response first, then parse (to handle thinking models)
        chain = NODE_RESOLUTION_PROMPT | llm
        llm_response = chain.invoke({
            "extracted_topic": extracted_topic,
            "extracted_concepts": extracted_concepts_str,
            "existing_topics": existing_topics_str,
            "existing_concepts": existing_concepts_str,
            "format_instructions": resolution_parser.get_format_instructions()
        })
        
        # Get the text content from the response
        response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        # Try to parse with the parser first
        try:
            result = resolution_parser.parse(response_text)
        except Exception as parse_error:
            # If parsing fails, try to extract JSON from the text (for thinking models)
            print(f"Resolution parser failed, attempting to extract JSON: {parse_error}")
            extracted_json = extract_json_from_text(response_text)
            if extracted_json:
                result = extracted_json
            else:
                # If extraction also fails, try to parse with Pydantic directly
                try:
                    result = resolution_parser.pydantic_object.parse_raw(response_text)
                except:
                    raise parse_error
        
        # Get resolved topic, with fallback to extracted if missing or "Unknown"
        resolved_topic = result.get("resolved_topic") or extracted_topic
        if not resolved_topic or resolved_topic.strip().lower() in ["unknown", "none", ""]:
            resolved_topic = extracted_topic
        
        # Handle resolved concepts
        # If extracted_concepts is empty/None, resolved_concepts should be empty list
        if not extracted_concepts or (isinstance(extracted_concepts, list) and len(extracted_concepts) == 0):
            resolved_concepts = []
        else:
            # Get resolved concepts from LLM result
            resolved_concepts = result.get("resolved_concepts")
            
            # If LLM didn't return concepts, fall back to extracted
            if resolved_concepts is None:
                resolved_concepts = extracted_concepts
            elif isinstance(resolved_concepts, list):
                if len(resolved_concepts) == 0:
                    # LLM returned empty list - fall back to extracted concepts
                    resolved_concepts = extracted_concepts
                else:
                    # Filter out "Unknown" values
                    filtered_concepts = [c for c in resolved_concepts if c and c.strip().lower() not in ["unknown", "none", ""]]
                    if not filtered_concepts:
                        # All were filtered out - fall back to extracted
                        resolved_concepts = extracted_concepts
                    else:
                        resolved_concepts = filtered_concepts
            else:
                # If not a list, fall back to extracted
                resolved_concepts = extracted_concepts
        
        return {
            "resolved_topic": resolved_topic,
            "resolved_concepts": resolved_concepts
        }
    except Exception as e:
        # Fallback: return extracted values if resolution fails
        print(f"Warning: Failed to resolve topic/concepts: {e}")
        # If extracted_concepts is empty, return empty list; otherwise return extracted
        resolved_concepts = [] if (not extracted_concepts or (isinstance(extracted_concepts, list) and len(extracted_concepts) == 0)) else extracted_concepts
        return {
            "resolved_topic": extracted_topic,
            "resolved_concepts": resolved_concepts
        }


def chunk_documents(docs):
    """Split documents into smaller chunks suitable for embedding.
    Uses character-based chunking to ensure consistent chunk sizes."""
    if not docs or len(docs) == 0:
        error_msg = "ERROR: Cannot chunk documents - no documents provided"
        print(error_msg)
        raise ValueError(error_msg)

    # Use character-based chunking
    # chunk_size is in characters, not tokens
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # 2000 characters
        chunk_overlap=500,  # 500 characters overlap
        length_function=len,  # Explicitly use character length
        separators=["\n\n", "\n", ". ", " ", ""],  # Character-based separators
    )
    split_docs = splitter.split_documents(docs)

    if len(split_docs) == 0:
        error_msg = "ERROR: Document chunking resulted in 0 chunks"
        print(error_msg)
        raise ValueError(error_msg)

    print(f"Split into {len(split_docs)} chunks")
    return split_docs


def add_topic_metadata_to_chunks(split_docs, doc_type: str = None, db_path: str = None):
    """Extract topic metadata for each chunk, resolve against existing ones, and add to metadata."""
    print(f"Extracting topic metadata for {len(split_docs)} chunks...")
    
    # Get existing topics and concepts from database
    if db_path:
        existing_topics = query_all_topics(db_path)
        existing_concepts = query_all_concepts(db_path)
    else:
        existing_topics = query_all_topics()
        existing_concepts = query_all_concepts()
    
    print(f"Found {len(existing_topics)} existing topics and {len(existing_concepts)} existing concepts in database")
    
    for i, chunk in enumerate(split_docs, 1):
        if i % 10 == 0:
            print(f"  Processing chunk {i}/{len(split_docs)}...")
        
        # Extract topic and concepts from chunk
        topic_metadata = extract_topic_metadata(chunk.page_content)
        extracted_topic = topic_metadata.get("topic_name", "Unknown")
        extracted_concepts = topic_metadata.get("sub_concepts", [])
        
        # Debug logging for first few chunks
        if i <= 3:
            print(f"  [DEBUG] Chunk {i}: extracted_topic='{extracted_topic}', extracted_concepts={extracted_concepts}")
        
        # Resolve against existing topics and concepts
        resolved = resolve_topic_and_concepts(
            extracted_topic, 
            extracted_concepts,
            existing_topics,
            existing_concepts
        )
        
        # Get resolved values
        resolved_topic = resolved.get("resolved_topic", extracted_topic)
        resolved_concepts = resolved.get("resolved_concepts", extracted_concepts)
        
        # Update existing lists with newly resolved items (for next chunks)
        # This ensures subsequent chunks can match against newly extracted topics/concepts
        if resolved_topic and resolved_topic.strip().lower() not in ["unknown", "none", ""]:
            if resolved_topic not in existing_topics:
                existing_topics.append(resolved_topic)
        
        # Add resolved concepts to existing list (filter out invalid values)
        if resolved_concepts and isinstance(resolved_concepts, list):
            for concept in resolved_concepts:
                if concept and concept.strip() and concept.strip().lower() not in ["unknown", "none", ""]:
                    if concept not in existing_concepts:
                        existing_concepts.append(concept)
        
        # Add resolved topic metadata and doc_type to chunk's existing metadata
        metadata_update = {
            "topic_name": resolved_topic,
            "sub_concepts": resolved_concepts
        }
        if doc_type:
            metadata_update["doc_type"] = doc_type
        chunk.metadata.update(metadata_update)
    
    print("Topic metadata extraction and resolution completed")
    return split_docs


def create_retriever(vectorstore):
    """Create a retriever from the FAISS vectorstore."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    return retriever


def create_qa_chain(retriever):
    """Create an optional RetrievalQA chain over the retriever."""
    if not RETRIEVAL_QA_AVAILABLE or RetrievalQA is None:
        raise ImportError("RetrievalQA is not available. Install langchain-classic for backward compatibility.")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    return qa


def run_retriever_test(vectorstore, test_query: str = "What are the main topics covered in these slides and docs?"):
    """
    Test the retriever with a sample query.
    
    Args:
        vectorstore: The FAISS vectorstore to test
        test_query: The query to test with
        
    Returns:
        The retriever instance for further use
    """
    print("\n=== Creating retriever ===")
    retriever = create_retriever(vectorstore)

    print("\n=== Retriever test ===")
    retrieved_docs = retriever.invoke(test_query)
    for i, d in enumerate(retrieved_docs, 1):
        print(f"\n--- Result {i} ---")
        print("Source:", d.metadata.get("source"))
        print("Topic:", d.metadata.get("topic_name", "N/A"))
        print("Sub-concepts:", d.metadata.get("sub_concepts", []))
        print(d.page_content[:500], "...\n")
    
    return retriever


def run_qa_chain_test(retriever, test_query: str = "What are the main topics covered in these slides and docs?"):
    """
    Test the RetrievalQA chain with a sample query.
    
    Args:
        retriever: The retriever to use for QA
        test_query: The query to test with
    """
    print("\n=== RetrievalQA test (optional) ===")
    try:
        qa = create_qa_chain(retriever)
        result = qa.invoke({"query": test_query})
        print("Answer:", result.get("result", result))
        if "source_documents" in result:
            print(f"\nSource documents: {len(result['source_documents'])} documents used")
    except Exception as e:
        print("RetrievalQA chain failed (LLM not configured?):", e)


def ingest_documents(DATA_DIR: str, doc_type: str = None):
    """
    Ingest documents from a directory.
    
    Args:
        DATA_DIR: Directory containing documents to ingest
        doc_type: Type of documents - either "teaching_material" or "question_bank"
    """
    if doc_type and doc_type not in ["teaching_material", "question_bank"]:
        raise ValueError(f"doc_type must be either 'teaching_material' or 'question_bank', got: {doc_type}")
    
    try:
        # 1. Convert old formats (.ppt/.doc) to new formats (.pptx/.docx) using LibreOffice
        print("=== Step 1: Converting old formats to new formats ===")
        convert_old_formats_to_new(DATA_DIR)
        
        # 2. Convert office documents to markdown
        print("\n=== Step 2: Converting office documents to markdown ===")
        convert_to_markdown(DATA_DIR)
        
        # 3. Load markdown documents
        print("\n=== Step 3: Loading markdown documents ===")
        docs = load_documents(DATA_DIR)
        
        # 4. Chunk documents
        print("\n=== Step 4: Chunking documents ===")
        split_docs = chunk_documents(docs)

        # 5. Extract topic metadata for each chunk and resolve against existing ones
        print("\n=== Step 5: Extracting topic metadata from chunks ===")
        # Initialize database first to check for existing topics/concepts
        init_sqlite_db()
        split_docs = add_topic_metadata_to_chunks(split_docs, doc_type=doc_type, db_path=settings.SQLITE_DB_PATH)

        # 6. Store chunks in SQLite database
        print("\n=== Step 6: Storing chunks in SQLite database ===")
        insert_chunks_to_sqlite(split_docs)
        
        # Display summary of topics
        print("\n=== Topics Summary ===")
        topics_with_counts = query_all_topics_with_counts()
        print(f"Found {len(topics_with_counts)} unique topics:")
        for topic_name, count in topics_with_counts[:20]:  # Show first 20 topics
            print(f"  - {topic_name}: {count} chunks")
        if len(topics_with_counts) > 20:
            print(f"  ... and {len(topics_with_counts) - 20} more topics")

        # 7. FAISS vectorstore
        print("\n=== Step 7: Building/loading FAISS vectorstore ===")
        vectorstore = get_faiss_vectorstore(split_docs)
        
        # 8. Ensure all data is saved
        print("\n=== Step 8: Saving all data ===")
        save_faiss_index(vectorstore)
        verify_sqlite_saved()
        print("\n✓ Ingestion completed successfully!")
        
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"\nFailed to load documents: {e}")
        return
def test_retrieval_with_random_topic():
    """
    Test the retriever and QA chain using a random topic from SQLite.
    """
    # Get all topics from SQLite
    topics = query_all_topics()
    if not topics:
        print("No topics found in SQLite database.")
        return
    
    # Select a random topic
    random_topic = random.choice(topics)
    print(f"\n=== Selected random topic: {random_topic} ===")
    
    # Build a query from the topic
    test_query = f"What are the key concepts and information about {random_topic}?"
    print(f"Test query: {test_query}")
    
    # Load the vectorstore
    vectorstore = get_faiss_vectorstore()
    if vectorstore is None:
        print("Failed to load FAISS vectorstore.")
        return
    
    # Run retriever test
    retriever = run_retriever_test(vectorstore, test_query)
    
    # Run QA chain test
    run_qa_chain_test(retriever, test_query)

if __name__ == "__main__":
    DATA_DIR = "/Users/kahingleung/Downloads/edu-doc/science/Teaching-Materials"
    QUIZ_DIR = "/Users/kahingleung/Downloads/edu-doc/science/Question-Bank"
    # Ingest with doc_type: "teaching_material" or "question_bank"
    # Example: ingest_documents(DATA_DIR, doc_type="teaching_material")
    ingest_documents(DATA_DIR, doc_type="teaching_material")
    ingest_documents(QUIZ_DIR, doc_type="question_bank")
    test_retrieval_with_random_topic()