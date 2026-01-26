from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document
import numpy as np
import json
import re
import time
from database import (
    vector_store, 
    get_concepts_from_question_bank, 
    query_question_bank_chunks_by_topics,
    query_teaching_material_chunks_by_topics,
    get_embeddings,
    settings
)
from llm_prompts import llm, quiz_parser, GENERATE_PROMPT, REFINE_PROMPT
from models import QuizSchema, QuizQuestion

# Use parser from llm_prompts
parser = quiz_parser

def generate_quiz_chain(
    topics: list[str],
    count: int,
    complexity_hard: int = 50,
    format_mc: int = 50
):
    """
    Generate quiz using sophisticated retrieval logic with complexity and format ratios.
    
    Args:
        topics: List of topic names
        count: Total number of questions
        complexity_hard: Percentage of hard questions (0-100), remainder split between Easy/Medium
        format_mc: Percentage of multiple choice questions (0-100), remainder are Short Answer
    
    Returns:
        QuizSchema with questions distributed according to ratios
    """
    # Calculate difficulty distribution
    num_hard = int(count * complexity_hard / 100)
    num_easy = int((count - num_hard) * 0.5)  # Split remaining between Easy and Medium
    num_medium = count - num_hard - num_easy
    
    # Calculate format distribution
    num_mcq = int(count * format_mc / 100)
    num_short = count - num_mcq
    
    print(f"Generating quiz: {count} questions")
    print(f"  Difficulty: {num_easy} Easy, {num_medium} Medium, {num_hard} Hard")
    print(f"  Format: {num_mcq} MCQ, {num_short} Short Answer")
    
    # Generate questions in batches by difficulty and format
    all_questions = []
    topic_str = ", ".join(topics)
    
    # Generate Hard questions
    if num_hard > 0:
        hard_mcq = int(num_hard * format_mc / 100)
        hard_short = num_hard - hard_mcq
        if hard_mcq > 0:
            hard_mcq_quiz = _generate_quiz_batch(topic_str, "Hard", hard_mcq, "MCQ")
            all_questions.extend(hard_mcq_quiz.questions)
        if hard_short > 0:
            hard_short_quiz = _generate_quiz_batch(topic_str, "Hard", hard_short, "Short Answer")
            all_questions.extend(hard_short_quiz.questions)
    
    # Generate Medium questions
    if num_medium > 0:
        medium_mcq = int(num_medium * format_mc / 100)
        medium_short = num_medium - medium_mcq
        if medium_mcq > 0:
            medium_mcq_quiz = _generate_quiz_batch(topic_str, "Medium", medium_mcq, "MCQ")
            all_questions.extend(medium_mcq_quiz.questions)
        if medium_short > 0:
            medium_short_quiz = _generate_quiz_batch(topic_str, "Medium", medium_short, "Short Answer")
            all_questions.extend(medium_short_quiz.questions)
    
    # Generate Easy questions
    if num_easy > 0:
        easy_mcq = int(num_easy * format_mc / 100)
        easy_short = num_easy - easy_mcq
        if easy_mcq > 0:
            easy_mcq_quiz = _generate_quiz_batch(topic_str, "Easy", easy_mcq, "MCQ")
            all_questions.extend(easy_mcq_quiz.questions)
        if easy_short > 0:
            easy_short_quiz = _generate_quiz_batch(topic_str, "Easy", easy_short, "Short Answer")
            all_questions.extend(easy_short_quiz.questions)
    
    # Assign IDs
    for i, q in enumerate(all_questions[:count], 1):
        q.id = f"q{i}"
    
    return QuizSchema(
        title=f"Quiz: {topic_str}",
        questions=all_questions[:count]
    )


def generate_quiz_chain_streaming(
    topics: list[str],
    count: int,
    complexity_hard: int = 50,
    format_mc: int = 50
):
    """
    Generate quiz with streaming - yields questions as they are generated.
    
    Args:
        topics: List of topic names
        count: Total number of questions
        complexity_hard: Percentage of hard questions (0-100), remainder split between Easy/Medium
        format_mc: Percentage of multiple choice questions (0-100), remainder are Short Answer
    
    Yields:
        QuizQuestion objects as they are generated, one at a time
    """
    # Calculate difficulty distribution
    num_hard = int(count * complexity_hard / 100)
    num_easy = int((count - num_hard) * 0.5)  # Split remaining between Easy and Medium
    num_medium = count - num_hard - num_easy
    
    # Calculate format distribution
    num_mcq = int(count * format_mc / 100)
    num_short = count - num_mcq
    
    print(f"Generating quiz: {count} questions")
    print(f"  Difficulty: {num_easy} Easy, {num_medium} Medium, {num_hard} Hard")
    print(f"  Format: {num_mcq} MCQ, {num_short} Short Answer")
    
    topic_str = ", ".join(topics)
    question_counter = 1
    
    # Generate Hard questions
    if num_hard > 0:
        hard_mcq = int(num_hard * format_mc / 100)
        hard_short = num_hard - hard_mcq
        if hard_mcq > 0:
            hard_mcq_quiz = _generate_quiz_batch(topic_str, "Hard", hard_mcq, "MCQ")
            for q in hard_mcq_quiz.questions:
                q.id = f"q{question_counter}"
                question_counter += 1
                yield q
        if hard_short > 0:
            hard_short_quiz = _generate_quiz_batch(topic_str, "Hard", hard_short, "Short Answer")
            for q in hard_short_quiz.questions:
                q.id = f"q{question_counter}"
                question_counter += 1
                yield q
    
    # Generate Medium questions
    if num_medium > 0:
        medium_mcq = int(num_medium * format_mc / 100)
        medium_short = num_medium - medium_mcq
        if medium_mcq > 0:
            medium_mcq_quiz = _generate_quiz_batch(topic_str, "Medium", medium_mcq, "MCQ")
            for q in medium_mcq_quiz.questions:
                q.id = f"q{question_counter}"
                question_counter += 1
                yield q
        if medium_short > 0:
            medium_short_quiz = _generate_quiz_batch(topic_str, "Medium", medium_short, "Short Answer")
            for q in medium_short_quiz.questions:
                q.id = f"q{question_counter}"
                question_counter += 1
                yield q
    
    # Generate Easy questions
    if num_easy > 0:
        easy_mcq = int(num_easy * format_mc / 100)
        easy_short = num_easy - easy_mcq
        if easy_mcq > 0:
            easy_mcq_quiz = _generate_quiz_batch(topic_str, "Easy", easy_mcq, "MCQ")
            for q in easy_mcq_quiz.questions:
                q.id = f"q{question_counter}"
                question_counter += 1
                yield q
        if easy_short > 0:
            easy_short_quiz = _generate_quiz_batch(topic_str, "Easy", easy_short, "Short Answer")
            for q in easy_short_quiz.questions:
                q.id = f"q{question_counter}"
                question_counter += 1
                yield q


def _generate_quiz_batch(topic: str, difficulty: str, count: int, format_type: str):
    """
    Internal helper to generate a batch of questions with specific difficulty and format.
    """
    # Convert topic to list (in case multiple topics are passed as comma-separated string)
    topics = [t.strip() for t in topic.split(',')] if ',' in topic else [topic]
    
    # Step 1: Get concepts from question_bank that are commonly tested for these topics
    concepts = get_concepts_from_question_bank(topics)
    print(f"Found {len(concepts)} concepts from question_bank for topics: {topics}")
    
    # Step 2: Use concepts to search teaching_material chunks via vector similarity
    # Strategy: Get candidate chunks from SQLite, then use vector similarity to rank them
    
    # Get teaching_material chunks from SQLite that match the topics
    teaching_material_chunks = query_teaching_material_chunks_by_topics(topics, limit=50)
    print(f"Found {len(teaching_material_chunks)} teaching_material chunks from SQLite for topics: {topics}")
    
    if teaching_material_chunks:
        # Convert SQLite chunks to Document objects for vector similarity search
        candidate_docs = [
            Document(
                page_content=chunk[3],  # chunk_text
                metadata={
                    "topic_name": chunk[1],  # topic_name
                    "sub_concepts": chunk[2],  # sub_concepts (JSON string)
                    "source": chunk[4],  # source
                    "doc_type": chunk[5]  # doc_type
                }
            )
            for chunk in teaching_material_chunks
        ]
        
        # Use concepts as query for vector similarity search
        concept_query = " ".join(concepts[:10]) if concepts else topic
        
        # Try to use embeddings for similarity search, fallback to simple text matching if connection fails
        try:
            embeddings_obj = get_embeddings()
            if embeddings_obj is None:
                raise ValueError("Embeddings not available")
            
            # Embed the query
            query_embedding = embeddings_obj.embed_query(concept_query)
            
            # Calculate similarity scores for each candidate document
            # Embed all candidate documents
            candidate_texts = [doc.page_content for doc in candidate_docs]
            candidate_embeddings = embeddings_obj.embed_documents(candidate_texts)
            
            # Calculate cosine similarity (simple dot product since embeddings are normalized)
            similarities = []
            query_embedding_np = np.array(query_embedding)
            for candidate_emb in candidate_embeddings:
                candidate_emb_np = np.array(candidate_emb)
                # Cosine similarity
                similarity = np.dot(query_embedding_np, candidate_emb_np) / (
                    np.linalg.norm(query_embedding_np) * np.linalg.norm(candidate_emb_np)
                )
                similarities.append(similarity)
            
            # Sort by similarity and get top k
            doc_similarities = list(zip(candidate_docs, similarities))
            doc_similarities.sort(key=lambda x: x[1], reverse=True)
            teaching_material_docs = [doc for doc, score in doc_similarities[:10]]
            
            print(f"Selected top {len(teaching_material_docs)} teaching_material chunks by similarity to concepts")
        except Exception as e:
            print(f"⚠ Warning: Embedding similarity search failed: {e}")
            print(f"  Falling back to using all candidate chunks (no similarity ranking)")
            # Fallback: use all candidate docs without similarity ranking
            teaching_material_docs = candidate_docs[:10]
    else:
        # Fallback: Use vector store directly if SQLite doesn't have results
        print("No teaching_material found in SQLite, trying vector store directly...")
        concept_query = " ".join(concepts[:10]) if concepts else topic
        
        if vector_store is None:
            print("Warning: Vector store is not available. Generating questions without context.")
            teaching_material_docs = []
        else:
            try:
                # Try with filter first (allow teaching_material or NULL doc_type)
                retriever = vector_store.as_retriever(
                    search_kwargs={
                        "k": 10
                        # Note: Filtering by doc_type is done manually below to allow NULL values
                    }
                )
                teaching_material_docs = retriever.invoke(concept_query)
                # Filter results manually: allow "teaching_material" or NULL/None doc_type
                teaching_material_docs = [
                    doc for doc in teaching_material_docs 
                    if doc.metadata.get("doc_type") == "teaching_material" or doc.metadata.get("doc_type") is None
                ]
            except Exception as e:
                error_msg = str(e)
                # Check if it's a connection error
                if "Connection" in error_msg or "connection" in error_msg.lower() or "refused" in error_msg.lower():
                    print(f"⚠ Connection error to embedding service: {error_msg}")
                    print(f"  Check that embedding service is running at {settings.EMBEDDING_BASE_URL}")
                else:
                    print(f"⚠ Filtered search failed: {error_msg}, trying without filter...")
                try:
                    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
                    teaching_material_docs = retriever.invoke(concept_query)
                    # Filter results manually: allow "teaching_material" or NULL/None doc_type
                    teaching_material_docs = [
                        doc for doc in teaching_material_docs 
                        if doc.metadata.get("doc_type") == "teaching_material" or doc.metadata.get("doc_type") is None
                    ]
                except Exception as e2:
                    error_msg2 = str(e2)
                    if "Connection" in error_msg2 or "connection" in error_msg2.lower() or "refused" in error_msg2.lower():
                        print(f"⚠ Vector store retrieval failed due to connection error: {error_msg2}")
                        print(f"  Check that embedding service is running at {settings.EMBEDDING_BASE_URL}")
                    else:
                        print(f"⚠ Vector store retrieval failed: {error_msg2}")
                    print("  Generating without context.")
                    teaching_material_docs = []
    
    if teaching_material_docs:
        context_text = "\n\n".join([d.page_content for d in teaching_material_docs])
        print(f"Retrieved {len(teaching_material_docs)} teaching_material chunks as context")
    else:
        context_text = f"No teaching material context available for topics: {topics}. Please ensure documents have been ingested and contain content related to these topics."
        print(f"Warning: No teaching material context found. Generating questions based on topic knowledge only.")
    
    # Step 3: Get question_bank chunks as few-shot examples
    few_shot_examples = query_question_bank_chunks_by_topics(topics, limit=5)
    print(f"Retrieved {len(few_shot_examples)} question_bank chunks as few-shot examples")
    
    # Format few-shot examples section
    if few_shot_examples:
        few_shot_text = "FEW-SHOT EXAMPLES FROM HISTORICAL QUESTION BANKS:\n" + "\n\n---\n\n".join(few_shot_examples)
        few_shot_instruction = "Study the few-shot examples from historical question banks to understand:\n   - What types of concepts are typically tested\n   - The style and format of questions used in past assessments\n   - The level of detail expected in answers"
        and_examples = " and the patterns shown in the examples"
        distractor_note = " similar to those in the examples"
    else:
        few_shot_text = ""
        if teaching_material_docs:
            few_shot_instruction = "Generate questions based on the teaching materials context."
        else:
            few_shot_instruction = "Generate questions based on general knowledge of the topics."
        and_examples = ""
        distractor_note = ""

    # Step 4: Generate with context and few-shot examples
    # First, get raw LLM output for debugging
    raw_chain = GENERATE_PROMPT | llm
    chain = GENERATE_PROMPT | llm | parser
    
    # Add format constraint to prompt
    format_constraint = ""
    if format_type == "MCQ":
        format_constraint = "IMPORTANT: All questions must be Multiple Choice (MCQ) format with 3-4 options."
    elif format_type == "Short Answer":
        format_constraint = "IMPORTANT: All questions must be Short Answer format (no multiple choice options)."
    
    # Prepare prompt inputs
    prompt_inputs = {
        "context": context_text,
        "few_shot_examples": few_shot_text,
        "few_shot_instruction": few_shot_instruction,
        "and_examples": and_examples,
        "distractor_note": distractor_note,
        "topic": topic,
        "difficulty": difficulty,
        "num_questions": count,
        "format_instructions": parser.get_format_instructions() + "\n\n" + format_constraint
    }
    
    result = None
    max_retries = settings.LLM_MAX_RETRIES
    attempt = 0
    last_error = None
    last_raw_content = None
    
    print(f"DEBUG: Using max retries: {max_retries}")
    while attempt < max_retries:
        try:
            # Get raw LLM output first for debugging
            print(f"\n{'='*80}")
            if attempt > 0:
                print(f"DEBUG: RETRY ATTEMPT {attempt}/{max_retries-1}")
            print(f"DEBUG: Invoking LLM for topic: {topic}, difficulty: {difficulty}, count: {count}")
            print(f"{'='*80}")
            
            raw_output = raw_chain.invoke(prompt_inputs)
            raw_content = str(raw_output.content).strip()
            last_raw_content = raw_content  # Store for error reporting
            
            print(f"\nDEBUG: Raw LLM output length: {len(raw_content)} characters")
            print(f"DEBUG: Raw LLM output (first 1000 chars):\n{raw_content[:1000]}")
            if len(raw_content) > 1000:
                print(f"DEBUG: ... (truncated, showing last 500 chars) ...\n{raw_content[-500:]}")
            print(f"{'='*80}\n")
            
            # Check if output is empty or just whitespace
            if not raw_content or len(raw_content.strip()) == 0:
                attempt += 1
                last_error = "LLM returned empty output"
                if attempt < max_retries:
                    print(f"⚠ LLM returned empty output. Retrying... (attempt {attempt}/{max_retries})")
                    time.sleep(3)  # Wait 3 seconds before retry
                    continue
                else:
                    raise ValueError(f"LLM returned empty output after {max_retries} attempts. This may indicate an issue with the LLM service.")
            
            # Now parse the raw output
            try:
                result = parser.parse(raw_content)
                print(f"✓ Successfully parsed JSON from LLM output")
                break  # Success, exit retry loop
            except Exception as parse_error:
                print(f"⚠ Parser failed on raw output: {parse_error}")
                # Try to extract JSON manually from the raw_content we already have
                error_msg = str(parse_error)
                
                # Handle JSON parsing errors specifically
                if isinstance(parse_error, OutputParserException) or "JSONDecodeError" in error_msg or "Invalid json output" in error_msg:
                    print(f"⚠ JSON parsing error: {error_msg}")
                    print(f"⚠ Attempting to extract JSON from raw output...")
                    
                    # Try to find JSON in markdown code blocks
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        print(f"⚠ Found JSON in markdown block, attempting to parse...")
                        try:
                            parsed_json = json.loads(json_str)
                            result = parsed_json
                            print(f"✓ Successfully parsed JSON from markdown block")
                            break  # Success, exit retry loop
                        except json.JSONDecodeError as je:
                            print(f"⚠ Failed to parse extracted JSON: {je}")
                            # Continue to retry
                    else:
                        # Try to find JSON object directly
                        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            print(f"⚠ Found JSON object in output, attempting to parse...")
                            try:
                                parsed_json = json.loads(json_str)
                                result = parsed_json
                                print(f"✓ Successfully parsed JSON from output")
                                break  # Success, exit retry loop
                            except json.JSONDecodeError as je:
                                print(f"⚠ Failed to parse extracted JSON: {je}")
                                # Continue to retry
                        else:
                            # No JSON found, retry if we have attempts left
                            attempt += 1
                            last_error = f"No JSON found in output: {error_msg}"
                            if attempt < max_retries:
                                print(f"⚠ No JSON found in output. Full raw output:\n{raw_content}")
                                print(f"⚠ Retrying... (attempt {attempt}/{max_retries})")
                                time.sleep(3)  # Wait 3 seconds before retry
                                continue
                            else:
                                # Show full output in error message
                                raise ValueError(
                                    f"LLM returned empty or non-JSON output after {max_retries} attempts.\n"
                                    f"Full raw output ({len(raw_content)} chars):\n{raw_content}"
                                ) from parse_error
                else:
                    # Not a JSON parsing error, retry if we have attempts left
                    attempt += 1
                    last_error = f"Unexpected error: {error_msg}"
                    if attempt < max_retries:
                        print(f"⚠ Unexpected error: {error_msg}. Retrying... (attempt {attempt}/{max_retries})")
                        time.sleep(3)  # Wait 3 seconds before retry
                        continue
                    else:
                        raise
        except Exception as e:
            error_msg = str(e)
            last_error = error_msg
            
            # Handle connection errors - don't retry these
            if "Connection" in error_msg or "connection" in error_msg.lower() or "refused" in error_msg.lower():
                print(f"⚠ Connection error to LLM service: {error_msg}")
                print(f"  Check that LLM service is running at {settings.LLM_BASE_URL}")
                raise
            
            # For other errors, retry if we have attempts left
            attempt += 1
            if attempt < max_retries:
                print(f"⚠ Error occurred: {error_msg}. Retrying... (attempt {attempt}/{max_retries})")
                time.sleep(3)  # Wait 3 seconds before retry
                continue
            else:
                # Re-raise other exceptions after all retries exhausted
                raise
    
    # Ensure we have a result - this should only happen if loop exits without break or raise
    if result is None:
        error_details = f"Topic: {topic}, Difficulty: {difficulty}, Count: {count}"
        if last_error:
            error_details += f"\nLast error: {last_error}"
        if last_raw_content:
            error_details += f"\nLast raw output ({len(last_raw_content)} chars):\n{last_raw_content[:1000]}"
        raise ValueError(f"Failed to generate quiz after {max_retries} attempts. {error_details}")
    
    # Convert dict result to QuizSchema object
    # The parser returns a dict, but we need a QuizSchema object
    if isinstance(result, dict):
        # Ensure questions is a list of QuizQuestion objects
        questions = []
        questions_data = result.get("questions", [])
        
        if not questions_data:
            raise ValueError(f"No questions returned from LLM for topic: {topic}, difficulty: {difficulty}, count: {count}")
        
        for q_dict in questions_data:
            try:
                # Convert each question dict to QuizQuestion object
                questions.append(QuizQuestion(**q_dict))
            except Exception as e:
                print(f"Error converting question dict to QuizQuestion: {e}")
                print(f"Question dict: {q_dict}")
                raise
        
        return QuizSchema(
            title=result.get("title", f"Quiz: {topic}"),
            questions=questions
        )
    elif isinstance(result, QuizSchema):
        # If it's already a QuizSchema, return as-is
        return result
    else:
        raise TypeError(f"Unexpected result type from chain: {type(result)}. Expected dict or QuizSchema.")

def refine_question_chain(question_data: dict, feedback: str):
    # 1. Retrieve Context specific to this question
    # We use the question text itself to find relevant backing data
    if vector_store is None:
        context_text = "No context available. Vector store not initialized."
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(question_data['question_text'])
        context_text = "\n\n".join([d.page_content for d in docs])

    # 2. Parse just one question
    question_parser = JsonOutputParser(pydantic_object=QuizSchema) # Using same parser mostly fine, or define specific

    chain = REFINE_PROMPT | llm | JsonOutputParser()
    
    result = chain.invoke({
        "original_question": question_data,
        "context": context_text,
        "feedback": feedback
    })
    return result


def reroll_question_chain(question: QuizQuestion, topics: list[str] = None):
    """
    Regenerate a single question using the same topics and maintaining its format/difficulty.
    """
    # Use the question's existing topics or provided topics
    if not topics:
        # Try to extract topics from question metadata or use a default
        topics = ["General"]
    
    topic_str = ", ".join(topics)
    
    # Get context for this specific question
    if vector_store is None:
        context_text = "No context available. Vector store not initialized."
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(question.question_text)
        context_text = "\n\n".join([d.page_content for d in docs])
    
    # Get few-shot examples
    few_shot_examples = query_question_bank_chunks_by_topics(topics, limit=3)
    few_shot_text = "FEW-SHOT EXAMPLES:\n" + "\n\n---\n\n".join(few_shot_examples) if few_shot_examples else ""
    
    # Create a prompt for single question regeneration
    from langchain_core.prompts import ChatPromptTemplate
    
    reroll_parser = JsonOutputParser(pydantic_object=QuizQuestion)
    
    REROLL_PROMPT = ChatPromptTemplate.from_template("""
You are regenerating a single quiz question. Maintain the same format and difficulty level.

ORIGINAL QUESTION (for reference):
{original_question}

CONTEXT FROM TEACHING MATERIALS:
{context}

{few_shot_examples}

INSTRUCTIONS:
1. Generate a NEW question (different from the original) on the same topic
2. Maintain the same difficulty level: {difficulty}
3. Maintain the same format: {format_type}
4. For MCQ: provide 3-4 options. For Short Answer: no options.
5. Provide a clear marking rubric.

Return ONLY the single question JSON object with:
- id: "{question_id}"
- type: "{format_type}"
- difficulty: "{difficulty}"
- question_text: the new question
- options: array of options (only if MCQ, otherwise null)
- correct_answer: the correct answer
- marking_rubric: how to grade this
- source_context: brief context used

{format_instructions}
""")
    
    format_type = question.type
    if format_type == "MCQ":
        format_type = "MCQ"
    else:
        format_type = "Short Answer"
    
    chain = REROLL_PROMPT | llm | reroll_parser
    
    result = chain.invoke({
        "original_question": question.dict(),
        "context": context_text,
        "few_shot_examples": few_shot_text,
        "difficulty": question.difficulty,
        "format_type": format_type,
        "question_id": question.id,
        "format_instructions": reroll_parser.get_format_instructions()
    })
    
    return result