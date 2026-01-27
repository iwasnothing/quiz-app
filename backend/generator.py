from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document
import numpy as np
import json
import re
import time
import random
from database import (
    vector_store, 
    get_concepts_from_question_bank, 
    query_question_bank_chunks_by_topics,
    query_teaching_material_chunks_by_topics,
    get_concepts_for_topic,
    get_embeddings,
    settings
)
from llm_prompts import llm, quiz_parser, GENERATE_SINGLE_PROMPT, REFINE_PROMPT
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
    Questions are evenly distributed across topics.
    
    Args:
        topics: List of topic names
        count: Total number of questions
        complexity_hard: Percentage of hard questions (0-100), remainder split between Easy/Medium
        format_mc: Percentage of multiple choice questions (0-100), remainder are Short Answer
    
    Returns:
        QuizSchema with questions distributed according to ratios and evenly across topics
    """
    # Evenly distribute questions across topics
    num_topics = len(topics)
    if num_topics == 0:
        raise ValueError("At least one topic must be provided")
    
    questions_per_topic = count // num_topics
    remainder = count % num_topics
    
    print(f"Generating quiz: {count} questions across {num_topics} topics")
    print(f"  Questions per topic: {questions_per_topic} (with {remainder} extra distributed)")
    
    # Calculate difficulty distribution per topic
    num_hard = int(questions_per_topic * complexity_hard / 100)
    num_easy = int((questions_per_topic - num_hard) * 0.5)  # Split remaining between Easy and Medium
    num_medium = questions_per_topic - num_hard - num_easy
    
    # Calculate format distribution
    num_mcq = int(questions_per_topic * format_mc / 100)
    num_short = questions_per_topic - num_mcq
    
    print(f"  Per-topic difficulty: {num_easy} Easy, {num_medium} Medium, {num_hard} Hard")
    print(f"  Per-topic format: {num_mcq} MCQ, {num_short} Short Answer")
    
    # Generate questions for each topic separately
    all_questions = []
    question_id_counter = 1
    
    for topic_idx, topic in enumerate(topics):
        # Add one extra question to the first 'remainder' topics
        topic_count = questions_per_topic + (1 if topic_idx < remainder else 0)
        if topic_count == 0:
            continue
        
        print(f"\nGenerating {topic_count} questions for topic: {topic}")
        
        # Calculate distributions for this topic (may have +1 if it gets remainder)
        topic_num_hard = int(topic_count * complexity_hard / 100)
        topic_num_easy = int((topic_count - topic_num_hard) * 0.5)
        topic_num_medium = topic_count - topic_num_hard - topic_num_easy
        topic_num_mcq = int(topic_count * format_mc / 100)
        topic_num_short = topic_count - topic_num_mcq
        
        # Generate Hard questions for this topic
        if topic_num_hard > 0:
            hard_mcq = int(topic_num_hard * format_mc / 100)
            hard_short = topic_num_hard - hard_mcq
            if hard_mcq > 0:
                hard_mcq_quiz = _generate_quiz_batch(topic, "Hard", hard_mcq, "MCQ", topics)
                for q in hard_mcq_quiz.questions:
                    q.id = f"q{question_id_counter}"
                    question_id_counter += 1
                    all_questions.append(q)
            if hard_short > 0:
                hard_short_quiz = _generate_quiz_batch(topic, "Hard", hard_short, "Short Answer", topics)
                for q in hard_short_quiz.questions:
                    q.id = f"q{question_id_counter}"
                    question_id_counter += 1
                    all_questions.append(q)
        
        # Generate Medium questions for this topic
        if topic_num_medium > 0:
            medium_mcq = int(topic_num_medium * format_mc / 100)
            medium_short = topic_num_medium - medium_mcq
            if medium_mcq > 0:
                medium_mcq_quiz = _generate_quiz_batch(topic, "Medium", medium_mcq, "MCQ", topics)
                for q in medium_mcq_quiz.questions:
                    q.id = f"q{question_id_counter}"
                    question_id_counter += 1
                    all_questions.append(q)
            if medium_short > 0:
                medium_short_quiz = _generate_quiz_batch(topic, "Medium", medium_short, "Short Answer", topics)
                for q in medium_short_quiz.questions:
                    q.id = f"q{question_id_counter}"
                    question_id_counter += 1
                    all_questions.append(q)
        
        # Generate Easy questions for this topic
        if topic_num_easy > 0:
            easy_mcq = int(topic_num_easy * format_mc / 100)
            easy_short = topic_num_easy - easy_mcq
            if easy_mcq > 0:
                easy_mcq_quiz = _generate_quiz_batch(topic, "Easy", easy_mcq, "MCQ", topics)
                for q in easy_mcq_quiz.questions:
                    q.id = f"q{question_id_counter}"
                    question_id_counter += 1
                    all_questions.append(q)
            if easy_short > 0:
                easy_short_quiz = _generate_quiz_batch(topic, "Easy", easy_short, "Short Answer", topics)
                for q in easy_short_quiz.questions:
                    q.id = f"q{question_id_counter}"
                    question_id_counter += 1
                    all_questions.append(q)
    
    topic_str = ", ".join(topics)
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
    Questions are evenly distributed across topics.
    
    Args:
        topics: List of topic names
        count: Total number of questions
        complexity_hard: Percentage of hard questions (0-100), remainder split between Easy/Medium
        format_mc: Percentage of multiple choice questions (0-100), remainder are Short Answer
    
    Yields:
        QuizQuestion objects as they are generated, one at a time
    """
    # Evenly distribute questions across topics
    num_topics = len(topics)
    if num_topics == 0:
        raise ValueError("At least one topic must be provided")
    
    questions_per_topic = count // num_topics
    remainder = count % num_topics
    
    print(f"Generating quiz: {count} questions across {num_topics} topics")
    print(f"  Questions per topic: {questions_per_topic} (with {remainder} extra distributed)")
    
    # Calculate difficulty distribution per topic
    num_hard = int(questions_per_topic * complexity_hard / 100)
    num_easy = int((questions_per_topic - num_hard) * 0.5)  # Split remaining between Easy and Medium
    num_medium = questions_per_topic - num_hard - num_easy
    
    # Calculate format distribution
    num_mcq = int(questions_per_topic * format_mc / 100)
    num_short = questions_per_topic - num_mcq
    
    print(f"  Per-topic difficulty: {num_easy} Easy, {num_medium} Medium, {num_hard} Hard")
    print(f"  Per-topic format: {num_mcq} MCQ, {num_short} Short Answer")
    
    question_counter = 1
    
    # Generate questions for each topic separately
    for topic_idx, topic in enumerate(topics):
        # Add one extra question to the first 'remainder' topics
        topic_count = questions_per_topic + (1 if topic_idx < remainder else 0)
        if topic_count == 0:
            continue
        
        print(f"\nGenerating {topic_count} questions for topic: {topic}")
        
        # Calculate distributions for this topic (may have +1 if it gets remainder)
        topic_num_hard = int(topic_count * complexity_hard / 100)
        topic_num_easy = int((topic_count - topic_num_hard) * 0.5)
        topic_num_medium = topic_count - topic_num_hard - topic_num_easy
        topic_num_mcq = int(topic_count * format_mc / 100)
        topic_num_short = topic_count - topic_num_mcq
        
        # Generate Hard questions for this topic
        if topic_num_hard > 0:
            hard_mcq = int(topic_num_hard * format_mc / 100)
            hard_short = topic_num_hard - hard_mcq
            if hard_mcq > 0:
                hard_mcq_quiz = _generate_quiz_batch(topic, "Hard", hard_mcq, "MCQ", topics)
                for q in hard_mcq_quiz.questions:
                    q.id = f"q{question_counter}"
                    question_counter += 1
                    yield q
            if hard_short > 0:
                hard_short_quiz = _generate_quiz_batch(topic, "Hard", hard_short, "Short Answer", topics)
                for q in hard_short_quiz.questions:
                    q.id = f"q{question_counter}"
                    question_counter += 1
                    yield q
        
        # Generate Medium questions for this topic
        if topic_num_medium > 0:
            medium_mcq = int(topic_num_medium * format_mc / 100)
            medium_short = topic_num_medium - medium_mcq
            if medium_mcq > 0:
                medium_mcq_quiz = _generate_quiz_batch(topic, "Medium", medium_mcq, "MCQ", topics)
                for q in medium_mcq_quiz.questions:
                    q.id = f"q{question_counter}"
                    question_counter += 1
                    yield q
            if medium_short > 0:
                medium_short_quiz = _generate_quiz_batch(topic, "Medium", medium_short, "Short Answer", topics)
                for q in medium_short_quiz.questions:
                    q.id = f"q{question_counter}"
                    question_counter += 1
                    yield q
        
        # Generate Easy questions for this topic
        if topic_num_easy > 0:
            easy_mcq = int(topic_num_easy * format_mc / 100)
            easy_short = topic_num_easy - easy_mcq
            if easy_mcq > 0:
                easy_mcq_quiz = _generate_quiz_batch(topic, "Easy", easy_mcq, "MCQ", topics)
                for q in easy_mcq_quiz.questions:
                    q.id = f"q{question_counter}"
                    question_counter += 1
                    yield q
            if easy_short > 0:
                easy_short_quiz = _generate_quiz_batch(topic, "Easy", easy_short, "Short Answer", topics)
                for q in easy_short_quiz.questions:
                    q.id = f"q{question_counter}"
                    question_counter += 1
                    yield q


def _generate_single_question(
    topic: str,
    concept: str,
    difficulty: str,
    format_type: str,
    all_topics: list[str] = None,
    other_concepts_in_quiz: list[str] = None,
):
    """
    Generate a single question for a specific topic and concept.
    
    Args:
        topic: Single topic name
        concept: Specific concept to generate question about
        difficulty: Difficulty level (Easy, Medium, Hard)
        format_type: Question format (MCQ or Short Answer)
        all_topics: All topics in the quiz (used to retrieve historical questions for diversity)
        other_concepts_in_quiz: Concepts already used for other questions in this quiz (avoids overlap)
    
    Returns:
        QuizQuestion object
    """
    topics = [topic]
    topics_for_history = all_topics if all_topics else topics
    other_concepts_in_quiz = other_concepts_in_quiz or []
    
    # Get teaching_material chunks that match the topic and concept
    teaching_material_chunks = query_teaching_material_chunks_by_topics(topics, limit=50)
    
    # Filter chunks that contain the concept
    concept_chunks = []
    for chunk in teaching_material_chunks:
        try:
            chunk_concepts_json = chunk[2]  # sub_concepts (JSON string)
            chunk_concepts = json.loads(chunk_concepts_json)
            if isinstance(chunk_concepts, list) and concept in chunk_concepts:
                concept_chunks.append(chunk)
        except (json.JSONDecodeError, TypeError):
            continue
    
    # If no chunks found for this concept, use all chunks for the topic
    if not concept_chunks:
        concept_chunks = teaching_material_chunks[:10]
        print(f"Warning: No chunks found specifically for concept '{concept}', using general topic chunks")
    
    # Convert to Document objects
    teaching_material_docs = [
        Document(
            page_content=chunk[3],  # chunk_text
            metadata={
                "topic_name": chunk[1],  # topic_name
                "sub_concepts": chunk[2],  # sub_concepts (JSON string)
                "source": chunk[4],  # source
                "doc_type": chunk[5]  # doc_type
            }
        )
        for chunk in concept_chunks[:10]
    ]
    
    if teaching_material_docs:
        context_text = "\n\n".join([d.page_content for d in teaching_material_docs])
    else:
        context_text = f"Generate a question about the concept '{concept}' in the topic '{topic}'."
    
    # Get historical questions for diversity
    historical_questions = query_question_bank_chunks_by_topics(topics_for_history, limit=15)
    
    if historical_questions:
        historical_text = "HISTORICAL QUESTIONS FROM QUESTION BANKS:\n" + "\n\n---\n\n".join(historical_questions[:10])
        diversity_instruction = (
            "CRITICAL DIVERSITY REQUIREMENT: The questions below are examples of previously generated questions. "
            "You MUST generate a NEW question that is DIFFERENT and DIVERSE from these historical examples. "
            "Avoid repeating the same question structure, wording, or phrasing patterns."
        )
    else:
        historical_text = ""
        diversity_instruction = "Generate a diverse question that covers the concept from a unique angle."

    # Reinforce concept-based diversification: this question tests ONLY this concept; others cover different concepts
    if other_concepts_in_quiz:
        diversity_instruction += (
            f"\n\nThis quiz already has questions on these concepts: {', '.join(other_concepts_in_quiz)}. "
            f"Your question MUST focus ONLY on \"{concept}\" and must NOT overlap with those. "
            "Ensure the question is clearly distinct and tests this specific concept."
        )
    else:
        diversity_instruction += (
            f"\n\nThis question must focus ONLY on the concept \"{concept}\". "
            "Other questions in this quiz cover different concepts—keep this one clearly concept-specific."
        )
    
    # Add format constraint
    format_constraint = ""
    if format_type == "MCQ":
        format_constraint = "IMPORTANT: The question must be Multiple Choice (MCQ) format with 3-4 options."
    elif format_type == "Short Answer":
        format_constraint = "IMPORTANT: The question must be Short Answer format (no multiple choice options)."
    
    # Prepare prompt inputs for single question
    prompt_inputs = {
        "context": context_text,
        "historical_questions": historical_text,
        "diversity_instruction": diversity_instruction,
        "topic": topic,
        "concept": concept,
        "difficulty": difficulty,
        "format_type": format_type,
        "format_instructions": parser.get_format_instructions() + "\n\n" + format_constraint,
    }
    
    # Generate single question
    raw_chain = GENERATE_SINGLE_PROMPT | llm
    chain = GENERATE_SINGLE_PROMPT | llm | parser
    
    max_retries = settings.LLM_MAX_RETRIES
    attempt = 0
    
    while attempt < max_retries:
        try:
            raw_output = raw_chain.invoke(prompt_inputs)
            raw_content = str(raw_output.content).strip()
            
            if not raw_content or len(raw_content.strip()) == 0:
                attempt += 1
                if attempt < max_retries:
                    print(f"⚠ LLM returned empty output. Retrying... (attempt {attempt}/{max_retries})")
                    time.sleep(3)
                    continue
                else:
                    raise ValueError(f"LLM returned empty output after {max_retries} attempts.")
            
            try:
                result = parser.parse(raw_content)
                # Extract single question from result
                def _to_question(q_dict: dict) -> QuizQuestion:
                    q = dict(q_dict)
                    q.setdefault("id", "q_temp")
                    q["topic"] = topic
                    q["concept"] = concept
                    return QuizQuestion(**q)

                if isinstance(result, dict):
                    questions_data = result.get("questions", [])
                    if questions_data:
                        return _to_question(questions_data[0])
                    else:
                        return _to_question(result)
                elif isinstance(result, QuizSchema):
                    if result.questions:
                        q = result.questions[0]
                        # Update topic and concept
                        q_dict = q.model_dump() if hasattr(q, 'model_dump') else q.dict()
                        q_dict["topic"] = topic
                        q_dict["concept"] = concept
                        return QuizQuestion(**q_dict)
                    else:
                        raise ValueError("No questions in result")
                break
            except Exception as parse_error:
                # Try to extract JSON manually
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        parsed_json = json.loads(json_str)
                        questions_data = parsed_json.get("questions", [])
                        q_dict = questions_data[0] if questions_data else parsed_json
                        q_dict = dict(q_dict)
                        q_dict.setdefault("id", "q_temp")
                        q_dict["topic"] = topic
                        q_dict["concept"] = concept
                        return QuizQuestion(**q_dict)
                    except Exception:
                        pass
                
                attempt += 1
                if attempt < max_retries:
                    print(f"⚠ Parser failed. Retrying... (attempt {attempt}/{max_retries})")
                    time.sleep(3)
                    continue
                else:
                    raise
        except Exception as e:
            attempt += 1
            if attempt < max_retries:
                print(f"⚠ Error occurred: {e}. Retrying... (attempt {attempt}/{max_retries})")
                time.sleep(3)
                continue
            else:
                raise
    
    raise ValueError(f"Failed to generate question after {max_retries} attempts")


def _generate_quiz_batch(topic: str, difficulty: str, count: int, format_type: str, all_topics: list[str] = None):
    """
    Internal helper to generate a batch of questions with specific difficulty and format.
    Each question is generated for a randomly selected concept from the topic.
    
    Args:
        topic: Single topic name for this batch
        difficulty: Difficulty level (Easy, Medium, Hard)
        count: Number of questions to generate
        format_type: Question format (MCQ or Short Answer)
        all_topics: All topics in the quiz (used to retrieve historical questions for diversity)
    """
    # Convert topic to list (in case multiple topics are passed as comma-separated string)
    topics = [t.strip() for t in topic.split(',')] if ',' in topic else [topic]
    
    # Get all concepts for this topic
    topic_concepts = get_concepts_for_topic(topic)
    if not topic_concepts:
        # Fallback: try to get concepts from question_bank
        topic_concepts = get_concepts_from_question_bank(topics)
        print(f"Warning: No concepts found in SQLite for topic '{topic}', using concepts from question_bank")
    
    if not topic_concepts:
        raise ValueError(f"No concepts found for topic '{topic}'. Please ensure the topic has been ingested with concepts.")
    
    print(f"Found {len(topic_concepts)} concepts for topic '{topic}'")
    
    # Assign concepts to diversify questions: use distinct concepts first, then allow repeats if needed
    n_distinct = min(count, len(topic_concepts))
    concept_assignments: list[str] = list(random.sample(topic_concepts, n_distinct))
    while len(concept_assignments) < count:
        concept_assignments.append(random.choice(topic_concepts))
    random.shuffle(concept_assignments)
    print(f"Concept assignment for {count} questions: {[c for c in concept_assignments]}")
    
    # Track concepts already used in this batch (for prompt diversity hint)
    concepts_used_so_far: list[str] = []
    
    # Generate questions one at a time, each with its assigned concept
    questions = []
    for i in range(count):
        selected_concept = concept_assignments[i]
        print(f"Generating question {i+1}/{count} for topic '{topic}' on concept '{selected_concept}'")
        
        try:
            question = _generate_single_question(
                topic, selected_concept, difficulty, format_type, all_topics,
                other_concepts_in_quiz=concepts_used_so_far,
            )
            questions.append(question)
            concepts_used_so_far.append(selected_concept)
        except Exception as e:
            print(f"⚠ Error generating question {i+1} for concept '{selected_concept}': {e}")
            # Retry with a different concept (prefer unused concepts)
            alternatives = [c for c in topic_concepts if c != selected_concept]
            if alternatives:
                alt = random.choice(alternatives)
                print(f"Retrying with concept '{alt}'")
                try:
                    question = _generate_single_question(
                        topic, alt, difficulty, format_type, all_topics,
                        other_concepts_in_quiz=concepts_used_so_far,
                    )
                    questions.append(question)
                    concepts_used_so_far.append(alt)
                except Exception as e2:
                    print(f"⚠ Failed to generate question after retry: {e2}")
                    continue
            else:
                print(f"⚠ Skipping question {i+1} due to error")
                continue
    
    if not questions:
        raise ValueError(f"Failed to generate any questions for topic '{topic}'")
    
    return QuizSchema(
        title=f"Quiz: {topic}",
        questions=questions
    )


def refine_question_chain(question_data: dict, feedback: str):
    # 1. Retrieve Context specific to this question
    # We use the question text itself to find relevant backing data
    if vector_store is None:
        context_text = "No context available. Vector store not initialized."
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(question_data['question_text'])
        context_text = "\n\n".join([d.page_content for d in docs])

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