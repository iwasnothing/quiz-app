from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from models import TopicResponse, ResolvedTopicResponse, QuizSchema
from config import settings

# Initialize LLM
llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
    temperature=settings.LLM_TEMPERATURE,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
)

# --- Topic Extraction Prompt ---
topic_parser = JsonOutputParser(pydantic_object=TopicResponse)
TOPIC_EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
/no_think

Analyze the following document chunk and extract the main topic and its sub-concepts.

DOCUMENT CHUNK:
{chunk_content}

INSTRUCTIONS:
1. Identify the main topic of this chunk
2. List 3-7 key sub-concepts or subtopics covered in this chunk
3. Be specific and concise

IMPORTANT: Output ONLY valid JSON. Do not include any reasoning, explanation, or thinking process. Output ONLY the JSON object.

Return the result in JSON format with:
- topic_name: the main topic (string)
- sub_concepts: list of sub-concepts (array of strings)

{format_instructions}
""")

# --- Node Resolution Prompt ---
NODE_RESOLUTION_PROMPT = ChatPromptTemplate.from_template("""
/no_think

You need to resolve extracted topics and concepts by matching them to existing ones in the database.

EXTRACTED TOPIC: {extracted_topic}
EXTRACTED CONCEPTS: {extracted_concepts}

EXISTING TOPICS IN DATABASE:
{existing_topics}

EXISTING CONCEPTS IN DATABASE:
{existing_concepts}

INSTRUCTIONS:
1. For the extracted topic, check if it's similar to any existing topic. If similar, return the EXACT existing topic name. If it's a brand new topic, return the extracted topic as-is.
2. For each extracted concept, check if it's similar to any existing concept. If similar, return the EXACT existing concept name. If it's brand new, return the extracted concept as-is.
3. Only create new entries if they are truly different from existing ones.

IMPORTANT: Output ONLY valid JSON. Do not include any reasoning, explanation, or thinking process. Output ONLY the JSON object.

Return the result in JSON format with:
- resolved_topic: the resolved topic name (use existing if similar, new if different)
- resolved_concepts: list of resolved concept names (use existing if similar, new if different)

{format_instructions}
""")

# --- Quiz Generation Prompts ---
quiz_parser = JsonOutputParser(pydantic_object=QuizSchema)
GENERATE_PROMPT = ChatPromptTemplate.from_template("""
You are an expert teacher. Create a quiz based on the requested topic and context.

CONTEXT FROM TEACHING MATERIALS:
{context}

{historical_questions}

{diversity_instruction}

USER REQUEST:
Topic: {topic}
Difficulty: {difficulty}
Number of Questions: {num_questions}

INSTRUCTIONS:
1. Use the teaching materials as the primary context for generating questions.
2. {few_shot_instruction}
3. Generate questions that align with the context{and_examples}.
4. Provide a clear marking rubric for each question.
5. For MCQs, provide plausible distractors{distractor_note}.
6. CRITICAL: Ensure all generated questions are DISTINCT and DIVERSE from the historical questions shown above. Avoid similar wording, structure, or concepts.

CRITICAL: You MUST respond with ONLY valid JSON. Do not include any explanatory text, markdown formatting, or code blocks. Output ONLY the raw JSON object.

FORMAT:
{format_instructions}
""")

REFINE_PROMPT = ChatPromptTemplate.from_template("""
You are editing a specific quiz question based on teacher feedback.

ORIGINAL QUESTION DATA:
{original_question}

CONTEXT:
{context}

TEACHER FEEDBACK:
{feedback}

INSTRUCTIONS:
Update the question fields to address the feedback. Keep the ID the same.
Return ONLY the single JSON object for this question.
""")
