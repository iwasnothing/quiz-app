from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# --- API Requests ---
class IngestResponse(BaseModel):
    filename: str
    chunks: int
    graph_nodes: int

class TopicResponse(BaseModel):
    topic_name: str
    sub_concepts: List[str]

class ResolvedTopicResponse(BaseModel):
    resolved_topic: str
    resolved_concepts: List[str]

# --- Quiz Structure (The "Canvas" Schema) ---
class QuizQuestion(BaseModel):
    id: str = Field(description="Unique identifier for the question (e.g., q1, q2)")
    type: Literal["MCQ", "Short Answer"]
    difficulty: Literal["Easy", "Medium", "Hard"]
    question_text: str
    options: Optional[List[str]] = Field(description="List of options for MCQ, empty for others")
    correct_answer: str
    marking_rubric: str = Field(description="Guide for the teacher on how to grade this")
    source_context: str = Field(description="The specific text snippet used to generate this")

class QuizSchema(BaseModel):
    title: str
    questions: List[QuizQuestion]

# --- Refinement Request ---
class RefineRequest(BaseModel):
    current_question: QuizQuestion
    feedback: str = Field(description="Teacher's instruction, e.g., 'Make it harder'")

# --- Frontend API Requests ---
class GenerateQuizRequest(BaseModel):
    topics: List[str] = Field(description="List of topic names to generate questions for")
    questionCount: int = Field(description="Total number of questions to generate", ge=1, le=50)
    complexityHard: int = Field(description="Percentage of hard questions (0-100)", ge=0, le=100)
    formatMC: int = Field(description="Percentage of multiple choice questions (0-100)", ge=0, le=100)

class RefineQuestionRequest(BaseModel):
    question: QuizQuestion = Field(description="Updated question object with new question_text")

class RerollQuestionRequest(BaseModel):
    id: str = Field(description="Question ID to regenerate")
    question: QuizQuestion = Field(description="Current question to regenerate")