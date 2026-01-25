from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List
import json

import ingest
import generator
from database import get_main_topics_from_sqlite, query_all_topics
from models import (
    QuizSchema, 
    TopicResponse, 
    RefineRequest, 
    QuizQuestion,
    GenerateQuizRequest,
    RefineQuestionRequest,
    RerollQuestionRequest
)

app = FastAPI(title="Teacher Quiz Canvas API")

# CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "running", "service": "DGX Quiz Backend"}

# --- 1. Ingestion ---
@app.post("/ingest", response_model=dict)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document (PDF, DOCX, etc.) for quiz generation.
    """
    try:
        num_chunks = await ingest.process_file(file)
        return {"message": "Success", "filename": file.filename, "chunks_processed": num_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 2. Topic Discovery (SQLite) ---
@app.get("/topics", response_model=List[TopicResponse])
def get_topics():
    """
    Returns main topics from SQLite (document_chunks.topic_name + aggregated sub_concepts).
    Used by frontend to populate topic suggestions.
    """
    try:
        rows = get_main_topics_from_sqlite()
        return [TopicResponse(topic_name=r["topic_name"], sub_concepts=r["sub_concepts"]) for r in rows]
    except Exception as e:
        print(f"Topics Error: {e}")
        return []

@app.get("/topics/names", response_model=List[str])
def get_topic_names():
    """
    Returns all unique topic names from SQLite database.
    Simpler endpoint that returns just topic names.
    """
    try:
        return query_all_topics()
    except Exception as e:
        print(f"Topic Names Error: {e}")
        return []

# --- 3. Quiz Generation (Frontend API) - Streaming ---
@app.post("/generate-quiz")
def create_quiz(request: GenerateQuizRequest):
    """
    Generate a quiz based on topics, question count, complexity ratio, and format ratio.
    Streams questions as they are generated using Server-Sent Events (SSE).
    This is the main endpoint called by the frontend Quiz DNA sidebar.
    """
    if not request.topics or len(request.topics) == 0:
        raise HTTPException(status_code=400, detail="At least one topic is required")
    
    def generate():
        try:
            topic_str = ", ".join(request.topics)
            # Send initial metadata
            yield f"data: {json.dumps({'type': 'start', 'title': f'Quiz: {topic_str}', 'total': request.questionCount})}\n\n"
            
            # Stream questions as they are generated
            for question in generator.generate_quiz_chain_streaming(
                topics=request.topics,
                count=request.questionCount,
                complexity_hard=request.complexityHard,
                format_mc=request.formatMC
            ):
                # Convert QuizQuestion to dict for JSON serialization
                question_dict = question.dict()
                yield f"data: {json.dumps({'type': 'question', 'question': question_dict})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            print(f"Generate Quiz Error: {e}")
            import traceback
            traceback.print_exc()
            # Send error in stream
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )

# --- 4. Legacy Quiz Generation (for backward compatibility) ---
@app.post("/generate-quiz-legacy", response_model=QuizSchema)
def create_quiz_legacy(topic: str, difficulty: str = "Medium", count: int = 5):
    """
    Legacy endpoint for backward compatibility.
    Use /generate-quiz instead.
    """
    try:
        topics = [t.strip() for t in topic.split(',')] if ',' in topic else [topic]
        # Default to 50% hard, 50% MC for legacy calls
        return generator.generate_quiz_chain(topics=topics, count=count, complexity_hard=50, format_mc=50)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. Refine Question (Frontend API - Simplified) ---
@app.post("/refine-question", response_model=QuizQuestion)
def refine_question_simple(request: RefineQuestionRequest):
    """
    Simplified refine endpoint: accepts updated question object.
    Frontend calls this when user edits question text inline.
    For now, we just return the updated question as-is.
    In a full implementation, you could use LLM to improve/validate the question.
    """
    try:
        # Return the question as-is (frontend already updated the text)
        # Optionally, you could add validation or LLM-based improvement here
        return request.question
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 6. Refine Question (Advanced - with LLM feedback) ---
@app.post("/refine-question-advanced", response_model=QuizQuestion)
def refine_question_advanced(request: RefineRequest):
    """
    Advanced refine endpoint: takes a question and teacher feedback, uses LLM to improve it.
    """
    try:
        updated_q = generator.refine_question_chain(
            request.current_question.dict(), 
            request.feedback
        )
        return updated_q
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 7. Reroll Question (AI Re-roll) ---
@app.post("/reroll-question", response_model=QuizQuestion)
def reroll_question(request: RerollQuestionRequest):
    """
    Regenerate a specific question using AI while maintaining format and difficulty.
    Called when user clicks "AI Re-roll" on a question card.
    """
    try:
        # Extract topics from question if available, or use a default
        topics = ["General"]  # Could be enhanced to extract from question metadata
        
        result = generator.reroll_question_chain(
            question=request.question,
            topics=topics
        )
        # Ensure the ID matches the original
        result.id = request.id
        return result
    except Exception as e:
        print(f"Reroll Question Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
