"use client";

import { useState, useCallback, useEffect } from "react";
import type { QuizQuestion } from "@/lib/mockData";
import { MOCK_QUESTIONS } from "@/lib/mockData";
import type { QuizResponse, QuizQuestion as ApiQuizQuestion } from "@/lib/api";
import QuestionCard from "./QuestionCard";
import FloatingActionMenu from "./FloatingActionMenu";

interface CanvasWorkspaceProps {
  generated: boolean;
  quizData: QuizResponse | null;
  isGenerating: boolean;
  streamingQuestions?: ApiQuizQuestion[];
}

// Convert API QuizQuestion to frontend QuizQuestion format
function convertApiQuestion(apiQ: ApiQuizQuestion): QuizQuestion {
  return {
    id: apiQ.id,
    type: apiQ.type,
    difficulty: apiQ.difficulty,
    question_text: apiQ.question_text,
    options: apiQ.options || null,
    correct_answer: apiQ.correct_answer,
  };
}

export default function CanvasWorkspace({ generated, quizData, isGenerating, streamingQuestions }: CanvasWorkspaceProps) {
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);

  // Update questions when streamingQuestions or quizData changes
  useEffect(() => {
    if (streamingQuestions && streamingQuestions.length > 0) {
      // Use streaming questions if available (for real-time updates)
      const convertedQuestions = streamingQuestions.map(convertApiQuestion);
      setQuestions(convertedQuestions);
    } else if (quizData && quizData.questions) {
      // Fallback to quizData if no streaming questions
      const convertedQuestions = quizData.questions.map(convertApiQuestion);
      setQuestions(convertedQuestions);
    } else if (generated && !quizData && !streamingQuestions) {
      // Fallback to mock data if generated but no quizData (for backward compatibility)
      setQuestions(MOCK_QUESTIONS);
    } else {
      setQuestions([]);
    }
  }, [quizData, generated, streamingQuestions]);

  const handleRefine = useCallback((id: string, text: string) => {
    setQuestions((prev) =>
      prev.map((q) => (q.id === id ? { ...q, question_text: text } : q))
    );
  }, []);

  const handleReroll = useCallback((id: string) => {
    // Mock: slightly alter question text to simulate re-roll
    setQuestions((prev) =>
      prev.map((q) =>
        q.id === id
          ? { ...q, question_text: `${q.question_text} (regenerated)` }
          : q
      )
    );
  }, []);

  const handleDelete = useCallback((id: string) => {
    setQuestions((prev) => prev.filter((q) => q.id !== id));
  }, []);

  const handleAddQuestion = useCallback(() => {
    const n = questions.length + 1;
    setQuestions((prev) => [
      ...prev,
      {
        id: `q${Date.now()}`,
        type: "MCQ",
        difficulty: "Medium",
        question_text: `New question ${n}. (Edit to customize.)`,
        options: ["Option A", "Option B", "Option C"],
        correct_answer: "Option A",
      },
    ]);
  }, [questions.length]);

  const handleShuffle = useCallback(() => {
    setQuestions((prev) => [...prev].sort(() => Math.random() - 0.5));
  }, []);

  const handleExport = useCallback(() => {
    // Mock: log for now
    console.log("Export (PDF / Google Forms / LMS)", questions);
    alert("Export: PDF / Google Forms / LMS (mock – backend not connected)");
  }, [questions]);

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="font-semibold text-white">Canvas Workspace</h2>
        <span className="text-sm text-zinc-500">
          {new Date().toLocaleString("en-US", {
            weekday: "long",
            year: "numeric",
            month: "long",
            day: "numeric",
            hour: "numeric",
            minute: "2-digit",
            second: "2-digit",
            timeZoneName: "short",
          })}
        </span>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto pr-2">
        {isGenerating && questions.length === 0 && (
          <div className="glass flex flex-col items-center justify-center rounded-2xl py-20 text-center">
            <p className="mb-2 text-zinc-400">Generating quiz...</p>
            <p className="text-sm text-zinc-500">Please wait while we create your questions.</p>
          </div>
        )}
        {!isGenerating && questions.length === 0 && !generated && (
          <div className="glass flex flex-col items-center justify-center rounded-2xl py-20 text-center">
            <p className="mb-2 text-zinc-400">No questions yet</p>
            <p className="text-sm text-zinc-500">
              Use Quiz DNA on the left to set topics and sliders, then click Generate Quiz.
            </p>
          </div>
        )}
        {!isGenerating && questions.length === 0 && generated && (
          <div className="glass rounded-2xl p-8 text-center text-zinc-400">
            All questions removed. Use Add Question or Generate Quiz again.
          </div>
        )}
        {questions.map((q, i) => (
          <QuestionCard
            key={q.id}
            question={q}
            index={i + 1}
            onRefine={handleRefine}
            onReroll={handleReroll}
            onDelete={handleDelete}
          />
        ))}
        {isGenerating && questions.length > 0 && (
          <div className="glass flex flex-col items-center justify-center rounded-2xl py-8 text-center">
            <p className="mb-2 text-zinc-400">Generating more questions...</p>
            <p className="text-sm text-zinc-500">{questions.length} question{questions.length !== 1 ? 's' : ''} generated so far</p>
          </div>
        )}
      </div>

      {generated && (
        <FloatingActionMenu
          onAddQuestion={handleAddQuestion}
          onShuffle={handleShuffle}
          onExport={handleExport}
        />
      )}
    </div>
  );
}
