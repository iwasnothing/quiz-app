"use client";

import { useState } from "react";
import QuizDNA from "@/components/QuizDNA";
import CanvasWorkspace from "@/components/CanvasWorkspace";
import { generateQuizStream, type QuizQuestion as ApiQuizQuestion, type QuizResponse } from "@/lib/api";

export default function Home() {
  const [generated, setGenerated] = useState(false);
  const [quizData, setQuizData] = useState<QuizResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [questions, setQuestions] = useState<ApiQuizQuestion[]>([]);
  const [quizTitle, setQuizTitle] = useState<string>("");

  const handleGenerate = async (opts: {
    topics: string[];
    questionCount: number;
    complexityHard: number;
    formatMC: number;
  }) => {
    setIsGenerating(true);
    setGenerated(false);
    setQuestions([]);
    setQuizTitle("");
    
    let collectedQuestions: ApiQuizQuestion[] = [];
    let finalTitle = "";
    
    try {
      await generateQuizStream(
        opts,
        // onQuestion: called when each question is generated
        (question: ApiQuizQuestion) => {
          collectedQuestions.push(question);
          setQuestions((prev) => [...prev, question]);
          // Mark as generated once we have at least one question
          setGenerated(true);
        },
        // onStart: called when generation starts
        (title: string, total: number) => {
          finalTitle = title;
          setQuizTitle(title);
          console.log(`Starting quiz generation: ${title}, ${total} questions`);
        },
        // onDone: called when generation is complete
        () => {
          setIsGenerating(false);
          // Set final quiz data
          setQuizData({
            title: finalTitle || `Quiz: ${opts.topics.join(", ")}`,
            questions: collectedQuestions,
          });
          console.log("Quiz generation completed");
        },
        // onError: called if there's an error
        (error: string) => {
          console.error("Failed to generate quiz:", error);
          alert(`Failed to generate quiz: ${error}`);
          setIsGenerating(false);
        }
      );
    } catch (error) {
      console.error("Failed to generate quiz:", error);
      alert("Failed to generate quiz. Please try again.");
      setIsGenerating(false);
    }
  };

  return (
    <div className="mesh-bg flex min-h-screen flex-col">
      <header className="shrink-0 border-b border-white/10 px-6 py-4">
        <h1 className="text-xl font-semibold text-white">
          QuizGenius — AI-Powered Quiz Generator
        </h1>
      </header>

      <div className="flex min-h-0 flex-1 gap-6 overflow-hidden px-6 py-6">
        <QuizDNA onGenerate={handleGenerate} />
        <main className="glass flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl p-6">
          <CanvasWorkspace 
            key={generated ? "generated" : "empty"} 
            generated={generated} 
            quizData={questions.length > 0 ? { title: quizTitle || "Quiz", questions } : null}
            isGenerating={isGenerating}
            streamingQuestions={questions}
          />
        </main>
      </div>
    </div>
  );
}
