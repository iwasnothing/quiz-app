"use client";

import { useState } from "react";
import type { QuizQuestion, Difficulty } from "@/lib/mockData";

interface QuestionCardProps {
  question: QuizQuestion;
  index: number;
  onRefine: (id: string, text: string) => void;
  onReroll: (id: string) => void;
  onDelete: (id: string) => void;
}

function DifficultyBadge({ difficulty }: { difficulty: Difficulty }) {
  const cn = difficulty === "Easy" ? "badge-easy" : difficulty === "Hard" ? "badge-hard" : "badge-medium";
  return (
    <span className={`rounded px-2 py-0.5 text-xs font-medium ${cn}`}>
      {difficulty}
    </span>
  );
}

export default function QuestionCard({ question, index, onRefine, onReroll, onDelete }: QuestionCardProps) {
  const [hover, setHover] = useState(false);
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(question.question_text);

  const handleBlur = () => {
    setEditing(false);
    if (text !== question.question_text) onRefine(question.id, text);
  };

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      className="glass group relative rounded-2xl p-5 transition-shadow hover:shadow-lg hover:shadow-violet-500/5"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <div className="mb-2 flex items-center gap-2">
            <span className="text-sm font-medium text-zinc-400">{index}.</span>
            <div className="flex flex-wrap gap-1.5">
              <DifficultyBadge difficulty={question.difficulty} />
            </div>
          </div>
          <div className="mb-2 rounded-md bg-white/5 px-2.5 py-1.5 text-xs text-zinc-400">
            <span className="text-zinc-500">Topic:</span>{" "}
            <span className="font-medium text-zinc-300">{question.topic ?? "—"}</span>
            <span className="mx-1.5 text-zinc-600">·</span>
            <span className="text-zinc-500">Concept:</span>{" "}
            <span className="font-medium text-zinc-300">{question.concept ?? "—"}</span>
          </div>
          {editing ? (
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              onBlur={handleBlur}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleBlur()}
              className="w-full resize-none rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-white focus:border-violet-500/50 focus:outline-none focus:ring-1 focus:ring-violet-500/50"
              rows={2}
              autoFocus
            />
          ) : (
            <p
              role="button"
              tabIndex={0}
              onClick={() => setEditing(true)}
              onKeyDown={(e) => e.key === "Enter" && setEditing(true)}
              className="cursor-text text-white focus:outline-none"
            >
              {text}
            </p>
          )}
          {question.options && question.options.length > 0 && (
            <ul className="mt-3 space-y-1 pl-4 text-sm text-zinc-400">
              {question.options.map((opt, i) => (
                <li key={i} className="list-disc">
                  {opt}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {hover && (
        <div className="glass-dock absolute bottom-4 right-4 flex items-center gap-1 rounded-xl px-2 py-1.5 shadow-lg">
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs text-zinc-300 transition hover:bg-white/10 hover:text-white"
            title="Refine"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
            </svg>
            Refine
          </button>
          <button
            type="button"
            onClick={() => onReroll(question.id)}
            className="rounded-lg p-1.5 text-zinc-400 transition hover:bg-white/10 hover:text-white"
            title="AI Re-roll"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
          <button
            type="button"
            onClick={() => onDelete(question.id)}
            className="flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs text-zinc-400 transition hover:bg-red-500/20 hover:text-red-400"
            title="Delete"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            Delete
          </button>
        </div>
      )}
    </div>
  );
}
