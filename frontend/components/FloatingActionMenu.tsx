"use client";

interface FloatingActionMenuProps {
  onAddQuestion: () => void;
  onShuffle: () => void;
  onExport: () => void;
}

export default function FloatingActionMenu({
  onAddQuestion,
  onShuffle,
  onExport,
}: FloatingActionMenuProps) {
  return (
    <div className="sticky bottom-4 left-0 right-0 z-10 flex justify-center pt-4">
      <div className="glass-dock flex items-center gap-1 rounded-2xl px-4 py-2 shadow-xl">
        <button
          type="button"
          onClick={onAddQuestion}
          className="flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium text-zinc-300 transition hover:bg-white/10 hover:text-white"
        >
          <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Add Question
        </button>
        <button
          type="button"
          onClick={onShuffle}
          className="flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium text-zinc-300 transition hover:bg-white/10 hover:text-white"
        >
          <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Shuffle Order
        </button>
        <button
          type="button"
          onClick={onExport}
          title="Export (PDF / Google Forms / LMS)"
          className="flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium text-zinc-300 transition hover:bg-white/10 hover:text-white"
        >
          <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Export
        </button>
      </div>
    </div>
  );
}
