"use client";

import { useState, useCallback, useEffect } from "react";
import { fetchTopics } from "@/lib/api";

interface QuizDNAProps {
  onGenerate: (opts: {
    topics: string[];
    questionCount: number;
    complexityHard: number;
    formatMC: number;
  }) => void;
}

export default function QuizDNA({ onGenerate }: QuizDNAProps) {
  const [search, setSearch] = useState("");
  const [topics, setTopics] = useState<string[]>([]);
  const [availableTopics, setAvailableTopics] = useState<string[]>([]);
  const [questionCount, setQuestionCount] = useState(25);
  const [complexityHard, setComplexityHard] = useState(60);
  const [formatMC, setFormatMC] = useState(70);
  const [isLoadingTopics, setIsLoadingTopics] = useState(true);
  const [isSearchFocused, setIsSearchFocused] = useState(false);

  // Fetch topics from backend on component mount
  useEffect(() => {
    async function loadTopics() {
      setIsLoadingTopics(true);
      try {
        const topicData = await fetchTopics();
        const topicNames = topicData.map((t) => t.topic_name);
        setAvailableTopics(topicNames);
      } catch (error) {
        console.error("Failed to load topics:", error);
      } finally {
        setIsLoadingTopics(false);
      }
    }
    loadTopics();
  }, []);

  const addTopic = useCallback(
    (t: string) => {
      const trimmed = t.trim();
      if (trimmed && !topics.includes(trimmed)) setTopics((prev) => [...prev, trimmed]);
      setSearch("");
    },
    [topics]
  );

  const removeTopic = useCallback((t: string) => {
    setTopics((prev) => prev.filter((x) => x !== t));
  }, []);

  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addTopic(search);
    }
  };

  const handleGenerate = () => {
    onGenerate({ topics, questionCount, complexityHard, formatMC });
  };

  const filteredSuggestions = availableTopics.filter(
    (t) =>
      !topics.includes(t) &&
      (search === "" || t.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <aside className="glass flex h-full w-[320px] shrink-0 flex-col gap-6 rounded-2xl p-5">
      <h2 className="font-semibold text-white">Quiz DNA</h2>

      {/* Topic Selection */}
      <div className="space-y-2">
        <div className="relative">
          <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400">
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </span>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={handleSearchKeyDown}
            onFocus={() => setIsSearchFocused(true)}
            onClick={() => setIsSearchFocused(true)}
            onBlur={() => {
              // Delay to allow click events on suggestions to fire
              setTimeout(() => setIsSearchFocused(false), 200);
            }}
            placeholder="Search & add topics..."
            className="w-full rounded-lg border border-white/10 bg-white/5 py-2 pl-9 pr-3 text-sm text-white placeholder-zinc-500 focus:border-violet-500/50 focus:outline-none focus:ring-1 focus:ring-violet-500/50"
          />
        </div>
        <p className="text-xs text-zinc-400">Main Topics</p>
        {isLoadingTopics ? (
          <p className="text-xs text-zinc-500">Loading topics...</p>
        ) : availableTopics.length === 0 ? (
          <p className="text-xs text-zinc-500">No topics available. Please ingest documents first.</p>
        ) : null}
        <div className="flex flex-wrap gap-2">
          {topics.map((t) => (
            <span
              key={t}
              className="inline-flex items-center gap-1 rounded-full border border-violet-500/40 bg-white/10 px-3 py-1 text-sm text-white"
            >
              {t}
              <button
                type="button"
                onClick={() => removeTopic(t)}
                className="ml-0.5 rounded-full p-0.5 text-zinc-400 hover:bg-white/10 hover:text-white"
                aria-label={`Remove ${t}`}
              >
                ×
              </button>
            </span>
          ))}
        </div>
        {isSearchFocused && filteredSuggestions.length > 0 && (
          <div className="max-h-60 overflow-y-auto rounded-lg border border-white/10 bg-zinc-900/80 py-1">
            {filteredSuggestions.slice(0, 20).map((t) => (
              <button
                key={t}
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  addTopic(t);
                  setIsSearchFocused(false);
                }}
                className="block w-full px-3 py-2 text-left text-sm text-zinc-300 hover:bg-white/5 hover:text-white"
              >
                {t}
              </button>
            ))}
            {filteredSuggestions.length > 20 && (
              <div className="px-3 py-2 text-xs text-zinc-500">
                Showing 20 of {filteredSuggestions.length} topics. Type to filter...
              </div>
            )}
          </div>
        )}
      </div>

      {/* Power Sliders */}
      <div className="space-y-5">
        <div>
          <div className="mb-1 flex justify-between text-sm">
            <span className="text-zinc-400">No. of Questions</span>
            <span className="font-medium text-white">{questionCount}</span>
          </div>
          <input
            type="range"
            min={1}
            max={50}
            value={questionCount}
            onChange={(e) => setQuestionCount(Number(e.target.value))}
            className="w-full"
          />
          <div className="mt-0.5 flex justify-between text-xs text-zinc-500">
            <span>1</span>
            <span>50</span>
          </div>
        </div>

        <div>
          <div className="mb-1 flex justify-between text-sm">
            <span className="text-zinc-400">Complexity Ratio</span>
            <span className="font-medium text-white">{complexityHard}% Hard</span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={complexityHard}
            onChange={(e) => setComplexityHard(Number(e.target.value))}
            className="w-full"
          />
          <div className="mt-0.5 flex justify-between text-xs text-zinc-500">
            <span>Easy</span>
            <span>Hard</span>
          </div>
        </div>

        <div>
          <div className="mb-1 flex justify-between text-sm">
            <span className="text-zinc-400">Format Ratio</span>
            <span className="font-medium text-white">{formatMC}% MC</span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={formatMC}
            onChange={(e) => setFormatMC(Number(e.target.value))}
            className="w-full"
          />
          <div className="mt-0.5 flex justify-between text-xs text-zinc-500">
            <span>Short Answer</span>
            <span>MC</span>
          </div>
        </div>
      </div>

      {/* Generate Quiz CTA */}
      <button
        type="button"
        onClick={handleGenerate}
        className="glow-violet mt-auto flex w-full items-center justify-center gap-2 rounded-xl bg-violet-600 py-3 font-medium text-white transition hover:bg-violet-500"
      >
        <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
        </svg>
        Generate Quiz
      </button>
    </aside>
  );
}
