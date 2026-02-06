// Mock data types and sample data for development

export type Difficulty = "Easy" | "Medium" | "Hard";

export interface QuizQuestion {
  id: string;
  type: "MCQ" | "Short Answer";
  difficulty: Difficulty;
  question_text: string;
  options: string[] | null;
  correct_answer: string;
  marking_rubric?: string;
  source_context?: string;
  topic?: string;
  concept?: string;
}

export const MOCK_QUESTIONS: QuizQuestion[] = [
  {
    id: "q1",
    type: "MCQ",
    difficulty: "Easy",
    question_text: "What is the capital of France?",
    options: ["London", "Berlin", "Paris", "Madrid"],
    correct_answer: "Paris",
  },
  {
    id: "q2",
    type: "Short Answer",
    difficulty: "Medium",
    question_text: "Explain the concept of photosynthesis in one sentence.",
    options: null,
    correct_answer: "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose molecules.",
  },
  {
    id: "q3",
    type: "MCQ",
    difficulty: "Hard",
    question_text: "Which of the following is NOT a characteristic of a well-designed API?",
    options: [
      "RESTful principles",
      "Consistent naming conventions",
      "Tight coupling between services",
      "Versioning support",
    ],
    correct_answer: "Tight coupling between services",
  },
];
