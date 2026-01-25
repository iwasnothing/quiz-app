// API client for the quiz backend

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Types matching backend models
export interface TopicResponse {
  topic_name: string;
  sub_concepts: string[];
}

export interface QuizQuestion {
  id: string;
  type: "MCQ" | "Short Answer";
  difficulty: "Easy" | "Medium" | "Hard";
  question_text: string;
  options: string[] | null;
  correct_answer: string;
  marking_rubric?: string;
  source_context?: string;
}

export interface QuizResponse {
  title: string;
  questions: QuizQuestion[];
}

// Fetch available topics from the backend
export async function fetchTopics(): Promise<TopicResponse[]> {
  try {
    const response = await fetch(`${API_URL}/topics`);
    if (!response.ok) {
      throw new Error(`Failed to fetch topics: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching topics:', error);
    throw error;
  }
}

// Generate quiz stream callback types
export type OnQuestionCallback = (question: QuizQuestion) => void;
export type OnStartCallback = (title: string, total: number) => void;
export type OnDoneCallback = () => void;
export type OnErrorCallback = (error: string) => void;

// Generate quiz with streaming support
export async function generateQuizStream(
  opts: {
    topics: string[];
    questionCount: number;
    complexityHard: number;
    formatMC: number;
  },
  onQuestion: OnQuestionCallback,
  onStart: OnStartCallback,
  onDone: OnDoneCallback,
  onError: OnErrorCallback
): Promise<void> {
  try {
    const response = await fetch(`${API_URL}/generate-quiz`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        topics: opts.topics,
        questionCount: opts.questionCount,
        complexityHard: opts.complexityHard,
        formatMC: opts.formatMC,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to generate quiz: ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            
            switch (data.type) {
              case 'start':
                onStart(data.title, data.total);
                break;
              case 'question':
                if (data.question) {
                  onQuestion(data.question as QuizQuestion);
                }
                break;
              case 'done':
                onDone();
                return;
              case 'error':
                onError(data.message || 'Unknown error occurred');
                return;
            }
          } catch (parseError) {
            console.error('Error parsing SSE data:', parseError);
          }
        }
      }
    }

    // If we exit the loop without a 'done' event, call onDone anyway
    onDone();
  } catch (error) {
    console.error('Error in generateQuizStream:', error);
    onError(error instanceof Error ? error.message : 'Unknown error occurred');
  }
}
