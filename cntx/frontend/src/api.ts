import type { GraphBundle, Note, TagSummary } from './types';

const API_BASE =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:4000';

async function request<T>(path: string, options?: RequestInit) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || 'Request failed');
  }

  return (await response.json()) as T;
}

export async function listNotes(query?: string) {
  const search = query ? `?q=${encodeURIComponent(query)}` : '';
  return request<{ notes: Note[]; query?: string }>(`/api/notes${search}`);
}

export interface NotePayload {
  content: string;
  manualTags: string[];
  useAi?: boolean;
}

export async function createNote(payload: NotePayload) {
  return request<{ note: Note }>('/api/notes', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function updateNote(id: string, payload: Partial<NotePayload>) {
  return request<{ note: Note }>(`/api/notes/${id}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  });
}

export async function fetchGraph() {
  return request<GraphBundle>('/api/graph');
}

export async function listTags() {
  return request<{ tags: TagSummary[] }>('/api/tags');
}
