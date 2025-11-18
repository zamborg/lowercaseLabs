import type { Note } from '../types';
import { NoteCard } from './NoteCard';

interface NoteListProps {
  notes: Note[];
  isLoading: boolean;
  query?: string;
  onRefresh: () => void;
  onUpdate: (id: string, payload: { content: string; manualTags: string[]; useAi?: boolean }) => Promise<void>;
  availableTags: string[];
}

export function NoteList({ notes, isLoading, query, onRefresh, onUpdate, availableTags }: NoteListProps) {
  if (isLoading) {
    return (
      <div className="note-list loading">
        <p>Gathering the latest corpus…</p>
      </div>
    );
  }

  if (!notes.length) {
    return (
      <div className="note-list empty">
        <p>No notes yet{query ? ` for “${query}”` : ''}.</p>
        <button type="button" onClick={onRefresh}>
          Refresh
        </button>
      </div>
    );
  }

  return (
    <div className="note-list">
      {notes.map((note) => (
        <NoteCard key={note.id} note={note} onUpdate={onUpdate} availableTags={availableTags} />
      ))}
    </div>
  );
}
