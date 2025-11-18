import { useEffect, useState } from 'react';
import type { Note } from '../types';
import { TagInput } from './TagInput';

interface NoteCardProps {
  note: Note;
  onUpdate: (id: string, payload: { content: string; manualTags: string[]; useAi?: boolean }) => Promise<void>;
  availableTags: string[];
}

function formatDate(value: string) {
  const date = new Date(value);
  return new Intl.DateTimeFormat('en', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric',
  }).format(date);
}

function computeFreshness(createdAt: string) {
  const created = new Date(createdAt).getTime();
  const ageHours = (Date.now() - created) / (1000 * 60 * 60);
  if (ageHours <= 4) {
    return 'new';
  }
  if (ageHours <= 24) {
    return 'recent';
  }
  return 'stale';
}

export function NoteCard({ note, onUpdate, availableTags }: NoteCardProps) {
  const freshness = computeFreshness(note.createdAt);
  const [isEditing, setIsEditing] = useState(false);
  const [draftContent, setDraftContent] = useState(note.content);
  const [draftTags, setDraftTags] = useState<string[]>(note.manualTags);
  const [isSaving, setIsSaving] = useState(false);
  const [useAi, setUseAi] = useState(false);

  useEffect(() => {
    setDraftContent(note.content);
    setDraftTags(note.manualTags);
  }, [note.content, note.manualTags]);

  const manualTagSet = new Set(note.manualTags);

  const handleSave = async () => {
    setIsSaving(true);
    await onUpdate(note.id, { content: draftContent, manualTags: draftTags, useAi });
    setIsSaving(false);
    setIsEditing(false);
    setUseAi(false);
  };

  return (
    <article className={`note-card ${freshness}`}>
      <header>
        <span className="timestamp">{formatDate(note.createdAt)}</span>
        {typeof note.score === 'number' && (
          <span className="score">match {(note.score * 100).toFixed(0)}%</span>
        )}
        <button type="button" className="ghost small" onClick={() => setIsEditing((prev) => !prev)}>
          {isEditing ? 'Close' : 'Edit'}
        </button>
      </header>
      {!isEditing && (
        <>
          <p className="note-body">{note.content}</p>
          {note.tags.length > 0 && (
            <ul className="tag-row">
              {note.tags.map((tag) => (
                <li key={tag} className={manualTagSet.has(tag) ? 'manual' : undefined}>
                  {tag}
                </li>
              ))}
            </ul>
          )}
          <p className="link-summary">{note.linkSummary}</p>
        </>
      )}
      {isEditing && (
        <div className="note-editor">
          <textarea value={draftContent} onChange={(event) => setDraftContent(event.target.value)} rows={4} />
          <TagInput value={draftTags} onChange={setDraftTags} suggestions={availableTags} />
          <label className="checkbox-row">
            <input type="checkbox" checked={useAi} onChange={(event) => setUseAi(event.target.checked)} />
            <span>Use AI for tags + description</span>
          </label>
          <div className="editor-actions">
            <button type="button" className="ghost" onClick={() => setIsEditing(false)}>
              Cancel
            </button>
            <button type="button" onClick={handleSave} disabled={isSaving || !draftContent.trim()}>
              {isSaving ? 'Updating…' : 'Save changes'}
            </button>
          </div>
        </div>
      )}
    </article>
  );
}
