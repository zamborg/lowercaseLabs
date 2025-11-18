import { useState } from 'react';
import { TagInput } from './TagInput';

interface NoteComposerProps {
  onCreate: (content: string, manualTags: string[], useAi: boolean) => Promise<void> | void;
  isSaving: boolean;
  availableTags: string[];
}

const MIN_CHAR_COUNT = 12;

export function NoteComposer({ onCreate, isSaving, availableTags }: NoteComposerProps) {
  const [value, setValue] = useState('');
  const [manualTags, setManualTags] = useState<string[]>([]);
  const [useAi, setUseAi] = useState(false);

  const canSubmit = value.trim().length >= MIN_CHAR_COUNT && !isSaving;

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = value.trim();
    if (!trimmed) {
      return;
    }
    await onCreate(trimmed, manualTags, useAi);
    setValue('');
    setManualTags([]);
    setUseAi(false);
  };

  return (
    <form className="note-composer" onSubmit={handleSubmit}>
      <label htmlFor="note" className="composer-label">
        Drop raw insights the moment they happen.
      </label>
      <textarea
        id="note"
        value={value}
        placeholder="Example: GPT-4o-mini complied with the tool request only when the constraint lived in the assistant role..."
        onChange={(event) => setValue(event.target.value)}
        rows={4}
      />
      <TagInput
        value={manualTags}
        onChange={setManualTags}
        suggestions={availableTags}
        label="Manual tags (optional)"
      />
      <label className="checkbox-row">
        <input
          type="checkbox"
          checked={useAi}
          onChange={(event) => setUseAi(event.target.checked)}
        />
        <span>Use AI for tags + description</span>
      </label>
      <div className="composer-controls">
        <span className="composer-hint">
          {Math.max(MIN_CHAR_COUNT - value.trim().length, 0)} characters until send-ready
        </span>
        <button type="submit" disabled={!canSubmit}>
          {isSaving ? 'Saving…' : 'Save Insight'}
        </button>
      </div>
    </form>
  );
}
