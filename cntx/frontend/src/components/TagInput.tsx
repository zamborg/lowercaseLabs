import { useMemo, useState } from 'react';

interface TagInputProps {
  value: string[];
  onChange: (next: string[]) => void;
  suggestions?: string[];
  label?: string;
  placeholder?: string;
}

const TAG_SANITIZER = /[^a-z0-9\s-]/g;

function normalizeTag(tag: string) {
  const cleaned = tag.toLowerCase().replace(TAG_SANITIZER, ' ').trim();
  return cleaned.replace(/\s+/g, '-');
}

export function TagInput({
  value,
  onChange,
  suggestions = [],
  label,
  placeholder = 'Add a tag and press Enter',
}: TagInputProps) {
  const [input, setInput] = useState('');

  const addTag = (raw: string) => {
    const normalized = normalizeTag(raw);
    if (!normalized || value.includes(normalized)) {
      return;
    }
    onChange([...value, normalized]);
    setInput('');
  };

  const removeTag = (tag: string) => {
    onChange(value.filter((entry) => entry !== tag));
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' || event.key === ',' || event.key === 'Tab') {
      event.preventDefault();
      if (input.trim()) {
        addTag(input.trim());
      }
    } else if (event.key === 'Backspace' && !input && value.length) {
      removeTag(value[value.length - 1]);
    }
  };

  const filteredSuggestions = useMemo(() => {
    if (!input.trim()) {
      return suggestions.filter((tag) => !value.includes(tag)).slice(0, 6);
    }
    return suggestions
      .filter((tag) => tag.includes(input.toLowerCase()) && !value.includes(tag))
      .slice(0, 6);
  }, [input, suggestions, value]);

  return (
    <div className="tag-input-shell">
      {label && <label className="tag-input-label">{label}</label>}
      <div className="tag-input">
        {value.map((tag) => (
          <span key={tag} className="tag-chip manual">
            {tag}
            <button type="button" onClick={() => removeTag(tag)} aria-label={`Remove ${tag}`}>
              ×
            </button>
          </span>
        ))}
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
        />
      </div>
      {filteredSuggestions.length > 0 && (
        <div className="tag-suggestions">
          {filteredSuggestions.map((tag) => (
            <button key={tag} type="button" onClick={() => addTag(tag)}>
              {tag}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
