import { useEffect, useState } from 'react';

interface SearchPanelProps {
  initialQuery?: string;
  isSearching: boolean;
  onSearch: (query: string) => void;
  onReset: () => void;
}

export function SearchPanel({
  initialQuery = '',
  isSearching,
  onSearch,
  onReset,
}: SearchPanelProps) {
  const [value, setValue] = useState(initialQuery);

  useEffect(() => {
    setValue(initialQuery);
  }, [initialQuery]);

  const handleSearch = (event: React.FormEvent) => {
    event.preventDefault();
    onSearch(value.trim());
  };

  const handleReset = () => {
    setValue('');
    onReset();
  };

  return (
    <form className="search-panel" onSubmit={handleSearch}>
      <div className="search-input">
        <input
          type="search"
          placeholder="Search last week’s prompts, agent behaviors, or tags…"
          value={value}
          onChange={(event) => setValue(event.target.value)}
        />
        <button type="submit" disabled={isSearching}>
          {isSearching ? 'Searching…' : 'Search'}
        </button>
        <button type="button" className="ghost" onClick={handleReset} disabled={!value}>
          Reset
        </button>
      </div>
      <p className="search-helper">
        Use plain language. We’ll handle semantic + tag-level retrieval for both precise and fuzzy matches.
      </p>
    </form>
  );
}
