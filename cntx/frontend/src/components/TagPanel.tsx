import type { TagSummary } from '../types';

interface TagPanelProps {
  tags: TagSummary[];
  onRefresh: () => void;
  isLoading: boolean;
}

export function TagPanel({ tags, onRefresh, isLoading }: TagPanelProps) {
  return (
    <div className="tag-panel panel">
      <div className="section-heading">
        <div>
          <h2>Active tags</h2>
          <p>Shared vocabulary powering retrieval + graph links. Click refresh anytime to sync.</p>
        </div>
        <button type="button" onClick={onRefresh} disabled={isLoading}>
          {isLoading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
      {tags.length === 0 ? (
        <p className="empty-copy">No tags yet. Start adding notes or assign manual tags to seed this list.</p>
      ) : (
        <ul className="tag-list">
          {tags.map((tag) => (
            <li key={tag.name}>
              <span>{tag.name}</span>
              <span className="count">{tag.noteCount}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
