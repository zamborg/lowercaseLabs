import { useCallback, useEffect, useState } from 'react';
import './App.css';
import { createNote, fetchGraph, listNotes, listTags, updateNote } from './api';
import { NoteComposer } from './components/NoteComposer';
import { NoteList } from './components/NoteList';
import { SearchPanel } from './components/SearchPanel';
import { GraphView } from './components/GraphView';
import { TagPanel } from './components/TagPanel';
import type { GraphBundle, Note, TagSummary } from './types';

function App() {
  const [notes, setNotes] = useState<Note[]>([]);
  const [graphData, setGraphData] = useState<GraphBundle | null>(null);
  const [tags, setTags] = useState<TagSummary[]>([]);
  const [query, setQuery] = useState('');
  const [isLoadingNotes, setIsLoadingNotes] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isLoadingGraph, setIsLoadingGraph] = useState(false);
  const [isLoadingTags, setIsLoadingTags] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadNotes = useCallback(
    async (nextQuery?: string) => {
      setIsLoadingNotes(true);
      try {
        const { notes: fetchedNotes } = await listNotes(nextQuery);
        setNotes(fetchedNotes);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to load notes');
      } finally {
        setIsLoadingNotes(false);
      }
    },
    []
  );

  const loadGraph = useCallback(async () => {
    setIsLoadingGraph(true);
    try {
      const graph = await fetchGraph();
      setGraphData(graph);
    } catch (err) {
      setError((prev) => prev ?? 'Unable to load concept graph');
    } finally {
      setIsLoadingGraph(false);
    }
  }, []);

  const loadTagDirectory = useCallback(async () => {
    setIsLoadingTags(true);
    try {
      const { tags: fetchedTags } = await listTags();
      setTags(fetchedTags);
    } catch (err) {
      setError((prev) => prev ?? 'Unable to load tags');
    } finally {
      setIsLoadingTags(false);
    }
  }, []);

  useEffect(() => {
    loadNotes();
    loadGraph();
    loadTagDirectory();
  }, [loadNotes, loadGraph, loadTagDirectory]);

  const handleCreate = async (content: string, manualTags: string[], useAi: boolean) => {
    setIsSaving(true);
    try {
      await createNote({ content, manualTags, useAi });
      await loadNotes(query || undefined);
      await loadGraph();
      await loadTagDirectory();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to create note');
    } finally {
      setIsSaving(false);
    }
  };

  const handleUpdateNote = async (
    id: string,
    payload: { content: string; manualTags: string[]; useAi?: boolean }
  ) => {
    try {
      await updateNote(id, payload);
      await loadNotes(query || undefined);
      await loadGraph();
      await loadTagDirectory();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to update note');
    }
  };

  const handleSearch = async (value: string) => {
    setQuery(value);
    await loadNotes(value || undefined);
  };

  const handleSearchReset = async () => {
    setQuery('');
    await loadNotes();
  };

  const tagNames = tags.map((tag) => tag.name);

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="eyebrow">CNTX · Context Without The O</p>
        <h1>Drop research sparks instantly. Retrieve them when you need signal.</h1>
        <p className="lede">
          CNTX keeps fast-moving AI teams aligned with raw notes, semantic search, and an auto-generated concept
          map of every experiment, prompt tweak, and agent quirk.
        </p>
      </header>

      {error && (
        <div className="banner" role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      <section className="panel note-panel">
        <NoteComposer onCreate={handleCreate} isSaving={isSaving} availableTags={tagNames} />
      </section>

      <section className="panel search-panel-wrapper">
        <SearchPanel
          initialQuery={query}
          isSearching={isLoadingNotes && Boolean(query)}
          onSearch={handleSearch}
          onReset={handleSearchReset}
        />
      </section>

      <section>
        <TagPanel tags={tags} onRefresh={loadTagDirectory} isLoading={isLoadingTags} />
      </section>

      <section className="panel notes-panel">
        <div className="section-heading">
          <div>
            <h2>Latest notes</h2>
            <p>Raw text on top, AI-generated tags + relational summary underneath.</p>
          </div>
          <button type="button" onClick={() => loadNotes(query || undefined)} disabled={isLoadingNotes}>
            {isLoadingNotes ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
        <NoteList
          notes={notes}
          isLoading={isLoadingNotes}
          query={query}
          onRefresh={() => loadNotes(query || undefined)}
          onUpdate={handleUpdateNote}
          availableTags={tagNames}
        />
      </section>

      <section className="panel graph-panel">
        <GraphView
          title="Note association graph"
          description="Notes link together when they share rarer tags. Heavier edges signal higher-shared, high-IDF tags."
          data={graphData?.noteGraph ?? null}
          isLoading={isLoadingGraph}
          onRefresh={loadGraph}
          emptyHint="Add a few notes with overlapping tags to visualize associations."
        />
      </section>

      <section className="panel graph-panel">
        <GraphView
          title="Tag association graph"
          description="Tags connect when they co-occur on notes. Use this to see topical clusters emerging in your workspace."
          data={graphData?.tagGraph ?? null}
          isLoading={isLoadingGraph}
          onRefresh={loadGraph}
          emptyHint="Tags will appear here once you add notes with overlapping labels."
        />
      </section>
    </div>
  );
}

export default App;
