import { useEffect, useMemo, useRef, useState } from 'react';
import type { GraphData, GraphEdge, GraphNode } from '../types';

interface GraphViewProps {
  title: string;
  description: string;
  data: GraphData | null;
  isLoading: boolean;
  onRefresh: () => void;
  emptyHint: string;
}

interface PositionedNode extends GraphNode {
  x: number;
  y: number;
}

interface PositionedEdge extends GraphEdge {
  start: { x: number; y: number };
  end: { x: number; y: number };
}

interface Layout {
  nodes: PositionedNode[];
  edges: PositionedEdge[];
}

export function GraphView({ title, description, data, isLoading, onRefresh, emptyHint }: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 640, height: 420 });

  useEffect(() => {
    const handleResize = () => {
      const element = containerRef.current;
      if (!element) {
        return;
      }
      setDimensions({
        width: element.clientWidth,
        height: element.clientHeight,
      });
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const layout = useMemo<Layout>(() => {
    if (!data || !data.nodes.length) {
      return { nodes: [], edges: [] };
    }
    const nodes = data.nodes.slice(0, 18);
    const radius = Math.max(Math.min(dimensions.width, dimensions.height) / 2 - 70, 80);
    const center = { x: dimensions.width / 2, y: dimensions.height / 2 };
    const positionedNodes = nodes.map((node, index) => {
      const angle = (2 * Math.PI * index) / nodes.length;
      return {
        ...node,
        x: center.x + radius * Math.cos(angle),
        y: center.y + radius * Math.sin(angle),
      };
    });
    const nodeMap = positionedNodes.reduce<Record<string, { x: number; y: number }>>((acc, node) => {
      acc[node.id] = { x: node.x, y: node.y };
      return acc;
    }, {});

    const edges = (data.edges || [])
      .map((edge) => {
        const start = nodeMap[edge.source];
        const end = nodeMap[edge.target];
        if (!start || !end) {
          return null;
        }
        return { ...edge, start, end };
      })
      .filter((edge): edge is PositionedEdge => Boolean(edge))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 40);

    return { nodes: positionedNodes, edges };
  }, [data, dimensions]);

  const highlightEdges = layout.edges.slice(0, 5);
  const labelMap = useMemo(() => {
    const map = new Map<string, string>();
    layout.nodes.forEach((node) => map.set(node.id, node.label));
    return map;
  }, [layout.nodes]);

  return (
    <div className="graph-shell">
      <div className="graph-header">
        <div>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
        <button type="button" onClick={onRefresh} disabled={isLoading}>
          {isLoading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
      <div className="graph-frame" ref={containerRef}>
        {isLoading && <p className="graph-loading">Loading graph…</p>}
        {!isLoading && !layout.nodes.length && <p className="graph-loading">{emptyHint}</p>}
        {!isLoading && layout.nodes.length > 0 && (
          <>
            <svg className="graph-lines" width={dimensions.width} height={dimensions.height}>
              {layout.edges.map((edge) => (
                <line
                  key={edge.id}
                  className="edge similarity"
                  x1={edge.start.x}
                  y1={edge.start.y}
                  x2={edge.end.x}
                  y2={edge.end.y}
                  style={{ strokeWidth: 1 + Math.log(1 + edge.weight) * 2 }}
                />
              ))}
            </svg>
            {layout.nodes.map((node) => (
              <div
                key={node.id}
                className={`graph-node ${node.noteCount ? 'tag' : 'note'}`}
                style={{ left: node.x, top: node.y }}
              >
                <span className="graph-node-label">{node.label}</span>
                {typeof node.noteCount === 'number' && (
                  <span className="graph-node-meta">{node.noteCount} notes</span>
                )}
                {node.tagCount ? <span className="graph-node-meta">{node.tagCount} tags</span> : null}
              </div>
            ))}
          </>
        )}
      </div>
      {highlightEdges.length > 0 && (
        <ul className="graph-context">
          {highlightEdges.map((edge) => (
            <li key={edge.id}>
              <span className="pair">
                {labelMap.get(edge.source) ?? edge.source} ↔ {labelMap.get(edge.target) ?? edge.target}
              </span>
              {edge.sharedTags && edge.sharedTags.length > 0 && (
                <span className="context">Shared tags: {edge.sharedTags.join(', ')}</span>
              )}
              {edge.sharedNotes && edge.sharedNotes.length > 0 && (
                <span className="context">
                  Shared notes:{' '}
                  {edge.sharedNotes.length > 2
                    ? `${edge.sharedNotes.slice(0, 2).join('; ')}…`
                    : edge.sharedNotes.join('; ')}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
