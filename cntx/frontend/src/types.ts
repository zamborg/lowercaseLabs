export interface Note {
  id: string;
  content: string;
  createdAt: string;
  autoTags: string[];
  manualTags: string[];
  tags: string[];
  linkSummary: string;
  score?: number;
}

export interface TagSummary {
  name: string;
  noteCount: number;
}

export interface GraphNode {
  id: string;
  label: string;
  createdAt?: string;
  tagCount?: number;
  noteCount?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  weight: number;
  sharedTags?: string[];
  sharedNotes?: string[];
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphBundle {
  noteGraph: GraphData;
  tagGraph: GraphData;
}
