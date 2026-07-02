export type Strategy = 'fixed' | 'recursive' | 'semantic' | 'hsc_rag';
export type Retriever = 'bm25' | 'dense' | 'hybrid';

export const strategies: Strategy[] = ['fixed', 'recursive', 'semantic', 'hsc_rag'];
export const retrievers: Retriever[] = ['bm25', 'dense', 'hybrid'];

const API_BASE = import.meta.env.VITE_API_BASE || '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = typeof payload?.detail === 'string' ? payload.detail : JSON.stringify(payload?.detail ?? payload);
    } catch {
      // Keep the HTTP status text if the server did not return JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export interface SourceAnchor {
  dataset: string;
  split?: string | null;
  source_doc_id: string;
  section_name?: string | null;
  paragraph_index?: number | null;
  asset_file?: string | null;
  extra?: Record<string, unknown>;
}

export interface GovernedBlock {
  block_id: string;
  doc_id: string;
  type: string;
  text: string;
  order: number;
  level?: number;
  title_path?: string[];
  source_anchor: SourceAnchor;
  parent_heading_id?: string | null;
  entity_tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface GovernedDocument {
  doc_id: string;
  dataset: string;
  split: string;
  source_doc_id: string;
  title: string;
  abstract?: string | null;
  normalization_status: string;
  term_policy?: string;
  governance_stage?: string;
  schema_version?: string;
  blocks: GovernedBlock[];
  queries?: unknown[];
  source_ref?: Record<string, unknown>;
  conversion_warnings?: string[];
  metadata?: Record<string, unknown>;
  created_at?: string;
}

export interface ChunkAgentRequest {
  document: GovernedDocument;
  strategy?: Strategy;
  config?: Record<string, unknown>;
  include_report?: boolean;
}

export interface ChunkSourceAnchor {
  dataset: string;
  split?: string | null;
  source_doc_id: string;
  sections: string[];
  first_block_id?: string | null;
  last_block_id?: string | null;
  block_count: number;
  assets: string[];
}

export interface RagChunk {
  chunk_id: string;
  doc_id: string;
  dataset: string;
  split: string;
  strategy: Strategy;
  text: string;
  token_count: number;
  title_path: string[];
  source_blocks: string[];
  source_anchor: ChunkSourceAnchor;
  tags: string[];
  summary: string | null;
  entity_tags: string[];
  quality_flags: string[];
  metadata: Record<string, any>;
}

export interface ChunkAgentResponse {
  agent: string;
  strategy: Strategy;
  doc_id: string;
  chunks: RagChunk[];
  chunk_count: number;
  report: Record<string, any>;
}

export interface MetricRow {
  strategy: Strategy;
  retriever: Retriever;
  chunks: number;
  queries_evaluated: number;
  search_scope: string;
  index_fields: string[];
  'recall@1': number;
  'recall@3': number;
  'recall@5': number;
  mrr: number;
  'ndcg@5': number;
  'hit_rate@5': number;
  'full_recall_rate@5': number;
}

export interface ChunkReport {
  strategy: Strategy;
  chunks: number;
  avg_tokens: number;
  min_tokens: number;
  max_tokens: number;
  quality_flag_counts: Record<string, number>;
}

export interface Overview {
  project: {
    name: string;
    title: string;
    governance_stage: string;
    dataset: string;
    split: string;
  };
  conversion: Record<string, number | string>;
  strategies: Strategy[];
  retrievers: Retriever[];
  chunk_reports: Record<Strategy, ChunkReport>;
  best_by_metric: Record<string, Array<{ strategy: Strategy; retriever: Retriever; value: number }>>;
}

export interface QuerySummary {
  query_id: string;
  doc_id: string;
  question: string;
  gold_block_count: number;
  case_type: string;
  strategies: Partial<Record<Strategy, {
    first_relevant_rank: number;
    reciprocal_rank: number;
    'recall@5': number;
    'hit@5': number;
    missing_gold_count: number;
    relevant_hits_top5: number;
  }>>;
}

export interface Hit {
  rank: number;
  chunk_id: string;
  score: number;
  is_relevant: boolean;
  covered_gold_block_ids: string[];
  title_path: string[];
  source_blocks: string[];
  token_count: number;
  quality_flags: string[];
  preview: string;
}

export interface StrategyComparison {
  first_relevant_rank: number;
  reciprocal_rank: number;
  recall_by_k: Record<string, number>;
  hit_by_k: Record<string, number>;
  full_recall_by_k: Record<string, number>;
  covered_gold_blocks_by_k: Record<string, string[]>;
  missing_gold_blocks_at_max_k: string[];
  'ndcg@5': number;
  top_hits: Hit[];
}

export interface QueryComparison {
  found: boolean;
  retriever: Retriever;
  query_id: string;
  doc_id: string;
  question: string;
  gold_block_ids: string[];
  case_type: string;
  strategies: Partial<Record<Strategy, StrategyComparison>>;
}

export function getOverview() {
  return request<Overview>('/api/overview');
}

export function getMetrics(retriever: Retriever) {
  return request<{ retriever: Retriever; rows: MetricRow[] }>(`/api/metrics?retriever=${retriever}`);
}

export function getBadCases(retriever: Retriever) {
  return request<{ retriever: Retriever; queries: QuerySummary[] }>(`/api/bad-cases?retriever=${retriever}`);
}

export function getQueries(retriever: Retriever) {
  return request<{ retriever: Retriever; queries: QuerySummary[] }>(`/api/queries?retriever=${retriever}`);
}

export function getComparison(queryId: string, retriever: Retriever) {
  return request<QueryComparison>(`/api/queries/${encodeURIComponent(queryId)}/comparison?retriever=${retriever}`);
}

export function postChunk(payload: ChunkAgentRequest) {
  return request<ChunkAgentResponse>('/api/v1/chunk', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json; charset=utf-8'
    },
    body: JSON.stringify(payload)
  });
}
