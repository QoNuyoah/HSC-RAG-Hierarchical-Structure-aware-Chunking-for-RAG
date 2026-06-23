export type Strategy = 'fixed' | 'recursive' | 'semantic' | 'hsc_rag';
export type Retriever = 'bm25' | 'dense' | 'hybrid';

export const strategies: Strategy[] = ['fixed', 'recursive', 'semantic', 'hsc_rag'];
export const retrievers: Retriever[] = ['bm25', 'dense', 'hybrid'];

const API_BASE = import.meta.env.VITE_API_BASE || '';

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
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

