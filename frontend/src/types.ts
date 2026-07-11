import { RagChunk, Strategy } from './api';

export type LlmEnrichmentState = {
  enabled: boolean;
  mode: 'stable' | 'autonomous';
  preferredTool: 'inspect_hsc_rag_context' | 'chunk_current_document' | 'chunk_and_enrich_current_document';
  instruction: string;
  includeQa: boolean;
  model: string;
  baseUrl: string;
  apiKeyEnv: string;
  timeoutSeconds: number;
  useResponseFormat: boolean;
};

export const DEFAULT_LLM_ENRICHMENT: LlmEnrichmentState = {
  enabled: false,
  mode: 'stable',
  preferredTool: 'chunk_and_enrich_current_document',
  instruction: '对当前文档进行分段，并生成摘要、主题标签、实体标签和语义完整性评分。',
  includeQa: false,
  model: 'Qwen/Qwen3-8B',
  baseUrl: 'https://api.siliconflow.cn/v1',
  apiKeyEnv: 'SILICONFLOW_API_KEY',
  timeoutSeconds: 180,
  useResponseFormat: true
};

export const strategyNames: Record<Strategy, string> = {
  fixed: 'Fixed',
  recursive: 'Recursive',
  semantic: 'Semantic',
  hsc_rag: 'HSC-RAG'
};

export const caseLabels: Record<string, string> = {
  hsc_missing: 'HSC 缺失',
  baseline_beats_hsc: '基线领先',
  strategy_disagreement: '策略分歧',
  hsc_ok: 'HSC 命中',
  missing_hsc: '缺少 HSC'
};

export type ChunkParameterState = {
  strategy: Strategy;
  min_tokens: number;
  target_tokens: number;
  max_tokens: number;
  overlap_tokens: number;
  breakpoint_percentile: number;
  include_title_context: boolean;
  protect_blocks: boolean;
  adaptive_boundary: boolean;
  semantic_boundary_scoring: boolean;
  semantic_boundary_threshold: number;
  semantic_soft_boundary_threshold: number;
  semantic_distance_threshold: number;
  semantic_window_blocks: number;
  structure_signal_weight: number;
  semantic_signal_weight: number;
  length_signal_weight: number;
  include_report: boolean;
};

export const DEFAULT_CHUNK_PARAMETERS: ChunkParameterState = {
  strategy: 'hsc_rag',
  min_tokens: 180,
  target_tokens: 512,
  max_tokens: 900,
  overlap_tokens: 64,
  breakpoint_percentile: 75,
  include_title_context: true,
  protect_blocks: true,
  adaptive_boundary: true,
  semantic_boundary_scoring: true,
  semantic_boundary_threshold: 0.62,
  semantic_soft_boundary_threshold: 0.52,
  semantic_distance_threshold: 0.72,
  semantic_window_blocks: 3,
  structure_signal_weight: 0.45,
  semantic_signal_weight: 0.35,
  length_signal_weight: 0.2,
  include_report: true
};

export type UploadTokenizerProfile = 'auto' | 'mixed' | 'english' | 'cjk_2_4gram';

export type UploadEvaluationHit = {
  rank: number;
  chunk: RagChunk;
  score: number;
  relevant: boolean;
  coveredGoldBlockIds: string[];
};

export type UploadEvaluationResult = {
  hits: UploadEvaluationHit[];
  recallAtK: number | null;
  hitAtK: number | null;
  mrr: number | null;
  ndcgAtK: number | null;
  coveredGoldBlockIds: string[];
  goldBlockIds: string[];
  tokenizerProfile: UploadTokenizerProfile;
};

export function format(value: number | undefined) {
  return typeof value === 'number' ? value.toFixed(3) : '-';
}

export function caseLabel(type: string) {
  return caseLabels[type] || type;
}

export function stableToolInstruction(tool: LlmEnrichmentState['preferredTool']) {
  if (tool === 'inspect_hsc_rag_context') return '请查看当前文档结构和可用能力，不执行分段。';
  if (tool === 'chunk_current_document') return '请使用当前配置对文档进行结构感知分段。';
  return '请对当前文档进行分段，并生成摘要、主题标签、实体标签和语义完整性评分。';
}

export function agentInstruction(settings: LlmEnrichmentState) {
  const base = settings.mode === 'stable'
    ? stableToolInstruction(settings.preferredTool)
    : settings.instruction.trim();
  if (settings.mode !== 'stable' || !settings.includeQa) return base;
  return `${base} 同时为每个 chunk 生成一个严格基于原文、可追溯的 QA 问答样本。`;
}
