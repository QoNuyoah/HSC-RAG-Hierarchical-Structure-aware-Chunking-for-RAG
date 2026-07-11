import { ChunkAgentRequest, ChunkAgentResponse, GovernedDocument, Strategy, strategies } from '../api';
import { ChunkParameterState, DEFAULT_CHUNK_PARAMETERS } from '../types';

export function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function isStrategy(value: unknown): value is Strategy {
  return typeof value === 'string' && strategies.includes(value as Strategy);
}

export function isChunkAgentResponse(value: unknown): value is ChunkAgentResponse {
  return isRecord(value) && Array.isArray(value.chunks) && typeof value.chunk_count === 'number';
}

export function numberFromConfig(config: Record<string, unknown>, key: string, fallback: number) {
  const value = config[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function booleanFromConfig(config: Record<string, unknown>, key: string, fallback: boolean) {
  const value = config[key];
  return typeof value === 'boolean' ? value : fallback;
}

export function toPositiveInt(value: number, fallback: number) {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(1, Math.round(value));
}

export function toNonNegativeInt(value: number, fallback: number) {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(0, Math.round(value));
}

export function toBoundedNumber(value: number, fallback: number, min = 0, max = 1) {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, Number(value.toFixed(4))));
}

export function defaultParametersForStrategy(strategy: Strategy): ChunkParameterState {
  const base = { ...DEFAULT_CHUNK_PARAMETERS, strategy };
  if (strategy === 'fixed' || strategy === 'recursive') {
    return {
      ...base,
      min_tokens: 128,
      max_tokens: 512,
      include_title_context: false
    };
  }
  if (strategy === 'semantic') {
    return {
      ...base,
      min_tokens: 160,
      max_tokens: 768,
      include_title_context: false
    };
  }
  return base;
}

export function chunkParametersFromPayload(payload: ChunkAgentRequest): ChunkParameterState {
  const config = isRecord(payload.config) ? payload.config : {};
  const strategy = isStrategy(payload.strategy) ? payload.strategy : DEFAULT_CHUNK_PARAMETERS.strategy;
  const defaults = defaultParametersForStrategy(strategy);
  return {
    ...defaults,
    min_tokens: numberFromConfig(config, 'min_tokens', defaults.min_tokens),
    target_tokens: numberFromConfig(config, 'target_tokens', defaults.target_tokens),
    max_tokens: numberFromConfig(config, 'max_tokens', defaults.max_tokens),
    overlap_tokens: numberFromConfig(config, 'overlap_tokens', defaults.overlap_tokens),
    breakpoint_percentile: numberFromConfig(config, 'breakpoint_percentile', defaults.breakpoint_percentile),
    include_title_context: booleanFromConfig(config, 'include_title_context', defaults.include_title_context),
    protect_blocks: booleanFromConfig(config, 'protect_blocks', defaults.protect_blocks),
    adaptive_boundary: booleanFromConfig(config, 'adaptive_boundary', defaults.adaptive_boundary),
    semantic_boundary_scoring: booleanFromConfig(
      config,
      'semantic_boundary_scoring',
      defaults.semantic_boundary_scoring
    ),
    semantic_boundary_threshold: numberFromConfig(
      config,
      'semantic_boundary_threshold',
      defaults.semantic_boundary_threshold
    ),
    semantic_soft_boundary_threshold: numberFromConfig(
      config,
      'semantic_soft_boundary_threshold',
      defaults.semantic_soft_boundary_threshold
    ),
    semantic_distance_threshold: numberFromConfig(
      config,
      'semantic_distance_threshold',
      defaults.semantic_distance_threshold
    ),
    semantic_window_blocks: numberFromConfig(config, 'semantic_window_blocks', defaults.semantic_window_blocks),
    structure_signal_weight: numberFromConfig(
      config,
      'structure_signal_weight',
      defaults.structure_signal_weight
    ),
    semantic_signal_weight: numberFromConfig(
      config,
      'semantic_signal_weight',
      defaults.semantic_signal_weight
    ),
    length_signal_weight: numberFromConfig(config, 'length_signal_weight', defaults.length_signal_weight),
    include_report: typeof payload.include_report === 'boolean' ? payload.include_report : defaults.include_report
  };
}

export function buildConfigFromParameters(parameters: ChunkParameterState): Record<string, unknown> {
  const minTokens = toPositiveInt(parameters.min_tokens, DEFAULT_CHUNK_PARAMETERS.min_tokens);
  const targetTokens = Math.max(minTokens, toPositiveInt(parameters.target_tokens, DEFAULT_CHUNK_PARAMETERS.target_tokens));
  const maxTokens = Math.max(targetTokens, toPositiveInt(parameters.max_tokens, DEFAULT_CHUNK_PARAMETERS.max_tokens));
  const config: Record<string, unknown> = {
    min_tokens: minTokens,
    target_tokens: targetTokens,
    max_tokens: maxTokens,
    include_title_context: parameters.include_title_context
  };

  if (parameters.strategy === 'fixed' || parameters.strategy === 'recursive') {
    config.overlap_tokens = Math.min(targetTokens - 1, toNonNegativeInt(parameters.overlap_tokens, 0));
  }

  if (parameters.strategy === 'semantic') {
    config.breakpoint_percentile = toBoundedNumber(parameters.breakpoint_percentile, 75, 1, 99);
    config.protect_blocks = parameters.protect_blocks;
  }

  if (parameters.strategy === 'hsc_rag') {
    config.protect_blocks = parameters.protect_blocks;
    config.adaptive_boundary = parameters.adaptive_boundary;
    config.semantic_boundary_scoring = parameters.semantic_boundary_scoring;
    config.semantic_boundary_threshold = toBoundedNumber(parameters.semantic_boundary_threshold, 0.62);
    config.semantic_soft_boundary_threshold = toBoundedNumber(parameters.semantic_soft_boundary_threshold, 0.52);
    config.semantic_distance_threshold = toBoundedNumber(parameters.semantic_distance_threshold, 0.72);
    config.semantic_window_blocks = toPositiveInt(parameters.semantic_window_blocks, 3);
    config.structure_signal_weight = toBoundedNumber(parameters.structure_signal_weight, 0.45);
    config.semantic_signal_weight = toBoundedNumber(parameters.semantic_signal_weight, 0.35);
    config.length_signal_weight = toBoundedNumber(parameters.length_signal_weight, 0.2);
  }

  return config;
}

export function applyChunkParameters(payload: ChunkAgentRequest, parameters: ChunkParameterState): ChunkAgentRequest {
  return {
    ...payload,
    strategy: parameters.strategy,
    config: buildConfigFromParameters(parameters),
    include_report: parameters.include_report
  };
}

export function isGovernedDocument(value: unknown): value is GovernedDocument {
  return isRecord(value)
    && typeof value.doc_id === 'string'
    && typeof value.dataset === 'string'
    && typeof value.split === 'string'
    && typeof value.source_doc_id === 'string'
    && typeof value.title === 'string'
    && typeof value.normalization_status === 'string'
    && Array.isArray(value.blocks);
}

export function normalizeChunkRequest(value: unknown): ChunkAgentRequest {
  if (isRecord(value) && isGovernedDocument(value.document)) {
    return {
      document: value.document,
      strategy: (value.strategy as Strategy | undefined) ?? 'hsc_rag',
      config: isRecord(value.config) ? value.config : {},
      include_report: typeof value.include_report === 'boolean' ? value.include_report : true
    };
  }
  if (isGovernedDocument(value)) {
    return {
      document: value,
      strategy: 'hsc_rag',
      config: {},
      include_report: true
    };
  }
  throw new Error('JSON 必须是 GovernedDocument，或包含 document 字段的 ChunkAgentRequest。');
}
