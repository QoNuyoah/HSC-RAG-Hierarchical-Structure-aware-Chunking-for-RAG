import { RagChunk, Retriever } from '../api';
import { UploadEvaluationResult, UploadTokenizerProfile } from '../types';

const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'how', 'in',
  'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'what', 'when',
  'where', 'which', 'who', 'why', 'with'
]);

function hasCjk(text: string) {
  return /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]/.test(text);
}

function cjkNgrams(text: string, minN = 2, maxN = 4) {
  const tokens: string[] = [];
  for (let n = minN; n <= Math.min(maxN, text.length); n += 1) {
    for (let index = 0; index <= text.length - n; index += 1) {
      tokens.push(text.slice(index, index + n));
    }
  }
  return tokens;
}

function tokenizeLocal(text: string, profile: UploadTokenizerProfile): string[] {
  const effectiveProfile = profile === 'auto' ? (hasCjk(text) ? 'cjk_2_4gram' : 'mixed') : profile;
  const value = text || '';
  if (effectiveProfile === 'cjk_2_4gram') {
    const latin = value.match(/[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?/g) ?? [];
    const cjkSpans = value.match(/[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+/g) ?? [];
    return [
      ...latin.map((token) => token.toLowerCase()),
      ...cjkSpans.flatMap((span) => cjkNgrams(span))
    ].filter((token) => token.length > 1 && !STOPWORDS.has(token));
  }

  const tokens = value.match(/[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]/g) ?? [];
  return tokens
    .map((token) => token.toLowerCase())
    .filter((token) => token.length > 1 && !STOPWORDS.has(token));
}

function chunkSearchText(chunk: RagChunk) {
  return [
    chunk.text,
    ...(chunk.title_path ?? []),
    ...(chunk.tags ?? []),
    chunk.summary ?? ''
  ].join(' ');
}

function termCounts(tokens: string[]) {
  const counts = new Map<string, number>();
  tokens.forEach((token) => counts.set(token, (counts.get(token) ?? 0) + 1));
  return counts;
}

function cosineScore(queryTokens: string[], docTokens: string[]) {
  if (!queryTokens.length || !docTokens.length) return 0;
  const query = termCounts(queryTokens);
  const doc = termCounts(docTokens);
  let dot = 0;
  let queryNorm = 0;
  let docNorm = 0;
  query.forEach((value, token) => {
    dot += value * (doc.get(token) ?? 0);
    queryNorm += value * value;
  });
  doc.forEach((value) => {
    docNorm += value * value;
  });
  if (!queryNorm || !docNorm) return 0;
  return dot / Math.sqrt(queryNorm * docNorm);
}

function bm25Scores(queryTokens: string[], documents: string[][]) {
  const k1 = 1.5;
  const b = 0.75;
  const docCount = documents.length || 1;
  const docLengths = documents.map((tokens) => tokens.length || 1);
  const avgDl = docLengths.reduce((sum, value) => sum + value, 0) / docCount;
  const documentFrequency = new Map<string, number>();
  documents.forEach((tokens) => {
    new Set(tokens).forEach((token) => documentFrequency.set(token, (documentFrequency.get(token) ?? 0) + 1));
  });
  return documents.map((tokens, index) => {
    const freqs = termCounts(tokens);
    const lengthNorm = 1 - b + b * (docLengths[index] / Math.max(1, avgDl));
    return queryTokens.reduce((score, token) => {
      const tf = freqs.get(token) ?? 0;
      if (!tf) return score;
      const df = documentFrequency.get(token) ?? 0;
      const idf = Math.log(1 + (docCount - df + 0.5) / (df + 0.5));
      return score + idf * ((tf * (k1 + 1)) / (tf + k1 * lengthNorm));
    }, 0);
  });
}

function normalizeScores(scores: number[]) {
  const max = Math.max(...scores, 0);
  if (!max) return scores.map(() => 0);
  return scores.map((score) => score / max);
}

function parseGoldBlocks(value: string) {
  return Array.from(new Set(value.split(/[\s,，;；]+/).map((item) => item.trim()).filter(Boolean)));
}

function dcg(relevances: number[]) {
  return relevances.reduce((sum, rel, index) => sum + rel / Math.log2(index + 2), 0);
}

export function evaluateUploadedChunks(
  chunks: RagChunk[],
  query: string,
  goldBlockInput: string,
  retriever: Retriever,
  tokenizerProfile: UploadTokenizerProfile,
  topK: number
): UploadEvaluationResult | null {
  const trimmedQuery = query.trim();
  if (!trimmedQuery || !chunks.length) return null;
  const effectiveProfile = tokenizerProfile === 'auto'
    ? (hasCjk(`${trimmedQuery} ${chunks.slice(0, 8).map((chunk) => chunk.text).join(' ')}`) ? 'cjk_2_4gram' : 'mixed')
    : tokenizerProfile;
  const docs = chunks.map((chunk) => tokenizeLocal(chunkSearchText(chunk), effectiveProfile));
  const queryTokens = tokenizeLocal(trimmedQuery, effectiveProfile);
  const bm25 = bm25Scores(queryTokens, docs);
  const dense = docs.map((tokens) => cosineScore(queryTokens, tokens));
  const bm25Norm = normalizeScores(bm25);
  const denseNorm = normalizeScores(dense);
  const scores = chunks.map((_chunk, index) => {
    if (retriever === 'bm25') return bm25[index];
    if (retriever === 'dense') return dense[index];
    return 0.55 * bm25Norm[index] + 0.45 * denseNorm[index];
  });
  const goldBlockIds = parseGoldBlocks(goldBlockInput);
  const gold = new Set(goldBlockIds);
  const ranked = chunks
    .map((chunk, index) => ({ chunk, score: scores[index] }))
    .sort((left, right) => right.score - left.score)
    .slice(0, topK)
    .map((item, index) => {
      const covered = item.chunk.source_blocks.filter((blockId) => gold.has(blockId));
      return {
        rank: index + 1,
        chunk: item.chunk,
        score: item.score,
        relevant: covered.length > 0,
        coveredGoldBlockIds: covered
      };
    });
  const coveredGoldBlockIds = Array.from(new Set(ranked.flatMap((hit) => hit.coveredGoldBlockIds)));
  const relevances = ranked.map((hit) => (hit.relevant ? 1 : 0));
  const ideal = Array.from({ length: Math.min(topK, goldBlockIds.length) }, () => 1);
  const firstRelevant = ranked.find((hit) => hit.relevant);
  return {
    hits: ranked,
    recallAtK: goldBlockIds.length ? coveredGoldBlockIds.length / goldBlockIds.length : null,
    hitAtK: goldBlockIds.length ? (coveredGoldBlockIds.length > 0 ? 1 : 0) : null,
    mrr: goldBlockIds.length && firstRelevant ? 1 / firstRelevant.rank : goldBlockIds.length ? 0 : null,
    ndcgAtK: goldBlockIds.length ? dcg(relevances) / Math.max(1, dcg(ideal)) : null,
    coveredGoldBlockIds,
    goldBlockIds,
    tokenizerProfile: effectiveProfile
  };
}
