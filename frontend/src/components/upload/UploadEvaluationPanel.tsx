import { useMemo, useState } from 'react';
import { CircleDot, Search } from 'lucide-react';
import { ChunkAgentResponse, Retriever } from '../../api';
import { UploadTokenizerProfile, format } from '../../types';
import { evaluateUploadedChunks } from '../../utils/localRetrieval';
import ResultMetric from '../ui/ResultMetric';

export default function UploadEvaluationPanel({ response }: { response: ChunkAgentResponse }) {
  const [query, setQuery] = useState('');
  const [goldBlocks, setGoldBlocks] = useState('');
  const [retriever, setRetriever] = useState<Retriever>('bm25');
  const [tokenizerProfile, setTokenizerProfile] = useState<UploadTokenizerProfile>('auto');
  const [topK, setTopK] = useState(5);
  const result = useMemo(
    () => evaluateUploadedChunks(response.chunks, query, goldBlocks, retriever, tokenizerProfile, topK),
    [response.chunks, query, goldBlocks, retriever, tokenizerProfile, topK]
  );
  const queryCount = response.chunks.reduce((sum, chunk) => sum + chunk.source_blocks.length, 0);

  return (
    <section className="upload-eval-panel">
      <div className="upload-eval-head">
        <div className="section-heading">
          <Search size={18} />
          <h2>当前分段评估</h2>
        </div>
        <span>{response.chunks.length} chunks · {queryCount} source blocks</span>
      </div>

      <div className="upload-eval-controls">
        <label className="field-block eval-query-field">
          <span>Query</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="输入要检索的问题或关键词"
          />
        </label>
        <label className="field-block">
          <span>Retriever</span>
          <select value={retriever} onChange={(event) => setRetriever(event.target.value as Retriever)}>
            <option value="bm25">BM25</option>
            <option value="dense">Dense-lite</option>
            <option value="hybrid">Hybrid</option>
          </select>
        </label>
        <label className="field-block">
          <span>Profile</span>
          <select
            value={tokenizerProfile}
            onChange={(event) => setTokenizerProfile(event.target.value as UploadTokenizerProfile)}
          >
            <option value="auto">Auto</option>
            <option value="mixed">Mixed</option>
            <option value="english">English</option>
            <option value="cjk_2_4gram">中文 2-4gram</option>
          </select>
        </label>
        <label className="field-block">
          <span>Top K</span>
          <input
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(event) => setTopK(Math.min(20, Math.max(1, Number(event.target.value) || 5)))}
          />
        </label>
        <label className="field-block eval-gold-field">
          <span>Gold Blocks</span>
          <input
            value={goldBlocks}
            onChange={(event) => setGoldBlocks(event.target.value)}
            placeholder="可选：block_id，用逗号或空格分隔"
          />
        </label>
      </div>

      {result ? (
        <>
          <div className="result-strip upload-eval-strip">
            <ResultMetric label={`Recall@${topK}`} value={result.recallAtK === null ? '-' : format(result.recallAtK)} />
            <ResultMetric label={`Hit@${topK}`} value={result.hitAtK === null ? '-' : format(result.hitAtK)} />
            <ResultMetric label="MRR" value={result.mrr === null ? '-' : format(result.mrr)} />
            <ResultMetric label={`nDCG@${topK}`} value={result.ndcgAtK === null ? '-' : format(result.ndcgAtK)} />
            <ResultMetric label="Profile" value={result.tokenizerProfile} />
          </div>
          <div className="upload-eval-hits">
            {result.hits.map((hit) => (
              <article className={`upload-eval-hit ${hit.relevant ? 'relevant' : ''}`} key={`${hit.rank}-${hit.chunk.chunk_id}`}>
                <div className="hit-title">
                  <span className="rank">#{hit.rank}</span>
                  <span>{hit.chunk.title_path.length ? hit.chunk.title_path.join(' > ') : hit.chunk.chunk_id}</span>
                  {hit.relevant && <CircleDot size={14} />}
                </div>
                <p>{hit.chunk.text.slice(0, 360)}</p>
                <div className="hit-foot">
                  <span>score {format(hit.score)}</span>
                  <span>{hit.chunk.token_count} tokens</span>
                  <span>{hit.chunk.source_blocks.length} blocks</span>
                </div>
                <div className="source-row">
                  {hit.chunk.source_blocks.slice(0, 8).map((blockId) => (
                    <code className={result.goldBlockIds.includes(blockId) ? 'gold-source' : ''} key={blockId}>{blockId}</code>
                  ))}
                  {hit.chunk.source_blocks.length > 8 && <code>+{hit.chunk.source_blocks.length - 8}</code>}
                </div>
              </article>
            ))}
          </div>
        </>
      ) : (
        <div className="empty-state upload-eval-empty">等待输入 query</div>
      )}
    </section>
  );
}
