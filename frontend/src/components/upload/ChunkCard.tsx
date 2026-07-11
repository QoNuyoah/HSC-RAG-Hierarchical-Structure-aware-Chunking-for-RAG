import { FileJson, Gauge, ListTree, Tags } from 'lucide-react';
import { RagChunk } from '../../api';
import { isRecord } from '../../utils/chunkParams';

export default function ChunkCard({ chunk, index }: { chunk: RagChunk; index: number }) {
  const decision = chunk.metadata?.closing_boundary_decision;
  const score = typeof decision?.boundary_score === 'number' ? decision.boundary_score.toFixed(4) : '-';
  const reason = typeof decision?.split_reason === 'string'
    ? decision.split_reason
    : chunk.quality_flags.includes('final_flush') ? 'final_flush' : '-';
  const title = chunk.title_path.length ? chunk.title_path.join(' > ') : 'Untitled';
  const enrichment = isRecord(chunk.metadata?.llm_enrichment) ? chunk.metadata.llm_enrichment : null;
  const execution = enrichment && typeof enrichment.provider_execution === 'string'
    ? enrichment.provider_execution
    : null;
  return (
    <article className="chunk-card">
      <header className="chunk-card-head">
        <div>
          <h3>Chunk {String(index).padStart(2, '0')}</h3>
          <p>{title}</p>
        </div>
        <span className="token-badge">{chunk.token_count} tokens</span>
      </header>

      {chunk.summary && <p className="chunk-summary-text">{chunk.summary}</p>}

      {enrichment && (
        <section className="llm-result">
          <div className="llm-result-head">
            <strong>LLM 语义增强</strong>
            <code className={execution === 'remote_llm_call' ? 'llm-success' : 'llm-fallback'}>
              {execution ?? 'unknown'}
            </code>
          </div>
          {typeof enrichment.summary === 'string' && <p>{enrichment.summary}</p>}
          <div className="llm-score-row">
            <span>完整性 {String(enrichment.semantic_integrity_score ?? '-')}</span>
            <span>忠实度 {String(enrichment.summary_faithfulness_score ?? '-')}</span>
            <span>标签 {String(enrichment.tag_accuracy_score ?? '-')}</span>
          </div>
          {Array.isArray(enrichment.topic_tags) && (
            <div className="tag-row">
              <Tags size={15} />
              {enrichment.topic_tags.map((tag: unknown) => <span key={String(tag)}>{String(tag)}</span>)}
            </div>
          )}
          {typeof enrichment.provider_error === 'string' && (
            <p className="llm-error">{enrichment.provider_error}</p>
          )}
          {Array.isArray(enrichment.qa_pairs) && enrichment.qa_pairs.length > 0 && (
            <div className="qa-result-list">
              <strong>QA 问答样本</strong>
              {enrichment.qa_pairs.map((pair: unknown, qaIndex: number) => {
                if (!isRecord(pair)) return null;
                return (
                  <article className="qa-result-item" key={`${chunk.chunk_id}-qa-${qaIndex}`}>
                    <p><strong>Q：</strong>{String(pair.question ?? '-')}</p>
                    <p><strong>A：</strong>{String(pair.answer ?? '-')}</p>
                    <div className="qa-result-meta">
                      <span>{String(pair.answerability ?? 'unknown')}</span>
                      <span>忠实度 {String(pair.faithfulness_score ?? '-')}</span>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>
      )}

      <div className="chunk-text-preview">{chunk.text}</div>

      <div className="chunk-insight-grid">
        <div>
          <Gauge size={15} />
          <span>boundary</span>
          <strong>{score}</strong>
        </div>
        <div>
          <ListTree size={15} />
          <span>reason</span>
          <strong>{reason}</strong>
        </div>
        <div>
          <FileJson size={15} />
          <span>source</span>
          <strong>{chunk.source_anchor.block_count}</strong>
        </div>
      </div>

      <div className="tag-row">
        <Tags size={15} />
        {chunk.tags.slice(0, 8).map((tag) => <span key={tag}>{tag}</span>)}
      </div>

      <div className="flag-row">
        {chunk.quality_flags.map((flag) => <code key={flag}>{flag}</code>)}
      </div>

      <div className="source-row">
        {chunk.source_blocks.slice(0, 8).map((blockId) => <code key={blockId}>{blockId}</code>)}
        {chunk.source_blocks.length > 8 && <code>+{chunk.source_blocks.length - 8}</code>}
      </div>
    </article>
  );
}
