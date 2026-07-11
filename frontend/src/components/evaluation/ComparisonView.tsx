import { AlertTriangle, CheckCircle2, CircleDot } from 'lucide-react';
import { QueryComparison, strategies } from '../../api';
import { caseLabel, format, strategyNames } from '../../types';

export default function ComparisonView({ comparison }: { comparison: QueryComparison }) {
  return (
    <>
      <div className="comparison-header">
        <div>
          <p className="eyebrow">{comparison.retriever.toUpperCase()} · 当前 Query Top-5 对比 · {caseLabel(comparison.case_type)}</p>
          <h2>{comparison.question}</h2>
          <p className="doc-line">{comparison.doc_id}</p>
          <p className="scope-note">右侧四列是当前选中问题的逐 query 指标；全局平均指标请看左侧 Retrieval Metrics 表格。</p>
        </div>
        <div className="gold-box">
          <span>Gold Blocks</span>
          <strong>{comparison.gold_block_ids.length}</strong>
        </div>
      </div>

      <div className="gold-list">
        {comparison.gold_block_ids.map((blockId) => (
          <code key={blockId}>{blockId}</code>
        ))}
      </div>

      <div className="strategy-grid">
        {strategies.map((strategy) => {
          const data = comparison.strategies[strategy];
          return (
            <section className={`strategy-card ${strategy}`} key={strategy}>
              <div className="strategy-head">
                <div>
                  <h3>{strategyNames[strategy]}</h3>
                  <p>当前问题 R@5 {format(data?.recall_by_k?.['5'])} · RR {format(data?.reciprocal_rank)}</p>
                </div>
                {data?.missing_gold_blocks_at_max_k?.length ? (
                  <AlertTriangle className="status-icon miss" size={18} />
                ) : (
                  <CheckCircle2 className="status-icon ok" size={18} />
                )}
              </div>

              <div className="hit-list">
                {data?.top_hits.map((hit) => (
                  <article className={`hit ${hit.is_relevant ? 'relevant' : ''}`} key={`${strategy}-${hit.rank}-${hit.chunk_id}`}>
                    <div className="hit-title">
                      <span className="rank">#{hit.rank}</span>
                      <span>{hit.title_path.length ? hit.title_path.join(' > ') : 'Untitled'}</span>
                      {hit.is_relevant && <CircleDot size={14} />}
                    </div>
                    <p>{hit.preview}</p>
                    <div className="hit-foot">
                      <span>score {format(hit.score)}</span>
                      <span>{hit.token_count} tokens</span>
                      <span>{hit.covered_gold_block_ids.length} covered</span>
                    </div>
                    {hit.covered_gold_block_ids.length > 0 && (
                      <div className="covered">
                        {hit.covered_gold_block_ids.map((blockId) => <code key={blockId}>{blockId}</code>)}
                      </div>
                    )}
                  </article>
                ))}
              </div>
            </section>
          );
        })}
      </div>
    </>
  );
}
