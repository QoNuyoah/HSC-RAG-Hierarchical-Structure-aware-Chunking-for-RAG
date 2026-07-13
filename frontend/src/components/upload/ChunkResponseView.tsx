import { CheckCircle2, Gauge } from 'lucide-react';
import { ChunkAgentResponse } from '../../api';
import ResultMetric from '../ui/ResultMetric';
import ChunkCard from './ChunkCard';

export default function ChunkResponseView({ response }: { response: ChunkAgentResponse }) {
  const report = response.report ?? {};
  const adaptive = response.chunks[0]?.metadata?.adaptive_boundary;
  const qualityCounts = (report.quality_flag_counts ?? {}) as Record<string, number>;
  return (
    <>
      <div className="chunk-output-head">
        <div>
          <p className="eyebrow">RagChunk[] · {response.doc_id}</p>
          <h2>{response.chunk_count} 个分段结果</h2>
        </div>
        <div className="output-status">
          <CheckCircle2 size={18} />
          <span>完成</span>
        </div>
      </div>

      <div className="result-strip">
        <ResultMetric label="Total Tokens" value={String(report.total_tokens ?? '-')} />
        <ResultMetric label="Avg Tokens" value={String(report.avg_tokens ?? '-')} />
        <ResultMetric label="Anchor OK" value={String(qualityCounts.source_anchor_complete ?? 0)} />
        <ResultMetric label="Length OK" value={String(qualityCounts.length_ok ?? 0)} />
      </div>

      {adaptive && (
        <div className="adaptive-summary">
          <div className="section-heading">
            <Gauge size={18} />
            <h2>自适应边界</h2>
          </div>
          <div className="adaptive-grid">
            <ResultMetric label="Profile" value={String(adaptive.profile ?? '-')} />
            <ResultMetric label="Strength" value={String(adaptive.boundary_strength ?? '-')} />
            <ResultMetric label="Context Need" value={String(adaptive.stats?.context_need ?? '-')} />
            <ResultMetric label="Structure Need" value={String(adaptive.stats?.structure_need ?? '-')} />
          </div>
          <p className="adaptive-reason">{String(adaptive.decision_reason ?? '')}</p>
        </div>
      )}



      <div className="chunk-list-output">
        {response.chunks.map((chunk, index) => (
          <ChunkCard key={chunk.chunk_id} chunk={chunk} index={index + 1} />
        ))}
      </div>
    </>
  );
}
