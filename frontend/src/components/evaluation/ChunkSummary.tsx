import { ChunkReport, Strategy, strategies } from '../../api';
import { strategyNames } from '../../types';

export default function ChunkSummary({ reports }: { reports?: Partial<Record<Strategy, ChunkReport>> }) {
  if (!reports) return null;
  return (
    <div className="chunk-grid">
      {strategies.map((strategy) => {
        const report = reports[strategy];
        if (!report) return null;
        return (
          <div className="chunk-stat" key={strategy}>
            <span>{strategyNames[strategy]}</span>
            <strong>{report.chunks}</strong>
            <small>
              title ok {report.quality_flag_counts.title_path_consistent ?? 0} · mixed {report.quality_flag_counts.mixed_title_paths ?? 0}
            </small>
          </div>
        );
      })}
    </div>
  );
}
