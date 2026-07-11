import { MetricRow } from '../../api';
import { format, strategyNames } from '../../types';

export default function MetricsTable({ rows }: { rows: MetricRow[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Chunks</th>
            <th>R@1</th>
            <th>R@3</th>
            <th>R@5</th>
            <th>MRR</th>
            <th>nDCG@5</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.strategy}-${row.retriever}`} className={row.strategy === 'hsc_rag' ? 'highlight-row' : ''}>
              <td>{strategyNames[row.strategy]}</td>
              <td>{row.chunks}</td>
              <td>{format(row['recall@1'])}</td>
              <td>{format(row['recall@3'])}</td>
              <td>{format(row['recall@5'])}</td>
              <td>{format(row.mrr)}</td>
              <td>{format(row['ndcg@5'])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
