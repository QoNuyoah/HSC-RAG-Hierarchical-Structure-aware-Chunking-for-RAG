import { AlertTriangle, BarChart3, CheckCircle2, CircleDot, RefreshCw, Search } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import {
  ChunkReport,
  MetricRow,
  Overview,
  QueryComparison,
  QuerySummary,
  Retriever,
  Strategy,
  getBadCases,
  getComparison,
  getMetrics,
  getOverview,
  getQueries,
  retrievers,
  strategies
} from './api';

const strategyNames: Record<Strategy, string> = {
  fixed: 'Fixed',
  recursive: 'Recursive',
  semantic: 'Semantic',
  hsc_rag: 'HSC-RAG'
};

const caseLabels: Record<string, string> = {
  hsc_missing: 'HSC 缺失',
  baseline_beats_hsc: '基线领先',
  strategy_disagreement: '策略分歧',
  hsc_ok: 'HSC 命中',
  missing_hsc: '缺少 HSC'
};

function format(value: number | undefined) {
  return typeof value === 'number' ? value.toFixed(3) : '-';
}

function caseLabel(type: string) {
  return caseLabels[type] || type;
}

function App() {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [queries, setQueries] = useState<QuerySummary[]>([]);
  const [selectedRetriever, setSelectedRetriever] = useState<Retriever>('bm25');
  const [queryMode, setQueryMode] = useState<'bad' | 'all'>('bad');
  const [selectedQueryId, setSelectedQueryId] = useState('');
  const [comparison, setComparison] = useState<QueryComparison | null>(null);
  const [searchText, setSearchText] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    void loadBase();
  }, []);

  useEffect(() => {
    void loadRetrieverData(selectedRetriever, queryMode);
  }, [selectedRetriever, queryMode]);

  useEffect(() => {
    if (!selectedQueryId) return;
    void loadComparison(selectedQueryId, selectedRetriever);
  }, [selectedQueryId, selectedRetriever]);

  async function loadBase() {
    try {
      setError('');
      const data = await getOverview();
      setOverview(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'API 请求失败');
    }
  }

  async function loadRetrieverData(retriever: Retriever, mode: 'bad' | 'all') {
    try {
      setLoading(true);
      setError('');
      const [metricData, queryData] = await Promise.all([
        getMetrics(retriever),
        mode === 'bad' ? getBadCases(retriever) : getQueries(retriever)
      ]);
      setMetrics(metricData.rows);
      setQueries(queryData.queries);
      setSelectedQueryId((current) => {
        if (current && queryData.queries.some((item) => item.query_id === current)) return current;
        return queryData.queries[0]?.query_id || '';
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'API 请求失败');
    } finally {
      setLoading(false);
    }
  }

  async function loadComparison(queryId: string, retriever: Retriever) {
    try {
      setError('');
      setComparison(await getComparison(queryId, retriever));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'API 请求失败');
      setComparison(null);
    }
  }

  const filteredQueries = useMemo(() => {
    const text = searchText.trim().toLowerCase();
    if (!text) return queries;
    return queries.filter((item) =>
      `${item.question} ${item.doc_id} ${item.query_id}`.toLowerCase().includes(text)
    );
  }, [queries, searchText]);

  const hscMetric = metrics.find((row) => row.strategy === 'hsc_rag');

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">QASPER · Post-normalization packaging</p>
          <h1>HSC-RAG Evaluation Console</h1>
        </div>
        <div className="toolbar">
          <select value={selectedRetriever} onChange={(event) => setSelectedRetriever(event.target.value as Retriever)}>
            {retrievers.map((retriever) => (
              <option key={retriever} value={retriever}>{retriever.toUpperCase()}</option>
            ))}
          </select>
          <button type="button" onClick={() => void loadRetrieverData(selectedRetriever, queryMode)} title="刷新当前评估数据">
            <RefreshCw size={16} />
            刷新
          </button>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="metric-strip">
        <Stat label="Answerable Queries" value={String(overview?.conversion?.answerable_queries ?? 28)} />
        <Stat label="HSC Recall@1" value={format(hscMetric?.['recall@1'])} />
        <Stat label="HSC Recall@5" value={format(hscMetric?.['recall@5'])} />
        <Stat label="HSC nDCG@5" value={format(hscMetric?.['ndcg@5'])} />
      </section>

      <main className="layout">
        <section className="left-panel">
          <div className="section-heading">
            <BarChart3 size={18} />
            <h2>Retrieval Metrics</h2>
          </div>
          <MetricsTable rows={metrics} />
          <ChunkSummary reports={overview?.chunk_reports} />

          <div className="query-tools">
            <div className="segmented">
              <button className={queryMode === 'bad' ? 'active' : ''} onClick={() => setQueryMode('bad')}>Bad Cases</button>
              <button className={queryMode === 'all' ? 'active' : ''} onClick={() => setQueryMode('all')}>All Queries</button>
            </div>
            <label className="search-box">
              <Search size={16} />
              <input
                value={searchText}
                onChange={(event) => setSearchText(event.target.value)}
                placeholder="Search query/doc id"
              />
            </label>
          </div>

          <div className="query-list">
            {loading && <div className="empty-state">Loading...</div>}
            {!loading && filteredQueries.map((item) => (
              <button
                type="button"
                className={`query-row ${selectedQueryId === item.query_id ? 'selected' : ''}`}
                key={item.query_id}
                onClick={() => setSelectedQueryId(item.query_id)}
              >
                <span className={`case-pill ${item.case_type}`}>{caseLabel(item.case_type)}</span>
                <span className="query-text">{item.question}</span>
                <span className="query-meta">{item.gold_block_count} gold · {item.doc_id}</span>
              </button>
            ))}
            {!loading && filteredQueries.length === 0 && <div className="empty-state">No query</div>}
          </div>
        </section>

        <section className="comparison-panel">
          {comparison ? <ComparisonView comparison={comparison} /> : <div className="empty-state">Select a query</div>}
        </section>
      </main>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricsTable({ rows }: { rows: MetricRow[] }) {
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

function ChunkSummary({ reports }: { reports?: Partial<Record<Strategy, ChunkReport>> }) {
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

function ComparisonView({ comparison }: { comparison: QueryComparison }) {
  return (
    <>
      <div className="comparison-header">
        <div>
          <p className="eyebrow">{comparison.retriever.toUpperCase()} · {caseLabel(comparison.case_type)}</p>
          <h2>{comparison.question}</h2>
          <p className="doc-line">{comparison.doc_id}</p>
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
                  <p>R@5 {format(data?.recall_by_k?.['5'])} · MRR {format(data?.reciprocal_rank)}</p>
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

export default App;

