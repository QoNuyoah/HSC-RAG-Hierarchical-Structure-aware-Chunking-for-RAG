import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  CircleDot,
  FileJson,
  Gauge,
  ListTree,
  PlayCircle,
  RefreshCw,
  Search,
  Tags,
  Upload
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import {
  ChunkAgentRequest,
  ChunkAgentResponse,
  ChunkReport,
  GovernedDocument,
  MetricRow,
  Overview,
  QueryComparison,
  QuerySummary,
  RagChunk,
  Retriever,
  Strategy,
  getBadCases,
  getComparison,
  getMetrics,
  getOverview,
  getQueries,
  postChunk,
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

function isRecord(value: unknown): value is Record<string, any> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isGovernedDocument(value: unknown): value is GovernedDocument {
  return isRecord(value)
    && typeof value.doc_id === 'string'
    && typeof value.dataset === 'string'
    && typeof value.split === 'string'
    && typeof value.source_doc_id === 'string'
    && typeof value.title === 'string'
    && typeof value.normalization_status === 'string'
    && Array.isArray(value.blocks);
}

function normalizeChunkRequest(value: unknown): ChunkAgentRequest {
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

function App() {
  const [activePage, setActivePage] = useState<'upload' | 'evaluation'>('upload');
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
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFileName, setSelectedFileName] = useState('');
  const [uploadPayload, setUploadPayload] = useState<ChunkAgentRequest | null>(null);
  const [chunkResponse, setChunkResponse] = useState<ChunkAgentResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');

  useEffect(() => {
    if (activePage === 'evaluation' && overview === null) {
      void loadBase();
    }
  }, [activePage, overview]);

  useEffect(() => {
    if (activePage === 'evaluation') {
      void loadRetrieverData(selectedRetriever, queryMode);
    }
  }, [activePage, selectedRetriever, queryMode]);

  useEffect(() => {
    if (activePage !== 'evaluation' || !selectedQueryId) return;
    void loadComparison(selectedQueryId, selectedRetriever);
  }, [activePage, selectedQueryId, selectedRetriever]);

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

  async function handleJsonFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    try {
      setUploadError('');
      setChunkResponse(null);
      const raw = await file.text();
      const parsed = JSON.parse(raw) as unknown;
      const payload = normalizeChunkRequest(parsed);
      setSelectedFileName(file.name);
      setUploadPayload(payload);
    } catch (err) {
      setSelectedFileName(file.name);
      setUploadPayload(null);
      setChunkResponse(null);
      setUploadError(err instanceof Error ? err.message : 'JSON 文件解析失败');
    }
  }

  async function submitUploadedJson() {
    if (!uploadPayload) return;
    try {
      setUploading(true);
      setUploadError('');
      setChunkResponse(await postChunk(uploadPayload));
    } catch (err) {
      setChunkResponse(null);
      setUploadError(err instanceof Error ? err.message : '分段请求失败');
    } finally {
      setUploading(false);
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
          <p className="eyebrow">{activePage === 'upload' ? 'GovernedDocument JSON' : 'QASPER · Post-normalization packaging'}</p>
          <h1>HSC-RAG Workbench</h1>
        </div>
        <div className="toolbar">
          <div className="page-tabs">
            <button
              type="button"
              className={activePage === 'upload' ? 'active' : ''}
              onClick={() => setActivePage('upload')}
              title="上传 GovernedDocument JSON 并查看分段结果"
            >
              <Upload size={16} />
              JSON 分段
            </button>
            <button
              type="button"
              className={activePage === 'evaluation' ? 'active' : ''}
              onClick={() => setActivePage('evaluation')}
              title="查看公开数据集评估看板"
            >
              <BarChart3 size={16} />
              评估看板
            </button>
          </div>
          {activePage === 'evaluation' && (
            <>
              <select value={selectedRetriever} onChange={(event) => setSelectedRetriever(event.target.value as Retriever)}>
                {retrievers.map((retriever) => (
                  <option key={retriever} value={retriever}>{retriever.toUpperCase()}</option>
                ))}
              </select>
              <button type="button" onClick={() => void loadRetrieverData(selectedRetriever, queryMode)} title="刷新当前评估数据">
                <RefreshCw size={16} />
                刷新
              </button>
            </>
          )}
        </div>
      </header>

      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept=".json,application/json"
        onChange={(event) => void handleJsonFile(event)}
      />

      {activePage === 'upload' ? (
        <UploadWorkbench
          fileName={selectedFileName}
          payload={uploadPayload}
          response={chunkResponse}
          error={uploadError}
          uploading={uploading}
          onPickFile={() => fileInputRef.current?.click()}
          onSubmit={() => void submitUploadedJson()}
        />
      ) : (
        <>
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
                  <button className={queryMode === 'bad' ? 'active' : ''} onClick={() => setQueryMode('bad')} title="优先展示 HSC 缺失、基线领先或策略分歧的样例">Bad Cases</button>
                  <button className={queryMode === 'all' ? 'active' : ''} onClick={() => setQueryMode('all')} title="展示全部参与评估的 query">All Queries</button>
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
        </>
      )}
    </div>
  );
}

function UploadWorkbench({
  fileName,
  payload,
  response,
  error,
  uploading,
  onPickFile,
  onSubmit
}: {
  fileName: string;
  payload: ChunkAgentRequest | null;
  response: ChunkAgentResponse | null;
  error: string;
  uploading: boolean;
  onPickFile: () => void;
  onSubmit: () => void;
}) {
  return (
    <main className="upload-layout">
      <section className="upload-panel">
        <div className="section-heading">
          <FileJson size={18} />
          <h2>JSON 分段</h2>
        </div>

        <div className="upload-action-row">
          <button type="button" className="primary-button" onClick={onPickFile}>
            <Upload size={17} />
            选择文件
          </button>
          <button type="button" className="confirm-button" disabled={!payload || uploading} onClick={onSubmit}>
            <PlayCircle size={17} />
            {uploading ? '分段中' : '确认上传'}
          </button>
        </div>

        {fileName && (
          <div className="selected-file">
            <FileJson size={16} />
            <span>{fileName}</span>
          </div>
        )}

        {payload ? (
          <DocumentSummary document={payload.document} config={payload.config ?? {}} />
        ) : (
          <div className="empty-state upload-empty">未选择 JSON 文件</div>
        )}

        {error && <div className="error-banner upload-error">{error}</div>}
      </section>

      <section className="chunk-output-panel">
        {response ? <ChunkResponseView response={response} /> : <div className="empty-state">等待分段结果</div>}
      </section>
    </main>
  );
}

function DocumentSummary({ document, config }: { document: GovernedDocument; config: Record<string, unknown> }) {
  const protectedTypes = new Set(['table', 'figure', 'code', 'formula', 'list']);
  const protectedBlocks = document.blocks.filter((block) => protectedTypes.has(block.type)).length;
  const contentBlocks = document.blocks.filter((block) => block.text?.trim()).length;
  return (
    <div className="document-summary">
      <div className="document-title">
        <strong>{document.title}</strong>
        <span>{document.doc_id}</span>
      </div>
      <div className="summary-grid">
        <ResultMetric label="Blocks" value={String(document.blocks.length)} />
        <ResultMetric label="Content" value={String(contentBlocks)} />
        <ResultMetric label="Protected" value={String(protectedBlocks)} />
        <ResultMetric label="Strategy" value="HSC-RAG" />
      </div>
      <dl className="contract-list">
        <div>
          <dt>dataset</dt>
          <dd>{document.dataset}</dd>
        </div>
        <div>
          <dt>split</dt>
          <dd>{document.split}</dd>
        </div>
        <div>
          <dt>normalization</dt>
          <dd>{document.normalization_status}</dd>
        </div>
        <div>
          <dt>config</dt>
          <dd>{Object.keys(config).length ? JSON.stringify(config) : 'default'}</dd>
        </div>
      </dl>
    </div>
  );
}

function ChunkResponseView({ response }: { response: ChunkAgentResponse }) {
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

function ResultMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="result-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ChunkCard({ chunk, index }: { chunk: RagChunk; index: number }) {
  const decision = chunk.metadata?.closing_boundary_decision;
  const score = typeof decision?.boundary_score === 'number' ? decision.boundary_score.toFixed(4) : '-';
  const reason = typeof decision?.split_reason === 'string'
    ? decision.split_reason
    : chunk.quality_flags.includes('final_flush') ? 'final_flush' : '-';
  const title = chunk.title_path.length ? chunk.title_path.join(' > ') : 'Untitled';
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

export default App;
