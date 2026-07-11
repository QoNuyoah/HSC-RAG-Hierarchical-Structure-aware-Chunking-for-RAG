import {
  BarChart3,
  RefreshCw,
  Search,
  Upload
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import {
  ChunkAgentRequest,
  ChunkAgentResponse,
  LangChainAgentResponse,
  MetricRow,
  Overview,
  QueryComparison,
  QuerySummary,
  Retriever,
  getBadCases,
  getComparison,
  getMetrics,
  getOverview,
  getQueries,
  postChunk,
  postChunkAndEnrich,
  retrievers
} from './api';
import {
  ChunkParameterState,
  DEFAULT_CHUNK_PARAMETERS,
  DEFAULT_LLM_ENRICHMENT,
  LlmEnrichmentState,
  agentInstruction,
  caseLabel,
  format,
} from './types';
import {
  applyChunkParameters,
  chunkParametersFromPayload,
  isChunkAgentResponse,
  normalizeChunkRequest
} from './utils/chunkParams';
import Stat from './components/ui/Stat';
import ChunkSummary from './components/evaluation/ChunkSummary';
import ComparisonView from './components/evaluation/ComparisonView';
import MetricsTable from './components/evaluation/MetricsTable';
import UploadWorkbench from './components/upload/UploadWorkbench';

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
  const [agentResponse, setAgentResponse] = useState<LangChainAgentResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [chunkParameters, setChunkParameters] = useState<ChunkParameterState>(DEFAULT_CHUNK_PARAMETERS);
  const [llmEnrichment, setLlmEnrichment] = useState<LlmEnrichmentState>(DEFAULT_LLM_ENRICHMENT);

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
      setAgentResponse(null);
      const raw = await file.text();
      const parsed = JSON.parse(raw) as unknown;
      const payload = normalizeChunkRequest(parsed);
      setSelectedFileName(file.name);
      setUploadPayload(payload);
      setChunkParameters(chunkParametersFromPayload(payload));
    } catch (err) {
      setSelectedFileName(file.name);
      setUploadPayload(null);
      setChunkResponse(null);
      setAgentResponse(null);
      setUploadError(err instanceof Error ? err.message : 'JSON 文件解析失败');
    }
  }

  async function submitUploadedJson() {
    const requestPayload = uploadPayload ? applyChunkParameters(uploadPayload, chunkParameters) : null;
    if (!requestPayload) return;
    try {
      setUploading(true);
      setUploadError('');
      setUploadPayload(requestPayload);
      if (llmEnrichment.enabled) {
        const agentResponse = await postChunkAndEnrich({
          instruction: agentInstruction(llmEnrichment),
          document: requestPayload.document,
          strategy: requestPayload.strategy ?? 'hsc_rag',
          config: requestPayload.config ?? {},
          include_report: requestPayload.include_report ?? true,
          preferred_tool: llmEnrichment.mode === 'stable' ? llmEnrichment.preferredTool : undefined,
          llm_provider: 'openai_compatible',
          llm_model: llmEnrichment.model.trim(),
          llm_base_url: llmEnrichment.baseUrl.trim(),
          llm_api_key_env: llmEnrichment.apiKeyEnv.trim(),
          llm_temperature: 0,
          llm_timeout_seconds: Math.max(1, llmEnrichment.timeoutSeconds),
          llm_use_response_format: llmEnrichment.useResponseFormat
        });
        setAgentResponse(agentResponse);
        setChunkResponse(isChunkAgentResponse(agentResponse.result) ? agentResponse.result : null);
      } else {
        setAgentResponse(null);
        setChunkResponse(await postChunk(requestPayload));
      }
    } catch (err) {
      setChunkResponse(null);
      setUploadError(err instanceof Error ? err.message : '分段请求失败');
    } finally {
      setUploading(false);
    }
  }

  function handleChunkParametersChange(nextParameters: ChunkParameterState) {
    setChunkParameters(nextParameters);
    setChunkResponse(null);
    setAgentResponse(null);
  }

  const filteredQueries = useMemo(() => {
    const text = searchText.trim().toLowerCase();
    if (!text) return queries;
    return queries.filter((item) =>
      `${item.question} ${item.doc_id} ${item.query_id}`.toLowerCase().includes(text)
    );
  }, [queries, searchText]);

  const hscMetric = metrics.find((row) => row.strategy === 'hsc_rag');
  const configuredUploadPayload = useMemo(
    () => (uploadPayload ? applyChunkParameters(uploadPayload, chunkParameters) : null),
    [uploadPayload, chunkParameters]
  );

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
          payload={configuredUploadPayload}
          response={chunkResponse}
          agentResponse={agentResponse}
          error={uploadError}
          uploading={uploading}
          parameters={chunkParameters}
          llmEnrichment={llmEnrichment}
          onParametersChange={handleChunkParametersChange}
          onLlmEnrichmentChange={(next) => {
            setLlmEnrichment(next);
            setChunkResponse(null);
            setAgentResponse(null);
          }}
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

export default App;
