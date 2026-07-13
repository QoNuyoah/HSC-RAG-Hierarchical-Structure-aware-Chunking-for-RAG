import { FileJson, PlayCircle, Upload } from 'lucide-react';
import { ChunkAgentRequest, ChunkAgentResponse, LangChainAgentResponse } from '../../api';
import { ChunkParameterState, LlmEnrichmentState } from '../../types';
import AgentExecutionCard from './AgentExecutionCard';
import ChunkResponseView from './ChunkResponseView';
import DocumentSummary from './DocumentSummary';
import LlmEnrichmentPanel from './LlmEnrichmentPanel';
import ParameterPanel from './ParameterPanel';

export default function UploadWorkbench({
  fileName,
  payload,
  response,
  agentResponse,
  error,
  uploading,
  parameters,
  llmEnrichment,
  onParametersChange,
  onLlmEnrichmentChange,
  onPickFile,
  onSubmit
}: {
  fileName: string;
  payload: ChunkAgentRequest | null;
  response: ChunkAgentResponse | null;
  agentResponse: LangChainAgentResponse | null;
  error: string;
  uploading: boolean;
  parameters: ChunkParameterState;
  llmEnrichment: LlmEnrichmentState;
  onParametersChange: (parameters: ChunkParameterState) => void;
  onLlmEnrichmentChange: (settings: LlmEnrichmentState) => void;
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

        <ParameterPanel parameters={parameters} onChange={onParametersChange} />

        <LlmEnrichmentPanel settings={llmEnrichment} onChange={onLlmEnrichmentChange} />

        {payload ? (
          <DocumentSummary document={payload.document} strategy={payload.strategy ?? 'hsc_rag'} config={payload.config ?? {}} />
        ) : (
          <div className="empty-state upload-empty">未选择 JSON 文件</div>
        )}

        {error && <div className="error-banner upload-error">{error}</div>}
      </section>

      <section className="chunk-output-panel">
        {agentResponse && <AgentExecutionCard response={agentResponse} />}
        {response ? (
          <ChunkResponseView response={response} />
        ) : (
          <div className="empty-state">{agentResponse ? 'Agent 任务已完成，本次工具未返回 chunks' : '等待分段结果'}</div>
        )}
      </section>
    </main>
  );
}
