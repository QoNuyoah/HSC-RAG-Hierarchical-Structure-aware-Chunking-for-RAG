import { LlmEnrichmentState } from '../../types';
import NumberField from '../ui/NumberField';
import TextField from '../ui/TextField';

export default function LlmEnrichmentPanel({
  settings,
  onChange
}: {
  settings: LlmEnrichmentState;
  onChange: (settings: LlmEnrichmentState) => void;
}) {
  const update = <K extends keyof LlmEnrichmentState>(key: K, value: LlmEnrichmentState[K]) => {
    onChange({ ...settings, [key]: value });
  };

  return (
    <section className="parameter-panel">
      <label className="toggle-field llm-toggle">
        <input
          type="checkbox"
          checked={settings.enabled}
          onChange={(event) => update('enabled', event.target.checked)}
        />
        <span>启用 agent 模式 </span>
      </label>

      {settings.enabled && (
        <>
          <div className="parameter-grid llm-parameter-grid">
            <label className="field-block">
              <span>Agent 执行模式</span>
              <select value={settings.mode} onChange={(event) => update('mode', event.target.value as LlmEnrichmentState['mode'])}>
                <option value="stable">稳定模式：指定工具</option>
                <option value="autonomous">自主模式：Agent 选择工具</option>
              </select>
            </label>
            {settings.mode === 'stable' && (
              <label className="field-block">
                <span>LangChain Tool</span>
                <select
                  value={settings.preferredTool}
                  onChange={(event) => update('preferredTool', event.target.value as LlmEnrichmentState['preferredTool'])}
                >
                  <option value="inspect_hsc_rag_context">查看文档上下文</option>
                  <option value="chunk_current_document">仅执行文档分段</option>
                  <option value="chunk_and_enrich_current_document">分段 + LLM 语义增强</option>
                </select>
              </label>
            )}
          </div>
          {settings.mode === 'autonomous' && (
            <label className="field-block full-field llm-instruction-field">
              <span>Agent 任务指令</span>
              <textarea value={settings.instruction} onChange={(event) => update('instruction', event.target.value)} />
            </label>
          )}
          {settings.mode === 'stable' && settings.preferredTool === 'chunk_and_enrich_current_document' && (
            <label className="toggle-field qa-toggle-field">
              <input
                type="checkbox"
                checked={settings.includeQa}
                onChange={(event) => update('includeQa', event.target.checked)}
              />
              <span>为每个 chunk 生成 QA 问答样本</span>
            </label>
          )}
          <div className="parameter-grid llm-parameter-grid">
            <TextField label="模型" value={settings.model} onChange={(value) => update('model', value)} />
            <TextField label="Base URL" value={settings.baseUrl} onChange={(value) => update('baseUrl', value)} />
            <TextField label="API Key 环境变量" value={settings.apiKeyEnv} onChange={(value) => update('apiKeyEnv', value)} />
            <NumberField
              label="超时（秒）"
              value={settings.timeoutSeconds}
              min={1}
              step={1}
              onChange={(value) => update('timeoutSeconds', value)}
            />
          </div>
          <p className="llm-hint">API Key 仅由后端从环境变量读取，不会发送到浏览器。</p>
        </>
      )}
    </section>
  );
}
