import { useState } from 'react';
import { LangChainAgentResponse } from '../../api';

export default function AgentExecutionCard({ response }: { response: LangChainAgentResponse }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <section className="agent-execution-card">
      <div className="agent-execution-head">
        <div>
          <span className="eyebrow">LangChain Agent</span>
          <h2>{response.selected_tool ? 'Agent 执行成功' : 'Agent 已返回'}</h2>
        </div>
        <button type="button" className="agent-detail-button" onClick={() => setExpanded((value) => !value)}>
          {expanded ? '收起' : '查看轨迹'}
        </button>
      </div>
      <div className="agent-summary-grid">
        <div><span>选择工具</span><strong>{response.selected_tool ?? '-'}</strong></div>
        <div><span>Provider</span><strong>{response.provider}</strong></div>
        <div><span>Model</span><strong>{response.model ?? '-'}</strong></div>
        <div><span>工具调用</span><strong>{response.tool_trace.length} 次</strong></div>
      </div>
      <p className="agent-instruction"><strong>任务：</strong>{response.instruction}</p>
      {expanded && (
        <div className="agent-detail">
          <p><strong>Agent 回答：</strong>{response.answer || '-'}</p>
          {response.warnings.length > 0 && <p className="llm-error"><strong>警告：</strong>{response.warnings.join('；')}</p>}
          <pre>{JSON.stringify(response.tool_trace, null, 2)}</pre>
        </div>
      )}
    </section>
  );
}
