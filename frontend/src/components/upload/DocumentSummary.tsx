import { GovernedDocument, Strategy } from '../../api';
import { strategyNames } from '../../types';
import ResultMetric from '../ui/ResultMetric';

export default function DocumentSummary({
  document,
  strategy,
  config
}: {
  document: GovernedDocument;
  strategy: Strategy;
  config: Record<string, unknown>;
}) {
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
        <ResultMetric label="Strategy" value={strategyNames[strategy]} />
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
