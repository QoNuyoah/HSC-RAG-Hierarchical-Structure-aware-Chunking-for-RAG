import { SlidersHorizontal } from 'lucide-react';
import { Strategy, strategies } from '../../api';
import { ChunkParameterState, strategyNames } from '../../types';
import NumberField from '../ui/NumberField';
import ToggleField from '../ui/ToggleField';

export default function ParameterPanel({
  parameters,
  onChange
}: {
  parameters: ChunkParameterState;
  onChange: (parameters: ChunkParameterState) => void;
}) {
  const update = <K extends keyof ChunkParameterState>(key: K, value: ChunkParameterState[K]) => {
    onChange({ ...parameters, [key]: value });
  };
  const showOverlap = parameters.strategy === 'fixed' || parameters.strategy === 'recursive';
  const showSemantic = parameters.strategy === 'semantic';
  const showHsc = parameters.strategy === 'hsc_rag';
  const showProtectedBlocks = showSemantic || showHsc;

  return (
    <section className="parameter-panel">
      <div className="parameter-head">
        <div className="section-heading">
          <SlidersHorizontal size={18} />
          <h2>分段参数</h2>
        </div>
        <span>{strategyNames[parameters.strategy]}</span>
      </div>

      <label className="field-block full-field">
        <span>分段策略</span>
        <select
          value={parameters.strategy}
          onChange={(event) => update('strategy', event.target.value as Strategy)}
        >
          {strategies.map((strategy) => (
            <option key={strategy} value={strategy}>{strategyNames[strategy]}</option>
          ))}
        </select>
      </label>

      <div className="parameter-grid">
        <NumberField
          label="Min Tokens"
          min={1}
          step={1}
          value={parameters.min_tokens}
          onChange={(value) => update('min_tokens', value)}
        />
        <NumberField
          label="Target Tokens"
          min={1}
          step={1}
          value={parameters.target_tokens}
          onChange={(value) => update('target_tokens', value)}
        />
        <NumberField
          label="Max Tokens"
          min={1}
          step={1}
          value={parameters.max_tokens}
          onChange={(value) => update('max_tokens', value)}
        />
        {showOverlap && (
          <NumberField
            label="Overlap"
            min={0}
            step={1}
            value={parameters.overlap_tokens}
            onChange={(value) => update('overlap_tokens', value)}
          />
        )}
        {showSemantic && (
          <NumberField
            label="Breakpoint %"
            min={1}
            max={99}
            step={1}
            value={parameters.breakpoint_percentile}
            onChange={(value) => update('breakpoint_percentile', value)}
          />
        )}
        {showHsc && (
          <NumberField
            label="Window Blocks"
            min={1}
            step={1}
            value={parameters.semantic_window_blocks}
            onChange={(value) => update('semantic_window_blocks', value)}
          />
        )}
      </div>

      <div className="toggle-grid">
        <ToggleField
          label="标题上下文"
          checked={parameters.include_title_context}
          onChange={(checked) => update('include_title_context', checked)}
        />
        <ToggleField
          label="返回报告"
          checked={parameters.include_report}
          onChange={(checked) => update('include_report', checked)}
        />
        {showProtectedBlocks && (
          <ToggleField
            label="保护表格/代码/公式"
            checked={parameters.protect_blocks}
            onChange={(checked) => update('protect_blocks', checked)}
          />
        )}
        {showHsc && (
          <>
            <ToggleField
              label="自适应边界"
              checked={parameters.adaptive_boundary}
              onChange={(checked) => update('adaptive_boundary', checked)}
            />
            <ToggleField
              label="语义边界评分"
              checked={parameters.semantic_boundary_scoring}
              onChange={(checked) => update('semantic_boundary_scoring', checked)}
            />
          </>
        )}
      </div>

      {showHsc && (
        <div className="advanced-parameter-grid">
          <NumberField
            label="Boundary"
            min={0}
            max={1}
            step={0.01}
            value={parameters.semantic_boundary_threshold}
            onChange={(value) => update('semantic_boundary_threshold', value)}
          />
          <NumberField
            label="Soft Boundary"
            min={0}
            max={1}
            step={0.01}
            value={parameters.semantic_soft_boundary_threshold}
            onChange={(value) => update('semantic_soft_boundary_threshold', value)}
          />
          <NumberField
            label="Distance"
            min={0}
            max={1}
            step={0.01}
            value={parameters.semantic_distance_threshold}
            onChange={(value) => update('semantic_distance_threshold', value)}
          />
          <NumberField
            label="W Structure"
            min={0}
            max={1}
            step={0.01}
            value={parameters.structure_signal_weight}
            onChange={(value) => update('structure_signal_weight', value)}
          />
          <NumberField
            label="W Semantic"
            min={0}
            max={1}
            step={0.01}
            value={parameters.semantic_signal_weight}
            onChange={(value) => update('semantic_signal_weight', value)}
          />
          <NumberField
            label="W Length"
            min={0}
            max={1}
            step={0.01}
            value={parameters.length_signal_weight}
            onChange={(value) => update('length_signal_weight', value)}
          />
        </div>
      )}
    </section>
  );
}
