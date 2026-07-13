export default function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="field-block">
      <span>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => {
          const raw = event.target.value.trim();
          if (!raw) return;
          const next = Number(raw);
          if (Number.isFinite(next)) {
            onChange(next);
          }
        }}
      />
    </label>
  );
}
