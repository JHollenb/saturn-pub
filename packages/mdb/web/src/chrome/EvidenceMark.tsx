// A 10px mark for an evidence class: solid (measured), hollow (derived-exact / derived-local /
// bound), hatched (supplied), dashed empty (unresolved), crossed (refused). Neutral ink only.
import type { Ev } from '../api/types.gen';
import { markStyle } from './evidence';

export function EvidenceMark({ ev, size = 10 }: { ev: Ev; size?: number }) {
  const style = markStyle(ev);
  const r = size / 2 - 1;
  const c = size / 2;
  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label={`${ev} evidence`}
      data-ev={ev}
      style={{ verticalAlign: 'middle', flex: 'none' }}
    >
      <title>{ev}</title>
      {style === 'solid' && <circle cx={c} cy={c} r={r} fill="currentColor" />}
      {style === 'hollow' && (
        <circle cx={c} cy={c} r={r} fill="none" stroke="currentColor" strokeWidth={1.4} />
      )}
      {style === 'hatched' && (
        <g stroke="currentColor" strokeWidth={1}>
          <circle cx={c} cy={c} r={r} fill="none" />
          <line x1={c - r * 0.7} y1={c + r * 0.7} x2={c + r * 0.7} y2={c - r * 0.7} />
          <line x1={c - r} y1={c} x2={c} y2={c - r} />
          <line x1={c} y1={c + r} x2={c + r} y2={c} />
        </g>
      )}
      {style === 'dashed-empty' && (
        <circle
          cx={c}
          cy={c}
          r={r}
          fill="none"
          stroke="currentColor"
          strokeWidth={1}
          strokeDasharray="2 1.5"
          opacity={0.7}
        />
      )}
      {style === 'refused' && (
        <g stroke="currentColor" strokeWidth={1.4}>
          <line x1={1} y1={1} x2={size - 1} y2={size - 1} />
          <line x1={size - 1} y1={1} x2={1} y2={size - 1} />
        </g>
      )}
    </svg>
  );
}
