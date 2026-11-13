// The error split of a `whatif` (P20): native change = linear + nonlinear + realisation +
// rounding (or exact + realisation + rounding when the linear term is refused). Scalar terms are
// drawn as one stacked bar (positive segments rightward from the origin, negative leftward),
// every segment labelled with its name and its value below. Summary terms (a vector consumer:
// max_abs / norm) do not add, so they are drawn side by side and the bar says so.
import { Fragment } from 'react';
import { OpValue } from '../../chrome/OpValue';
import { at, isRec, magnitude, recs, str, valOf } from '../../chrome/labelled';

const W = 320;
const H = 18;

export function SplitBar({ split }: { split: unknown }) {
  const terms = recs(isRec(split) ? split.terms : undefined);
  if (!terms.length) return <OpValue value={undefined} what="error split" />;
  const parts = terms.map((t) => ({
    name: str(t.name) ?? '?',
    value: t.value,
    meaning: str(t.meaning),
    bound: t.bound,
    within: t.within_bound,
    m: magnitude(valOf(t.value)),
  }));
  const scalar = parts.every((p) => p.m === null || p.m.signed);
  const pos = parts.reduce((s, p) => s + (p.m && p.m.v > 0 ? p.m.v : 0), 0);
  const neg = parts.reduce((s, p) => s + (p.m && p.m.v < 0 ? -p.m.v : 0), 0);
  const span = scalar ? pos + neg : Math.max(0, ...parts.map((p) => (p.m ? Math.abs(p.m.v) : 0)));
  const origin = scalar && span > 0 ? (neg / span) * W : 0;
  let right = origin;
  let left = origin;
  return (
    <div className="split" data-testid="split">
      <div className="faint">
        error split{str(at(split, 'proof')) ? ` (${str(at(split, 'proof'))})` : ''}:{' '}
        {str(at(split, 'rule')) ?? ''}
        {!scalar && ' · a vector consumer: each term is a summary (norm drawn); they do not add'}
      </div>
      <svg width={W + 2} height={scalar ? H + 14 : parts.length * (H + 4)} role="img" aria-label="error split bar">
        {parts.map((p, i) => {
          if (!p.m || span === 0) return null;
          const w = (Math.abs(p.m.v) / span) * W;
          let x: number;
          let y = 0;
          if (scalar) {
            if (p.m.v >= 0) {
              x = right;
              right += w;
            } else {
              left -= w;
              x = left;
            }
          } else {
            x = 0;
            y = i * (H + 4);
          }
          return (
            <g key={p.name} data-term={p.name}>
              <rect
                x={x + 1}
                y={y}
                width={Math.max(w, 1)}
                height={H}
                fill={i % 2 ? 'var(--panel-2)' : 'var(--panel)'}
                stroke="var(--ink-2)"
                strokeDasharray={p.m.v < 0 ? '4 2' : undefined}
              >
                <title>{`${p.name}${p.meaning ? `: ${p.meaning}` : ''}`}</title>
              </rect>
              <text x={x + 4} y={y + 13} fontSize={10} fill="var(--ink)">
                {w > p.name.length * 6 + 6 ? p.name : w > 10 ? String(i + 1) : ''}
              </text>
            </g>
          );
        })}
        {scalar && span > 0 && (
          <line x1={origin + 1} x2={origin + 1} y1={0} y2={H + 4} stroke="var(--ink)">
            <title>origin (0)</title>
          </line>
        )}
      </svg>
      <ol className="split-terms">
        {parts.map((p) => (
          <li key={p.name} data-term={p.name}>
            <strong>{p.name}</strong> <OpValue value={p.value} what={p.name} />
            {p.bound !== undefined && (
              <Fragment>
                {' '}
                <span className="faint">bound</span> <OpValue value={p.bound} what="bound" />
                {p.within === true ? ' within' : p.within === false ? ' OUTSIDE' : ''}
              </Fragment>
            )}
            {p.meaning && <span className="faint"> — {p.meaning}</span>}
          </li>
        ))}
      </ol>
    </div>
  );
}
