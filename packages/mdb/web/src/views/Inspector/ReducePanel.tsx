// Reduce controls: the browser asks the server for bounded reductions of one pointer; it never
// receives the tensor itself.
import { useState } from 'react';
import type { PointerRef } from '../../api/types.gen';
import { RefusedView } from '../../chrome/NotAvailable';
import { useReduce, reduceKey, type ReduceOp } from '../../state/reduce';
import { ReduceResultView } from './ReduceResultView';

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return undefined;
  }
}

export function ReducePanel({ sid, pointer }: { sid: string; pointer: PointerRef }) {
  const run = useReduce((s) => s.run);
  const entries = useReduce((s) => s.entries);
  const [op, setOp] = useState<ReduceOp>('stats');
  const [k, setK] = useState(8);
  const [bins, setBins] = useState(32);
  const [level, setLevel] = useState(0);
  const [selector, setSelector] = useState('[0]');
  const [direction, setDirection] = useState('');
  const [error, setError] = useState<string | null>(null);
  const params = (): object | null => {
    if (op === 'topk') return { k };
    if (op === 'histogram') return { bins };
    if (op === 'tile') return { level };
    if (op === 'slice') {
      const s = parseJson(selector);
      return Array.isArray(s) ? { selector: s } : null;
    }
    if (op === 'project') {
      const d = parseJson(direction);
      return d && typeof d === 'object' ? { direction: d } : null;
    }
    return {};
  };
  const p = params();
  const key = p ? reduceKey(sid, pointer.digest, op, p) : null;
  const entry = key ? entries.get(key) : undefined;
  if (pointer.status === 'frontier' || !pointer.spec) {
    return <RefusedView what="reduce" refused={{ refused: 'frontier', detail: pointer.reason }} />;
  }
  return (
    <div style={{ display: 'grid', gap: 6 }}>
      <form
        style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}
        onSubmit={(e) => {
          e.preventDefault();
          if (!p)
            return setError(
              op === 'slice' ? 'selector is a JSON list' : 'direction is a JSON pointer spec',
            );
          setError(null);
          void run(sid, pointer, op, p);
        }}
      >
        <select
          aria-label="reduce op"
          value={op}
          onChange={(e) => setOp(e.target.value as ReduceOp)}
        >
          {(['stats', 'topk', 'histogram', 'tile', 'slice', 'project'] as const).map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
        {op === 'topk' && (
          <input
            aria-label="k"
            type="number"
            min={1}
            max={64}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            style={{ width: 56 }}
          />
        )}
        {op === 'histogram' && (
          <input
            aria-label="bins"
            type="number"
            min={1}
            max={256}
            value={bins}
            onChange={(e) => setBins(Number(e.target.value))}
            style={{ width: 56 }}
          />
        )}
        {op === 'tile' && (
          <input
            aria-label="level"
            type="number"
            min={0}
            max={16}
            value={level}
            onChange={(e) => setLevel(Number(e.target.value))}
            style={{ width: 56 }}
          />
        )}
        {op === 'slice' && (
          <input
            aria-label="selector"
            className="mono"
            value={selector}
            onChange={(e) => setSelector(e.target.value)}
            placeholder="[0, [0, 16]]"
          />
        )}
        {op === 'project' && (
          <input
            aria-label="direction"
            className="mono"
            value={direction}
            onChange={(e) => setDirection(e.target.value)}
            placeholder='{"kind":"blob","ref":"…"}'
          />
        )}
        <button type="submit">run</button>
      </form>
      {error && <div className="muted">{error}</div>}
      {entry?.state === 'loading' && <div className="faint">reducing…</div>}
      {entry?.state === 'refused' && <RefusedView what={op} refused={entry.refused} />}
      {entry?.state === 'ok' && <ReduceResultView r={entry.result} />}
    </div>
  );
}
