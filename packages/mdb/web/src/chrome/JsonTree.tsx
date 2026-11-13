// Bounded JSON display for record bodies and bindings. Numbers follow the evidence rule:
// a labelled record (`{v, ev}` or the version-2 `{value, authority}`) renders through NumView; a bare number under a structural key (ids,
// indices, shapes, counts, clock positions) renders as a Count; any other bare number is
// refused ("no evidence class") rather than printed.
import { useState } from 'react';
import { asNum } from './evidence';
import { STRUCTURAL_KEYS } from '../api/types.gen';
import { Count, NumView } from './NumView';

// One list: the server's allowlist of structural keys (generated into `api/types.gen.ts`).
const STRUCTURAL = new Set<string>(STRUCTURAL_KEYS);

export function isStructuralKey(key: string | null): boolean {
  return key !== null && STRUCTURAL.has(key);
}

const MAX_CHILDREN = 50;

export function JsonTree({
  value,
  name = null,
  depth = 0,
}: {
  value: unknown;
  name?: string | null;
  depth?: number;
}) {
  return <Node value={value} keyName={name} structural={isStructuralKey(name)} depth={depth} />;
}

function Node({
  value,
  keyName,
  structural,
  depth,
}: {
  value: unknown;
  keyName: string | null;
  structural: boolean;
  depth: number;
}) {
  const [open, setOpen] = useState(depth < 2);
  if (value === null || value === undefined) return <span className="faint">null</span>;
  if (typeof value === 'number') {
    if (structural && Number.isInteger(value)) return <Count n={value} />;
    return (
      <NumView
        n={{
          v: null,
          ev: 'refused',
          unit: null,
          proof: null,
          ptr: null,
          note: null,
          recorded_authority: null,
          reason: `number at "${keyName ?? '?'}" has no evidence class`,
        }}
        showReason
      />
    );
  }
  if (typeof value === 'string')
    return <span className="mono json-str">{JSON.stringify(value)}</span>;
  if (typeof value === 'boolean') return <span className="mono">{String(value)}</span>;
  const labelled = asNum(value);
  if (labelled) return <NumView n={labelled} showReason />;
  const entries: [string | null, unknown][] = Array.isArray(value)
    ? value.map((v) => [null, v])
    : Object.entries(value as Record<string, unknown>);
  const label = Array.isArray(value) ? `[${entries.length}]` : `{${entries.length}}`;
  if (!entries.length)
    return <span className="faint mono">{Array.isArray(value) ? '[]' : '{}'}</span>;
  return (
    <span>
      <button type="button" className="chip" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? '−' : '+'} {label}
      </button>
      {open && (
        <ul style={{ listStyle: 'none', margin: 0, paddingLeft: 14 }}>
          {entries.slice(0, MAX_CHILDREN).map(([k, v], i) => (
            <li key={k ?? i}>
              {k !== null && <span className="muted mono">{k}: </span>}
              <Node
                value={v}
                keyName={k ?? keyName}
                structural={k === null ? structural : isStructuralKey(k)}
                depth={depth + 1}
              />
            </li>
          ))}
          {entries.length > MAX_CHILDREN && (
            <li className="faint">
              +<Count n={entries.length - MAX_CHILDREN} /> more not shown
            </li>
          )}
        </ul>
      )}
    </span>
  );
}
