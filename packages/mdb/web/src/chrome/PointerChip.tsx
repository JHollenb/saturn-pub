// A pointer to retained tensor bytes (or to the frontier where bytes are not retained).
// Clicking it inspects the pointer and moves the cursor to its address when it has one.
import type { PointerRef } from '../api/types.gen';
import { useCursor } from '../state/cursor';
import { useUi } from '../state/ui';

function describe(p: PointerRef): string {
  const spec = p.spec;
  if (!spec) return p.addr ?? p.digest.slice(0, 10);
  if (spec.kind === 'slot') {
    const item = spec.item === null ? '' : `[${String(spec.item)}]`;
    return `${spec.slot}${item}@${spec.cut.slice(0, 8)}`;
  }
  if (spec.kind === 'blob') return `blob ${spec.ref.slice(0, 10)}`;
  return spec.address;
}

export function PointerChip({ pointer }: { pointer: PointerRef }) {
  const go = useCursor((s) => s.go);
  const inspect = useUi((s) => s.inspect);
  const set = useUi((s) => s.set);
  const active = inspect?.digest === pointer.digest;
  const shape = pointer.shape ? `[${pointer.shape.join('×')}]` : '';
  return (
    <button
      type="button"
      className={`chip${active ? ' selected-row' : ''}`}
      data-status={pointer.status}
      style={pointer.status === 'frontier' ? { borderStyle: 'dashed' } : undefined}
      title={[pointer.status, pointer.reason, pointer.addr, ...pointer.via]
        .filter(Boolean)
        .join('\n')}
      onClick={() => {
        // Move first: a cursor move clears the chosen pointer, so choose it afterwards.
        if (pointer.addr) go(pointer.addr);
        set({ inspect: pointer });
      }}
    >
      <span className="mono">{describe(pointer)}</span>
      {shape && <span className="faint mono">{shape}</span>}
      {pointer.dtype && <span className="faint">{pointer.dtype}</span>}
      {pointer.status === 'frontier' && <span className="faint">frontier</span>}
    </button>
  );
}
