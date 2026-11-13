// Branch rail: one button per recorded branch. Clicking moves the cursor to the same clock
// point on that branch (the branch is the address origin; `main` is the default origin).
import { originOf } from '../../api/address';
import { Count } from '../../chrome/NumView';
import { useCursor } from '../../state/cursor';
import { useView } from '../../state/records';

export function BranchRail({ sid }: { sid: string }) {
  const addr = useCursor((s) => s.addr);
  const go = useCursor((s) => s.go);
  const tree = useView('branches', `/s/${encodeURIComponent(sid)}/branches`);
  if (tree.state !== 'ok') return null;
  const origin = addr ? originOf(addr) : '';
  const bare = addr ? addr.slice(origin.length) : null;
  const active = origin && !origin.startsWith('@') ? origin.slice(0, -1) : 'main';
  return (
    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }} aria-label="branches">
      {tree.env.data.branches.map((b) => (
        <button
          key={b.name}
          type="button"
          className={`chip${b.name === active ? ' selected-row' : ''}`}
          aria-pressed={b.name === active}
          title={`head ${b.head}${b.base ? `, forked from ${b.base}` : ''}`}
          disabled={!bare}
          onClick={() => bare && go(b.name === 'main' ? bare : `${b.name}:${bare}`)}
        >
          {b.name} <Count n={b.cuts} unit="cuts" />
        </button>
      ))}
    </div>
  );
}
