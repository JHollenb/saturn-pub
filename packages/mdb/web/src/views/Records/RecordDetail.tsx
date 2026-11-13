// One record: schema, provenance, comparison arms (sham first, as the server orders them),
// pointers to retained bytes, links, and the bounded body.
import type { ArmRow } from '../../api/types.gen';
import { JsonTree } from '../../chrome/JsonTree';
import { OmittedNote, Show } from '../../chrome/Loaded';
import { Count, NumView } from '../../chrome/NumView';
import { PointerChip } from '../../chrome/PointerChip';
import { useCursor } from '../../state/cursor';
import { useView } from '../../state/records';

export function RecordDetailView({ sid, id }: { sid: string; id: string }) {
  const loaded = useView(
    'record',
    `/s/${encodeURIComponent(sid)}/records/${encodeURIComponent(id)}`,
  );
  const open = useCursor((s) => s.open);
  const go = useCursor((s) => s.go);
  return (
    <Show what="record" loaded={loaded}>
      {(d, env) => (
        <div style={{ display: 'grid', gap: 6 }}>
          <div>
            <strong>{d.summary.title || d.summary.id}</strong>{' '}
            <span className="chip mono" title="record schema">
              {d.summary.schema}
            </span>{' '}
            <span className="chip">{d.summary.status}</span>
          </div>
          <div className="faint mono" style={{ wordBreak: 'break-all' }}>
            {d.summary.id}
          </div>
          <div className="muted">
            authority: {d.summary.authority || 'not stated'}
            {d.summary.addr && (
              <>
                {' '}
                ·{' '}
                <button
                  type="button"
                  className="chip mono"
                  onClick={() => go(d.summary.addr ?? '')}
                >
                  {d.summary.addr}
                </button>
              </>
            )}
          </div>
          {d.arms && <Arms arms={d.arms} onOpen={(rid) => open({ record: rid })} />}
          {d.pointers.length > 0 && (
            <div>
              <strong>pointers</strong>{' '}
              {d.pointers.map((p) => (
                <PointerChip key={p.digest} pointer={p} />
              ))}
            </div>
          )}
          {d.links.length > 0 && (
            <div>
              <strong>links</strong>{' '}
              {d.links.map((l) => (
                <button
                  key={`${l.rel}:${l.id}`}
                  type="button"
                  className="chip"
                  onClick={() => open({ record: l.id })}
                >
                  {l.rel} {l.id.slice(0, 24)}
                </button>
              ))}
            </div>
          )}
          <JsonTree value={d.body} name="body" depth={1} />
          <OmittedNote env={env} />
        </div>
      )}
    </Show>
  );
}

function Arms({ arms, onOpen }: { arms: ArmRow[]; onOpen: (id: string) => void }) {
  return (
    <div className="table-scroll arms-table">
      <table aria-label="comparison arms">
        <thead>
          <tr>
            <th>arm</th>
            <th>status</th>
            <th>value</th>
            <th>effect vs sham</th>
            <th>prediction</th>
            <th>residual</th>
            <th>decision</th>
          </tr>
        </thead>
        <tbody>
          {arms.map((a) => (
            <tr key={a.record}>
              <td>
                <button type="button" className="row-button" onClick={() => onOpen(a.record)}>
                  {a.label}{' '}
                  <span className="faint">
                    mask <Count n={a.mask} />
                  </span>
                </button>
              </td>
              <td>{a.status}</td>
              <td>
                <NumView n={a.value} compact />
              </td>
              <td>
                <NumView n={a.effect} compact />
              </td>
              <td>
                {a.prediction ? (
                  <NumView n={a.prediction} compact />
                ) : (
                  <span className="faint">none</span>
                )}
              </td>
              <td>
                {a.residual ? (
                  <NumView n={a.residual} compact />
                ) : (
                  <span className="faint">none</span>
                )}
              </td>
              <td>{a.decision ?? <span className="faint">none</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
