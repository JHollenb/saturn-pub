// One symbol opened: locations, read edges (mass + verdict), lifetime, uses, binding detail.
import type { SymbolRow } from '../../api/types.gen';
import { JsonTree } from '../../chrome/JsonTree';
import { Show } from '../../chrome/Loaded';
import { Count, NumView } from '../../chrome/NumView';
import { useCursor } from '../../state/cursor';
import { useView } from '../../state/records';

export function SymbolExpanded({ row }: { row: SymbolRow }) {
  const sid = useCursor((s) => s.sid);
  const go = useCursor((s) => s.go);
  const detail = useView(
    'symbol',
    sid ? `/s/${encodeURIComponent(sid)}/symbols/${encodeURIComponent(row.id)}` : null,
  );
  return (
    <div style={{ display: 'grid', gap: 6 }}>
      <div className="muted">{row.claim}</div>
      {row.abstainReason && <div className="muted">abstained: {row.abstainReason}</div>}
      <div>
        <strong>locations</strong>{' '}
        {row.locations.map((l) => (
          <button key={l} type="button" className="chip mono" onClick={() => go(l)}>
            {l}
          </button>
        ))}
        {row.locationsMore > 0 && (
          <span className="faint">
            {' '}
            +<Count n={row.locationsMore} /> more
          </span>
        )}
      </div>
      {row.readers.length > 0 && (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>reader</th>
                <th>source rows</th>
                <th>mass</th>
                <th>verdict</th>
              </tr>
            </thead>
            <tbody>
              {row.readers.map((e) => (
                <tr key={e.id}>
                  <td>
                    {e.addr ? (
                      <button type="button" className="chip mono" onClick={() => go(e.addr ?? '')}>
                        {e.addr}
                      </button>
                    ) : (
                      <span className="mono">
                        L{String(e.reader.block)} h{String(e.reader.head)}
                      </span>
                    )}
                    {e.reader.kvGroup !== null && (
                      <span className="faint"> kv{String(e.reader.kvGroup)}</span>
                    )}
                  </td>
                  <td className="mono">
                    {e.source.name} [{e.source.rows.map(String).join(',')}]
                  </td>
                  <td>
                    <NumView n={e.mass} />
                    <div className="faint">{e.massKind}</div>
                  </td>
                  <td>
                    <span className="chip">{e.verdict}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {row.readersMore > 0 && (
        <div className="faint">
          +<Count n={row.readersMore} /> more readers not shown
        </div>
      )}
      <div>
        <strong>lifetime</strong>{' '}
        {[
          ...row.lifetime.defs.map((d) => ['def', d] as const),
          ...row.lifetime.reads.map((d) => ['read', d] as const),
        ].map(([k, d], i) => (
          <span key={i} className="chip" title={d.cut ?? ''}>
            {k} {d.label}
          </span>
        ))}
        {!row.lifetime.defs.length && !row.lifetime.reads.length && (
          <span className="faint">no recorded lifetime points</span>
        )}
      </div>
      {row.uses.length > 0 && (
        <div>
          <strong>uses</strong>{' '}
          {row.uses.map((u) => (
            <span key={u.id} className="chip" title={u.closure ?? ''}>
              {u.consumer}: {u.support}
            </span>
          ))}
        </div>
      )}
      <Show what="symbol detail" loaded={detail}>
        {(d) => (
          <div>
            {d.claimBoundary && <div className="muted">claim boundary: {d.claimBoundary}</div>}
            <div>
              {Object.entries(d.statuses).map(([k, v]) => (
                <span key={k} className="chip">
                  {k}: {v}
                </span>
              ))}
            </div>
            <JsonTree value={d.binding} name="binding" depth={1} />
          </div>
        )}
      </Show>
    </div>
  );
}
