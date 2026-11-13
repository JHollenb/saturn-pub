// Inspector: what is at the cursor. Canonical address, clock position, provenance (cut,
// receipts, symbols, records), pointers to retained bytes, and reduce controls.
import type { Frame } from '../../api/types.gen';
import { JsonTree } from '../../chrome/JsonTree';
import { Show } from '../../chrome/Loaded';
import { NotAvailable } from '../../chrome/NotAvailable';
import { Count } from '../../chrome/NumView';
import { PointerChip } from '../../chrome/PointerChip';
import { Receipt } from '../../chrome/Receipt';
import { useCursor } from '../../state/cursor';
import { useSession } from '../../state/session';
import { axisStatus } from '../Timeline/axis';
import { useView } from '../../state/records';
import { useUi } from '../../state/ui';
import { ReducePanel } from './ReducePanel';

export function Inspector() {
  const sid = useCursor((s) => s.sid);
  const addr = useCursor((s) => s.addr);
  const frame = useView('frame', sid && addr ? `/s/${encodeURIComponent(sid)}/frame` : null, {
    addr,
  });
  if (!sid) return null;
  return (
    <section className="panel" aria-label="inspector">
      <h2>Inspector</h2>
      <div className="panel-body">
        {!addr ? (
          <NotAvailable what="cursor" reason="No address yet: click a cell, or press g." />
        ) : (
          <Show what="frame" loaded={frame}>
            {(f) => <FrameView sid={sid} f={f} />}
          </Show>
        )}
      </div>
    </section>
  );
}

function FrameView({ sid, f }: { sid: string; f: Frame }) {
  const go = useCursor((s) => s.go);
  const inspect = useUi((s) => s.inspect);
  const set = useUi((s) => s.set);
  const open = useCursor((s) => s.open);
  const layersStatus = axisStatus(useSession((s) => s.caps?.layers ?? null));
  const pointers = f.focus?.pointers ?? [];
  const chosen =
    (inspect && pointers.find((p) => p.digest === inspect.digest)) ??
    inspect ??
    pointers.find((p) => p.status === 'resolvable') ??
    null;
  return (
    <div style={{ display: 'grid', gap: 8 }}>
      <div>
        <code style={{ wordBreak: 'break-all' }}>{f.addr}</code>
        <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
          <button
            type="button"
            disabled={!f.prev}
            onClick={() => f.prev && go(f.prev)}
            title="previous recorded state (←)"
          >
            ← prev
          </button>
          <button
            type="button"
            disabled={!f.next}
            onClick={() => f.next && go(f.next)}
            title="next recorded state (→)"
          >
            next →
          </button>
        </div>
      </div>
      <div className="table-scroll">
        <table>
          <tbody>
            <tr>
              <td className="muted">run</td>
              <td className="mono">{f.occ.run}</td>
            </tr>
            <tr>
              <td className="muted">branch</td>
              <td>{f.occ.branch ?? <span className="faint">none</span>}</td>
            </tr>
            {f.axes.map((a) => (
              <tr key={a.id}>
                <td className="muted">{a.label}</td>
                <td>
                  {f.pos[a.id] === null || f.pos[a.id] === undefined ? (
                    <span className="faint">not on this axis</span>
                  ) : (
                    <>
                      {a.ticks[f.pos[a.id] ?? 0]?.label ?? ''}{' '}
                      <span className="faint">
                        (index <Count n={f.pos[a.id] ?? 0} />)
                      </span>
                    </>
                  )}
                </td>
              </tr>
            ))}
            <tr>
              <td className="muted">live</td>
              <td>
                {f.live.state}
                {f.live.frontier.map((x) => (
                  <div key={x} className="faint">
                    {x}
                  </div>
                ))}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      {f.key === null && layersStatus === 'unknown' && (
        <div className="na" data-testid="cell-identity-unknown">
          cell identity unknown: layer count not recorded
        </div>
      )}
      {f.focus ? (
        <div style={{ display: 'grid', gap: 4 }}>
          {f.focus.note && <div className="muted">{f.focus.note}</div>}
          {f.focus.cut && (
            <div>
              cut <code title={f.focus.cut}>{f.focus.cut.slice(0, 16)}</code>
            </div>
          )}
          <div>
            {f.focus.receipts.map((r) => (
              <Receipt key={r} id={r} />
            ))}
          </div>
          {f.focus.symbols.length > 0 && (
            <div>
              symbols{' '}
              {f.focus.symbols.map((s) => (
                <button key={s} type="button" className="chip" onClick={() => open({ symbol: s })}>
                  {s.slice(0, 24)}
                </button>
              ))}
            </div>
          )}
          {f.focus.records.length > 0 && (
            <div>
              records{' '}
              {f.focus.records.map((r) => (
                <button
                  key={r}
                  type="button"
                  className="chip"
                  onClick={() => {
                    open({ record: r });
                    set({ drawerOpen: true });
                  }}
                >
                  {r.slice(0, 24)}
                </button>
              ))}
            </div>
          )}
          <div>
            pointers{' '}
            {pointers.length ? (
              pointers.map((p) => <PointerChip key={p.digest} pointer={p} />)
            ) : (
              <span className="faint">none retained here</span>
            )}
          </div>
        </div>
      ) : (
        <NotAvailable what="state at this address" reason="Nothing recorded at this address." />
      )}
      {chosen && <ReducePanel sid={sid} pointer={chosen} />}
      {f.stateClock && (
        <details>
          <summary className="muted">state clock (provenance)</summary>
          <JsonTree value={f.stateClock} name="stateClock" depth={1} />
        </details>
      )}
    </div>
  );
}
