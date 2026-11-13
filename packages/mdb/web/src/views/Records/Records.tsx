// Records drawer: every record the project holds (v2 records, legacy circuit rows, findings,
// slices, symbol lifecycles) with its schema, and one opened record's detail.
import { useState } from 'react';
import { OmittedNote, Show } from '../../chrome/Loaded';
import { Count, NumView } from '../../chrome/NumView';
import { useCursor } from '../../state/cursor';
import { useView } from '../../state/records';
import { useUi } from '../../state/ui';
import { RecordDetailView } from './RecordDetail';

export function Records() {
  const sid = useCursor((s) => s.sid);
  const go = useCursor((s) => s.go);
  const open = useCursor((s) => s.record);
  const openRecord = useCursor((s) => s.open);
  const kind = useUi((s) => s.recordKind);
  const drawerOpen = useUi((s) => s.drawerOpen);
  const set = useUi((s) => s.set);
  const [page, setPage] = useState(0);
  const path = sid ? `/s/${encodeURIComponent(sid)}/records` : null;
  const loaded = useView('records', path, { kind, page });
  if (!sid) return null;
  return (
    <section className="panel" aria-label="records">
      <h2>
        <button
          type="button"
          className="chip"
          aria-expanded={drawerOpen}
          onClick={() => set({ drawerOpen: !drawerOpen })}
        >
          {drawerOpen ? '▾' : '▸'} Records
        </button>
        {loaded.state === 'ok' && (
          <select
            aria-label="record kind"
            value={kind ?? ''}
            onChange={(e) => {
              set({ recordKind: e.target.value || null });
              setPage(0);
            }}
          >
            <option value="">all kinds</option>
            {Object.keys(loaded.env.data.kinds).map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        )}
      </h2>
      {drawerOpen && (
        <div className="panel-body records-body">
          <div style={{ overflow: 'auto', minWidth: 0 }}>
            <Show what="records" loaded={loaded}>
              {(data, env) => (
                <>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>kind</th>
                          <th>title</th>
                          <th>status</th>
                          <th>schema</th>
                          <th>values</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.rows.map((r) => (
                          <tr key={r.id} className={open === r.id ? 'selected-row' : undefined}>
                            <td>{r.kind}</td>
                            <td>
                              <button
                                type="button"
                                className="row-button"
                                title={r.id}
                                onClick={() => {
                                  openRecord({ record: r.id });
                                  if (r.addr) go(r.addr);
                                }}
                              >
                                {r.title || r.id}
                              </button>
                            </td>
                            <td>{r.status}</td>
                            <td className="mono faint">{r.schema}</td>
                            <td>
                              {r.nums.map((n) => (
                                <div key={n.name}>
                                  <span className="faint">{n.name}</span> <NumView n={n.n} />
                                </div>
                              ))}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {data.pages > 1 && (
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                      <button type="button" disabled={page <= 0} onClick={() => setPage(page - 1)}>
                        ‹
                      </button>
                      page <Count n={data.page + 1} /> of <Count n={data.pages} /> (
                      <Count n={data.total} /> records)
                      <button
                        type="button"
                        disabled={page >= data.pages - 1}
                        onClick={() => setPage(page + 1)}
                      >
                        ›
                      </button>
                    </div>
                  )}
                  <OmittedNote env={env} />
                </>
              )}
            </Show>
          </div>
          <div style={{ overflow: 'auto', minWidth: 0 }}>
            {open ? (
              <RecordDetailView sid={sid} id={open} />
            ) : (
              <div className="faint">Open a record to see its detail.</div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
