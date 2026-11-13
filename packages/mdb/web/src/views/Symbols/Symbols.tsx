// Symbols: registered symbols with their locations, readers and verdicts, plus an always-present
// "Unbound" row stating what the project does NOT cover (unresolved, never zero).
import { useState } from 'react';
import type { SymbolPage, SymbolRow } from '../../api/types.gen';
import { OmittedNote, Show } from '../../chrome/Loaded';
import { Count, NumView } from '../../chrome/NumView';
import { useCursor } from '../../state/cursor';
import { useView } from '../../state/records';
import { useUi } from '../../state/ui';
import { SymbolExpanded } from './SymbolExpanded';

export function Symbols() {
  const sid = useCursor((s) => s.sid);
  const addr = useCursor((s) => s.addr);
  const atCursor = useUi((s) => s.symbolsAtCursor);
  const set = useUi((s) => s.set);
  const [q, setQ] = useState('');
  const [page, setPage] = useState(0);
  const path = sid ? `/s/${encodeURIComponent(sid)}/symbols` : null;
  const loaded = useView('symbols', path, {
    at: atCursor ? addr : null,
    q: q || null,
    page,
  });
  if (!sid) return null;
  return (
    <section className="panel" aria-label="symbols">
      <h2>Symbols</h2>
      <div className="panel-body">
        <div style={{ display: 'flex', gap: 6, marginBottom: 6, flexWrap: 'wrap' }}>
          <input
            aria-label="filter symbols"
            placeholder="filter"
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setPage(0);
            }}
            style={{ flex: 1, minWidth: 80 }}
          />
          <label className="muted" title="only symbols defined at or before the cursor's token row">
            <input
              type="checkbox"
              checked={atCursor}
              onChange={(e) => set({ symbolsAtCursor: e.target.checked })}
            />{' '}
            at cursor
          </label>
        </div>
        <Show what="symbols" loaded={loaded}>
          {(data, env) => (
            <>
              <SymbolTable data={data} />
              {data.pages > 1 && (
                <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 6 }}>
                  <button type="button" disabled={page <= 0} onClick={() => setPage(page - 1)}>
                    ‹
                  </button>
                  page <Count n={data.page + 1} /> of <Count n={data.pages} />
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
    </section>
  );
}

function SymbolTable({ data }: { data: SymbolPage }) {
  const expanded = useCursor((s) => s.symbol);
  const open = useCursor((s) => s.open);
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>symbol</th>
            <th>status</th>
            <th>readers</th>
            <th>verdict</th>
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row) => (
            <SymbolRowView
              key={row.id}
              row={row}
              open={expanded === row.id}
              toggle={() => open({ symbol: expanded === row.id ? null : row.id })}
            />
          ))}
          <tr data-testid="unbound-row">
            <td>
              <em>Unbound</em>
            </td>
            <td colSpan={3}>
              <NumView n={data.unbound} showReason />
              <div className="faint">{data.coverage}</div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function SymbolRowView({
  row,
  open,
  toggle,
}: {
  row: SymbolRow;
  open: boolean;
  toggle: () => void;
}) {
  const go = useCursor((s) => s.go);
  const first = row.locations[0];
  return (
    <>
      <tr className={open ? 'selected-row' : undefined}>
        <td>
          <button type="button" className="row-button" aria-expanded={open} onClick={toggle}>
            <span style={{ color: 'var(--role-symbol)' }}>●</span> <strong>{row.symbol}</strong>
            {row.assignment && <div className="faint mono">{row.assignment}</div>}
          </button>
          {first && (
            <button
              type="button"
              className="chip mono"
              onClick={() => go(first)}
              title="go to first location"
            >
              {first}
            </button>
          )}
        </td>
        <td>
          {row.status}
          {row.confidence && (
            <div>
              <NumView n={row.confidence} />
            </div>
          )}
        </td>
        <td>
          <Count n={row.readers.length + row.readersMore} />
        </td>
        <td>{row.verdict}</td>
      </tr>
      {open && (
        <tr>
          <td colSpan={4}>
            <SymbolExpanded row={row} />
          </td>
        </tr>
      )}
    </>
  );
}
