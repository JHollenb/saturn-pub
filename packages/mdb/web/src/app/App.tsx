// Layout: header; left Symbols; centre Timeline over Graph; right Inspector; bottom Records.
import { useEffect } from 'react';
import { AddressBar } from '../chrome/AddressBar';
import { CommandLine } from '../chrome/CommandLine';
import { ErrorBoundary } from '../chrome/ErrorBoundary';
import { Help } from '../chrome/Help';
import { NotAvailable, RefusedView } from '../chrome/NotAvailable';
import { useCursor } from '../state/cursor';
import { useView } from '../state/records';
import { useSession } from '../state/session';
import { useUi, type Theme } from '../state/ui';
import { Graph } from '../views/Graph/Graph';
import { Inspector } from '../views/Inspector/Inspector';
import { Records } from '../views/Records/Records';
import { Symbols } from '../views/Symbols/Symbols';
import { Timeline } from '../views/Timeline/Timeline';
import { useEvents } from './events';
import { useKeys } from './keys';

export function App() {
  const boot = useSession((s) => s.boot);
  const hello = useSession((s) => s.hello);
  const sessions = useSession((s) => s.sessions);
  const refused = useSession((s) => s.refused);
  const loadCaps = useSession((s) => s.loadCaps);
  const sid = useCursor((s) => s.sid);
  const addr = useCursor((s) => s.addr);
  const setSession = useCursor((s) => s.setSession);
  const theme = useUi((s) => s.theme);
  const set = useUi((s) => s.set);
  const link = useEvents(sid);
  const frame = useView('frame', sid && addr ? `/s/${encodeURIComponent(sid)}/frame` : null, {
    addr,
  });
  useKeys(frame.state === 'ok' ? frame.env.data : null);

  useEffect(() => {
    useCursor.getState().syncFromLocation();
    void boot();
  }, [boot]);
  useEffect(() => {
    const first = sessions[0];
    if (!sid && first) setSession(first.id, first.home);
    else if (sid && !addr) {
      const s = sessions.find((x) => x.id === sid);
      if (s?.home) useCursor.getState().go(s.home);
    }
  }, [sid, addr, sessions, setSession]);
  useEffect(() => {
    if (sid) void loadCaps(sid);
  }, [sid, loadCaps]);
  useEffect(() => {
    if (theme === 'system') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
  }, [theme]);

  return (
    <div className="app">
      <header className="app-header">
        <strong>mdb</strong>
        <select
          aria-label="session"
          value={sid ?? ''}
          onChange={(e) => {
            const s = sessions.find((x) => x.id === e.target.value);
            if (s) setSession(s.id, s.home);
          }}
        >
          {sessions.map((s) => (
            <option key={s.id} value={s.id}>
              {s.id} · {s.model}
            </option>
          ))}
        </select>
        <AddressBar />
        <span className="chip header-chip" title="server link (recorded project: no live model)">
          {hello?.readOnly ? 'read-only' : 'writable'} · {link}
        </span>
        <select
          aria-label="theme"
          value={theme}
          onChange={(e) => set({ theme: e.target.value as Theme })}
        >
          <option value="system">system</option>
          <option value="light">light</option>
          <option value="dark">dark</option>
        </select>
        <button type="button" onClick={() => set({ help: true })} title="help (?)">
          ?
        </button>
      </header>
      {refused && (
        <div className="app-banner">
          <RefusedView what="server" refused={refused} />
        </div>
      )}
      {hello && !sessions.length ? (
        <div className="app-banner">
          <NotAvailable what="sessions" reason="This project holds no sessions." />
        </div>
      ) : (
        <>
          <aside className="app-left">
            <ErrorBoundary name="Symbols">
              <Symbols />
            </ErrorBoundary>
          </aside>
          <main className="app-centre">
            <ErrorBoundary name="Timeline">
              <Timeline />
            </ErrorBoundary>
            <ErrorBoundary name="Graph">
              <Graph />
            </ErrorBoundary>
          </main>
          <aside className="app-right">
            <ErrorBoundary name="Inspector">
              <Inspector />
            </ErrorBoundary>
          </aside>
          <footer className="app-bottom">
            <ErrorBoundary name="Records">
              <Records />
            </ErrorBoundary>
          </footer>
        </>
      )}
      <CommandLine />
      <Help />
    </div>
  );
}
