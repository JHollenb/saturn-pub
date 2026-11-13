// Layout: header; left Symbols; centre Timeline over Graph (or, on the live session, the Live
// debugger); right Inspector; bottom Records.
import { useEffect } from 'react';
import { AddressBar } from '../chrome/AddressBar';
import { CommandLine } from '../chrome/CommandLine';
import { ErrorBoundary } from '../chrome/ErrorBoundary';
import { Help } from '../chrome/Help';
import { NotAvailable, RefusedView } from '../chrome/NotAvailable';
import { useCursor } from '../state/cursor';
import { useInspect } from '../state/inspect';
import { useView } from '../state/records';
import { useSession } from '../state/session';
import { useUi, type Theme } from '../state/ui';
import { Graph } from '../views/Graph/Graph';
import { Inspector } from '../views/Inspector/Inspector';
import { Live } from '../views/Live/Live';
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
  const liveSid = hello?.live ?? null;
  const isLive = !!liveSid && sid === liveSid;
  useEffect(() => {
    const first = sessions[0];
    if (!sid && liveSid) setSession(liveSid);
    else if (!sid && first) setSession(first.id, first.home);
    else if (sid && !addr) {
      const s = sessions.find((x) => x.id === sid);
      if (s?.home) useCursor.getState().go(s.home);
    }
  }, [sid, addr, sessions, setSession, liveSid]);
  useEffect(() => {
    if (sid) void loadCaps(sid);
  }, [sid, loadCaps]);
  useEffect(() => {
    // the live session gives the Live pane the room: the Records drawer starts collapsed, and
    // the Inspector opens by itself when something is chosen for it (a circuit node, a pointer)
    if (!isLive) return;
    set({ drawerOpen: false });
    const offInspect = useInspect.subscribe((state, prev) => {
      if (state.node !== null && state.node !== prev.node) set({ inspectorOpen: true });
    });
    const offUi = useUi.subscribe((state, prev) => {
      if (state.inspect && state.inspect !== prev.inspect && !state.inspectorOpen)
        set({ inspectorOpen: true });
    });
    return () => {
      offInspect();
      offUi();
    };
  }, [isLive, set]);
  const inspectorOpen = useUi((s) => s.inspectorOpen);
  useEffect(() => {
    if (theme === 'system') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
  }, [theme]);

  return (
    <div className={isLive ? 'app app-live' : 'app'}>
      <header className="app-header">
        <strong>mdb</strong>
        <select
          aria-label="session"
          value={sid ?? ''}
          onChange={(e) => {
            const s = sessions.find((x) => x.id === e.target.value);
            if (e.target.value === liveSid) setSession(liveSid);
            else if (s) setSession(s.id, s.home);
          }}
        >
          {liveSid && !sessions.some((s) => s.id === liveSid) && (
            <option value={liveSid}>{liveSid} · live</option>
          )}
          {sessions.map((s) => (
            <option key={s.id} value={s.id}>
              {s.id} · {s.id === liveSid ? 'live' : s.model}
            </option>
          ))}
        </select>
        <AddressBar />
        <span
          className="chip header-chip"
          title={
            liveSid ? `live session ${liveSid}` : 'server link (recorded project: no live model)'
          }
        >
          {liveSid ? 'live' : hello?.readOnly ? 'read-only' : 'writable'} · {link}
        </span>
        {liveSid && !isLive && (
          <button type="button" onClick={() => setSession(liveSid)} title="open the live debugger">
            live
          </button>
        )}
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
      {hello && !sessions.length && !liveSid ? (
        <div className="app-banner">
          <NotAvailable what="sessions" reason="This project holds no sessions." />
        </div>
      ) : (
        <>
          {!isLive && (
            // the recorded project's symbol list; the live session has its own Symbols tab
            <aside className="app-left">
              <ErrorBoundary name="Symbols">
                <Symbols />
              </ErrorBoundary>
            </aside>
          )}
          {isLive && liveSid ? (
            <main className="app-centre app-centre-live">
              <ErrorBoundary name="Live">
                <Live sid={liveSid} />
              </ErrorBoundary>
            </main>
          ) : (
            <main className="app-centre">
              <ErrorBoundary name="Timeline">
                <Timeline />
              </ErrorBoundary>
              <ErrorBoundary name="Graph">
                <Graph />
              </ErrorBoundary>
            </main>
          )}
          <aside
            className={isLive && !inspectorOpen ? 'app-right app-right-closed' : 'app-right'}
            data-testid="inspector-aside"
          >
            {isLive && (
              <button
                type="button"
                className="chip inspector-toggle"
                aria-expanded={inspectorOpen}
                aria-controls="inspector-panel"
                onClick={() => set({ inspectorOpen: !inspectorOpen })}
                title={inspectorOpen ? 'collapse the Inspector' : 'open the Inspector'}
              >
                {inspectorOpen ? 'Inspector ▸' : '◂ Inspector'}
              </button>
            )}
            {(!isLive || inspectorOpen) && (
              <div id="inspector-panel" className="inspector-panel">
                <ErrorBoundary name="Inspector">
                  <Inspector />
                </ErrorBoundary>
              </div>
            )}
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
