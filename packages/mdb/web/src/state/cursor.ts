// The one shared cursor: an occurrence address, its history, and the URL fragment it mirrors.
// A click in any view calls `go`; every view re-renders from `addr`. No view keeps its own
// notion of a selected item.
import { create } from 'zustand';
import { formatFragment, parseFragment, type Detail } from '../api/address';

export interface LocationPort {
  read(): string;
  write(hash: string): void;
  listen(onChange: () => void): () => void;
}

export interface CursorState {
  sid: string | null;
  addr: string | null;
  /** Record whose comparison detail is open (URL `?record=`). */
  record: string | null;
  /** Symbol whose row is expanded (URL `?symbol=`). */
  symbol: string | null;
  history: string[];
  index: number;
  go(addr: string): void;
  /** Open or close the record / symbol detail; leaves the address alone. */
  open(patch: Partial<Detail>): void;
  setSession(sid: string, addr?: string | null, detail?: Partial<Detail>): void;
  back(): void;
  forward(): void;
  syncFromLocation(): void;
}

const MAX_HISTORY = 200;

function moveTo(
  state: Pick<CursorState, 'addr' | 'history' | 'index'>,
  addr: string,
): Pick<CursorState, 'addr' | 'history' | 'index'> | null {
  if (addr === state.addr) return null;
  const next = [...state.history.slice(0, state.index + 1), addr].slice(-MAX_HISTORY);
  return { addr, history: next, index: next.length - 1 };
}

export function createCursorStore(location: LocationPort) {
  const store = create<CursorState>((set, get) => ({
    sid: null,
    addr: null,
    record: null,
    symbol: null,
    history: [],
    index: -1,
    go(addr) {
      const move = moveTo(get(), addr);
      if (move) set(move);
    },
    open(patch) {
      const next: Partial<Detail> = {};
      if (patch.record !== undefined && patch.record !== get().record) next.record = patch.record;
      if (patch.symbol !== undefined && patch.symbol !== get().symbol) next.symbol = patch.symbol;
      if (Object.keys(next).length) set(next);
    },
    setSession(sid, addr = null, detail = {}) {
      set({
        sid,
        addr,
        record: detail.record ?? null,
        symbol: detail.symbol ?? null,
        history: addr ? [addr] : [],
        index: addr ? 0 : -1,
      });
    },
    back() {
      const { history, index } = get();
      if (index > 0) set({ index: index - 1, addr: history[index - 1] ?? null });
    },
    forward() {
      const { history, index } = get();
      if (index < history.length - 1) set({ index: index + 1, addr: history[index + 1] ?? null });
    },
    syncFromLocation() {
      const loc = parseFragment(location.read());
      if (!loc) return;
      const { sid } = get();
      if (loc.sid !== sid) {
        get().setSession(loc.sid, loc.addr, loc);
        return;
      }
      // One write, so the URL is rewritten at most once for a pasted link.
      set({
        ...(loc.addr ? moveTo(get(), loc.addr) : null),
        record: loc.record,
        symbol: loc.symbol,
      });
    },
  }));
  // Store -> URL. The fragment always equals the cursor, so a copied URL reopens this view.
  store.subscribe((state, prev) => {
    if (
      !state.sid ||
      (state.sid === prev.sid &&
        state.addr === prev.addr &&
        state.record === prev.record &&
        state.symbol === prev.symbol)
    )
      return;
    const hash = formatFragment({
      sid: state.sid,
      addr: state.addr,
      record: state.record,
      symbol: state.symbol,
    });
    if (location.read() !== hash) location.write(hash);
  });
  // URL -> store (back/forward buttons, pasted links).
  location.listen(() => store.getState().syncFromLocation());
  return store;
}

const browser: LocationPort = {
  read: () => (typeof window === 'undefined' ? '' : window.location.hash),
  write: (hash) => {
    if (typeof window !== 'undefined') window.history.pushState(null, '', hash);
  },
  listen: (onChange) => {
    if (typeof window === 'undefined') return () => undefined;
    window.addEventListener('hashchange', onChange);
    window.addEventListener('popstate', onChange);
    return () => {
      window.removeEventListener('hashchange', onChange);
      window.removeEventListener('popstate', onChange);
    };
  },
};

export const useCursor = createCursorStore(browser);
