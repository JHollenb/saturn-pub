// Server hello, the session list and the active session's capabilities.
import { create } from 'zustand';
import { getView } from '../api/client';
import type { Capabilities, Hello, Refused, SessionSummary } from '../api/types.gen';

interface SessionState {
  hello: Hello | null;
  sessions: SessionSummary[];
  caps: Capabilities | null;
  capsFor: string | null;
  refused: Refused | null;
  boot(): Promise<void>;
  loadCaps(sid: string): Promise<void>;
}

export const useSession = create<SessionState>((set, get) => ({
  hello: null,
  sessions: [],
  caps: null,
  capsFor: null,
  refused: null,
  async boot() {
    const hello = await getView('hello', '/hello');
    if (!hello.ok) return set({ refused: hello.refused });
    const sessions = await getView('sessions', '/sessions');
    if (!sessions.ok) return set({ hello: hello.env.data, refused: sessions.refused });
    set({ hello: hello.env.data, sessions: sessions.env.data, refused: null });
  },
  async loadCaps(sid) {
    if (get().capsFor === sid) return;
    set({ capsFor: sid, caps: null });
    const caps = await getView('capabilities', `/s/${encodeURIComponent(sid)}/capabilities`);
    if (get().capsFor !== sid) return;
    if (caps.ok) set({ caps: caps.env.data, refused: null });
    else set({ refused: caps.refused });
  },
}));
