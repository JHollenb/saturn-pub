// `/` command line. It only navigates (cursor, session, lens, zoom, records); it never runs ops.
import { useState } from 'react';
import { useCursor } from '../state/cursor';
import { useUi } from '../state/ui';

export type Command =
  | { kind: 'go'; addr: string }
  | { kind: 'session'; sid: string }
  | { kind: 'lens'; lens: string }
  | { kind: 'zoom'; zoom: 'slice' | 'session' }
  | { kind: 'record'; id: string }
  | { kind: 'symbol'; id: string }
  | { kind: 'error'; message: string };

export function parseCommand(text: string): Command {
  const t = text.trim().replace(/^\//, '');
  const [head = '', ...rest] = t.split(/\s+/);
  const arg = rest.join(' ');
  switch (head) {
    case 'go':
      return arg ? { kind: 'go', addr: arg } : { kind: 'error', message: 'go needs an address' };
    case 'session':
    case 's':
      return arg
        ? { kind: 'session', sid: arg }
        : { kind: 'error', message: 'session needs an id' };
    case 'lens':
      return arg ? { kind: 'lens', lens: arg } : { kind: 'error', message: 'lens needs a name' };
    case 'zoom':
      return arg === 'slice' || arg === 'session'
        ? { kind: 'zoom', zoom: arg }
        : { kind: 'error', message: 'zoom is slice or session' };
    case 'record':
    case 'r':
      return arg ? { kind: 'record', id: arg } : { kind: 'error', message: 'record needs an id' };
    case 'symbol':
      return arg ? { kind: 'symbol', id: arg } : { kind: 'error', message: 'symbol needs an id' };
    case '':
      return { kind: 'error', message: 'empty command' };
    default:
      // Anything else is an address.
      return { kind: 'go', addr: t };
  }
}

export function CommandLine() {
  const open = useUi((s) => s.command);
  const set = useUi((s) => s.set);
  const go = useCursor((s) => s.go);
  const openDetail = useCursor((s) => s.open);
  const setSession = useCursor((s) => s.setSession);
  const [text, setText] = useState('');
  const [error, setError] = useState<string | null>(null);
  if (!open) return null;
  const run = () => {
    const c = parseCommand(text);
    if (c.kind === 'error') return setError(c.message);
    if (c.kind === 'go') go(c.addr);
    if (c.kind === 'session') setSession(c.sid);
    if (c.kind === 'lens') set({ lens: c.lens });
    if (c.kind === 'zoom') set({ zoom: c.zoom });
    if (c.kind === 'record') {
      openDetail({ record: c.id });
      set({ drawerOpen: true });
    }
    if (c.kind === 'symbol') openDetail({ symbol: c.id });
    setText('');
    setError(null);
    set({ command: false });
  };
  return (
    <div className="command" role="dialog" aria-label="command line">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
      >
        <input
          autoFocus
          className="mono"
          style={{ width: '100%' }}
          value={text}
          placeholder="address | go ADDR | session ID | lens NAME | zoom slice|session | record ID | symbol ID"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') set({ command: false });
          }}
        />
      </form>
      {error && <div className="muted">{error}</div>}
    </div>
  );
}
