// The Live view: the live store wired to `LiveView`. Shown when the cursor's session is the
// server's live session; the URL keeps the live cursor address like every other view.
import { useEffect, useMemo } from 'react';
import { useCursor } from '../../state/cursor';
import { useLive } from '../../state/live';
import { LiveView, type WatchRow } from './LiveView';

/** One IntersectionObserver for the pipeline rows: the store asks the server only for the
 *  values of rows on screen (a value outside the cursor's layer costs a layer re-entry). */
function useRowWatcher(): WatchRow | undefined {
  return useMemo(() => {
    if (typeof IntersectionObserver === 'undefined') return undefined;
    const addrs = new WeakMap<Element, string>();
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const addr = addrs.get(entry.target);
        if (addr) useLive.getState().setVisible(addr, entry.isIntersecting);
      }
    });
    return (addr: string, el: Element) => {
      addrs.set(el, addr);
      observer.observe(el);
      return () => {
        observer.unobserve(el);
        useLive.getState().setVisible(addr, false);
      };
    };
  }, []);
}

export function Live({ sid }: { sid: string }) {
  const s = useLive();
  const addr = useCursor((c) => c.addr);
  const watchRow = useRowWatcher();
  useEffect(() => {
    // rows report themselves when they reach the screen; until then none is asked for
    if (watchRow) useLive.setState({ visible: new Set() });
  }, [watchRow]);
  useEffect(() => {
    useLive.getState().setSid(sid);
  }, [sid]);
  const where = s.frame?.where ?? null;
  return (
    <section className="panel" aria-label="live">
      <h2>Live · {sid}</h2>
      <div className="panel-body">
        <LiveView
          frame={s.frame}
          timeline={s.timeline}
          running={s.running}
          last={s.last}
          refused={s.refused}
          summaries={s.summaries}
          console={s.console}
          expanded={s.expanded}
          extra={s.extra}
          watchRow={watchRow}
          urlAddr={addr && where && addr !== where ? addr : null}
          act={{
            op: (name, args) => void s.op(name, args),
            command: (line) => void s.command(line),
            interrupt: () => void s.interrupt(),
            expand: (a) => s.expand(a),
            more: (field) => void s.more(field),
          }}
        />
      </div>
    </section>
  );
}
