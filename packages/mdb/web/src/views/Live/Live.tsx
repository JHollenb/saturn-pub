// The Live view: the live store wired to `LiveView`. Shown when the cursor's session is the
// server's live session; the URL keeps the live cursor address like every other view.
import { useEffect } from 'react';
import { useCursor } from '../../state/cursor';
import { useLive } from '../../state/live';
import { LiveView } from './LiveView';

export function Live({ sid }: { sid: string }) {
  const s = useLive();
  const addr = useCursor((c) => c.addr);
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
          urlAddr={addr && where && addr !== where ? addr : null}
          act={{
            op: (name, args) => void s.op(name, args),
            command: (line) => void s.command(line),
            interrupt: () => void s.interrupt(),
            expand: (a) => s.expand(a),
          }}
        />
      </div>
    </section>
  );
}
