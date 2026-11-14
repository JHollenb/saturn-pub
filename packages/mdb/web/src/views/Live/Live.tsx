// The Live view: the live store and the inspection store wired to `LiveView` and its panes
// (Symbols, Circuits, Predict) and the architecture strip. Shown when the cursor's session is the
// server's live session; the URL keeps the live cursor address like every other view. Every
// action is one registry op through the one op route.
import { useEffect, useMemo } from 'react';
import { isRec, str } from '../../chrome/labelled';
import { useCursor } from '../../state/cursor';
import { useInspect, type PendingSeek } from '../../state/inspect';
import { useLive } from '../../state/live';
import { ArchStrip } from '../Arch/ArchStrip';
import { CircuitView } from '../Circuit/CircuitView';
import { flatten } from '../Circuit/layout';
import { LiveSymbols } from '../LiveSymbols/LiveSymbols';
import { Predict } from '../Predict/Predict';
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

/** The line that confirms a seek before it runs: `seek` moves the session. */
export function SeekLine({
  seek,
  where,
  busy,
  onConfirm,
  onCancel,
}: {
  seek: PendingSeek | null;
  where: string | null;
  busy: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  if (!seek) return null;
  return (
    <div className="seek-line" role="alertdialog" aria-label="confirm seek" data-testid="seek-line">
      <span>
        <code>seek {seek.target}</code> moves the session from <code>{where ?? 'no run'}</code> to
        it (forward with <code>until</code>, back by layers, or by replaying a stored recipe);
        asked from {seek.from}.
      </span>{' '}
      <button
        type="button"
        disabled={busy}
        onClick={onConfirm}
        ref={(el) => {
          if (el && document.activeElement?.closest('[data-own-keys]')) el.focus();
        }}
      >
        seek
      </button>{' '}
      <button type="button" onClick={onCancel}>
        cancel
      </button>
    </div>
  );
}

export function Live({ sid }: { sid: string }) {
  const s = useLive();
  const ins = useInspect();
  const addr = useCursor((c) => c.addr);
  const watchRow = useRowWatcher();
  useEffect(() => {
    // rows report themselves when they reach the screen; until then none is asked for
    if (watchRow) useLive.setState({ visible: new Set() });
  }, [watchRow]);
  useEffect(() => {
    useLive.getState().setSid(sid);
    useInspect.getState().reset();
  }, [sid]);
  const where = s.frame?.where ?? null;
  const busy = !!s.running;
  const ledger = s.ledger;
  const interrupt = () => void s.interrupt();
  const seek = (target: string, from: string) => ins.set({ seek: { target, from } });
  const entry = ins.selected !== null ? (ins.circuits.get(ins.selected) ?? null) : null;
  const nodeAddrs = useMemo(
    () => (entry && ins.overlay ? flatten(entry.body).map((n) => n.address) : []),
    [entry, ins.overlay],
  );
  const circuitPorts = useMemo(
    () => new Set(nodeAddrs.map((a) => a.split('[')[0] ?? a)),
    [nodeAddrs],
  );
  const port = s.frame?.layer !== null && s.frame?.layer !== undefined
    ? (() => {
        const stop = s.frame.stop as { port?: unknown } | null;
        const name = isRec(stop) ? str(stop.port) : null;
        return name ? `L${s.frame.layer}.${name}` : null;
      })()
    : null;
  const forecasts = ledger?.forecasts ?? [];
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
          tab={ins.tab}
          labels={ins.labels[ins.tab]}
          onLabels={(on) => ins.set({ labels: { ...ins.labels, [ins.tab]: on } })}
          forecasts={forecasts}
          circuitPorts={circuitPorts}
          seekLine={
            <SeekLine
              seek={ins.seek}
              where={where}
              busy={busy}
              onConfirm={() => void ins.confirmSeek()}
              onCancel={() => ins.set({ seek: null })}
            />
          }
          strip={
            s.frame && (
              <ArchStrip
                frame={s.frame}
                timeline={s.timeline}
                circuit={nodeAddrs}
                busy={busy}
                seek={seek}
              />
            )
          }
          panes={{
            symbols: (
              <LiveSymbols
                slot={ins.slots.symbols}
                reads={ins.slots.reads}
                args={ins.symbolsArgs}
                where={where}
                busy={busy}
                declared={ledger?.symbols ?? []}
                act={{
                  run: (args) => void ins.symbols(args),
                  declare: (name, addresses) =>
                    void ins
                      .exec('symbols', 'symbol', { name, addresses }, `symbol ${name} = ${addresses}`)
                      .then((slot) => {
                        if (slot.state === 'done') void ins.symbols(useInspect.getState().symbolsArgs);
                      }),
                  measure: (source, consumer) =>
                    void ins.exec(
                      'reads',
                      'reads',
                      consumer ? { source, consumer } : { source },
                      `reads ${source}${consumer ? ` consumer=${consumer}` : ''}`,
                    ),
                  openRead: (focus) => ins.set({ focus, tab: 'graph' }),
                  seek,
                  interrupt,
                }}
              />
            ),
            graph: (
              <CircuitView
                ledger={ledger?.circuits ?? []}
                runs={ledger?.runs ?? []}
                circuits={ins.circuits}
                selected={ins.selected}
                node={ins.node}
                focus={ins.focus}
                slot={ins.slots.circuit}
                testSlot={ins.slots.test}
                busy={busy}
                overlay={ins.overlay}
                act={{
                  build: (consumer, top, depth) => void ins.build(consumer, top, depth),
                  select: (id) => {
                    if (ins.circuits.has(id)) ins.set({ selected: id, node: null });
                    else void ins.show(id);
                  },
                  test: (id, ref) => void ins.test(id, ref),
                  node: (id) => ins.set({ node: id }),
                  seek,
                  overlay: (on) => ins.set({ overlay: on }),
                  interrupt,
                }}
              />
            ),
            predict: (
              <Predict
                whatif={ins.slots.whatif}
                compare={ins.slots.compare}
                forecast={ins.slots.forecast}
                forecasts={forecasts}
                checkpoints={s.frame?.checkpoints ?? []}
                port={port}
                where={where}
                busy={busy}
                act={{
                  whatif: (target, value, op, consumer) =>
                    void ins.exec(
                      'whatif',
                      'whatif',
                      { target, value, op, consumer },
                      `whatif ${target} = ${value} ${op} consumer=${consumer}`,
                    ),
                  stage: (target, value, op) => void s.op('set', { target, value, op }),
                  compare: (a, b, consumer) =>
                    void ins.exec(
                      'compare',
                      'compare',
                      consumer ? { a, b, consumer } : { a, b },
                      `compare ${a} | ${b}${consumer ? ` consumer=${consumer}` : ''}`,
                    ),
                  forecast: (target) =>
                    void ins.exec(
                      'forecast',
                      'forecast',
                      target ? { target } : {},
                      `forecast${target ? ` ${target}` : ''}`,
                    ),
                  verify: (ids) =>
                    void ins.exec('forecast', 'verify', ids ? { ids } : {}, `verify${ids ? ` ${ids.join(' ')}` : ''}`),
                  interrupt,
                }}
              />
            ),
          }}
          act={{
            op: (name, args) => void s.op(name, args),
            command: (line) => void s.command(line),
            interrupt,
            expand: (a) => s.expand(a),
            more: (field) => void s.more(field),
            tab: (t) => ins.set({ tab: t }),
            forecast: (target) =>
              void ins.exec('forecast', 'forecast', { target }, `forecast ${target}`),
            verify: (ids) => void ins.exec('forecast', 'verify', { ids }, `verify ${ids.join(' ')}`),
          }}
        />
      </div>
    </section>
  );
}
