// The live session (`mdb serve --tiny|--model`): its frame and clock, the op in flight, the last
// op's outcome, the console log and per-port value summaries. Every action is one registry op
// (`POST /op`) or one REPL line (`POST /console`); the server runs them on one thread in order.
import { create } from 'zustand';
import { getView, postView } from '../api/client';
import type { LiveFrame, LiveTimeline, Omitted, Refused, ReduceResult } from '../api/types.gen';
import { useCursor } from './cursor';
import { lruSet } from './lru';
import type { ReduceEntry } from './reduce';

/** What the last op came to, from its REPL-equal result. */
export type Outcome =
  | { kind: 'ok'; op: string; seq: number; where: string | null; stopped: string | null }
  | { kind: 'unresolved'; op: string; seq: number; where: string | null; reason: string }
  | { kind: 'refused'; op: string; seq: number; reason: string; type: string | null }
  | { kind: 'interrupted'; op: string; seq: number; where: string };

export interface Summary {
  stats: ReduceEntry;
  topk: ReduceEntry | null;
}

export interface ConsoleLine {
  seq: number;
  line: string;
  text: string;
}

/** Items of a frame list fetched past what the frame showed (`GET .../live/items`). */
export interface Extra {
  items: Record<string, unknown>[];
  /** Offset of the next page; null when the list is complete. */
  next: number | null;
  refused: Refused | null;
}

const STOP_REASONS = new Set([
  'breakpoint',
  'watch',
  'advance',
  'reached',
  'budget',
  'end-of-context',
  'error',
  'no-breakpoints',
]);

/** Classify an op result (the JSON `python -m mdb.repl --json` prints). */
export function outcomeOf(op: string, seq: number, result: Record<string, unknown>): Outcome {
  if (typeof result.interrupted === 'string') {
    return { kind: 'interrupted', op, seq, where: result.interrupted };
  }
  const reason = typeof result.reason === 'string' ? result.reason : '';
  const where = typeof result.where === 'string' ? result.where : null;
  if (result.error && typeof result.error === 'object') {
    const type = (result.error as { type?: unknown }).type;
    return { kind: 'refused', op, seq, reason, type: typeof type === 'string' ? type : null };
  }
  if (result.state === 'unresolved') return { kind: 'unresolved', op, seq, where, reason };
  return { kind: 'ok', op, seq, where, stopped: STOP_REASONS.has(reason) ? reason : null };
}

export const MAX_SUMMARIES = 160;
export const MAX_CONSOLE = 200;

export function summaryKey(seq: number, addr: string): string {
  return `${seq}|${addr}`;
}

/** The frame's `more` entry for a list (`breakpoints`, `committed`, `upstream:<addr>`, ...). */
export function moreOf(frame: LiveFrame | null, field: string): Omitted | null {
  return frame?.more.find((m) => m.field === field) ?? null;
}

/** `[list, addr]` of a `more` field: `upstream:<addr>` names a port's upstream edits. */
export function listOf(field: string): [string, string | null] {
  return field.startsWith('upstream:') ? ['upstream', field.slice(9)] : [field, null];
}

/** Items the frame itself shows for a `more` field (the offset its first page starts at). */
export function shownIn(frame: LiveFrame, field: string): number {
  const [list, addr] = listOf(field);
  const edits = (frame.edits ?? {}) as Record<string, unknown[] | undefined>;
  switch (list) {
    case 'staged':
    case 'committed':
      return edits[list]?.length ?? 0;
    case 'breakpoints':
      return frame.breakpoints.length;
    case 'displays':
      return frame.displays.length;
    case 'checkpoints':
      return frame.checkpoints.length;
    case 'upstream':
      return frame.ports.find((p) => p.addr === addr)?.edit?.upstream.length ?? 0;
    default:
      return 0;
  }
}

interface LiveStore {
  sid: string | null;
  frame: LiveFrame | null;
  timeline: LiveTimeline | null;
  /** A refusal of a request itself (busy, network, not live, an unanswerable frame), not of an
   *  op. Shown beside the controls, which stay usable. */
  refused: Refused | null;
  /** Op or console line this page is waiting for. */
  running: string | null;
  last: Outcome | null;
  console: ConsoleLine[];
  summaries: ReadonlyMap<string, Summary>;
  /** The row whose top entries are shown besides the cursor row. */
  expanded: string | null;
  /** Rows on screen (`addr`), when the page reports them; null: every row counts as shown. */
  visible: ReadonlySet<string> | null;
  /** Pages of frame lists fetched past the frame, by `more` field (for the current frame). */
  extra: ReadonlyMap<string, Extra>;
  setSid(sid: string | null): void;
  refresh(): Promise<void>;
  op(name: string, args?: Record<string, unknown>): Promise<void>;
  command(line: string): Promise<void>;
  interrupt(): Promise<void>;
  expand(addr: string | null): void;
  /** Report a pipeline row entering (`true`) or leaving the screen. */
  setVisible(addr: string, on: boolean): void;
  /** Fetch the next page of a frame list the frame shortened. */
  more(field: string): Promise<void>;
  onEvent(data: Record<string, unknown>): void;
}

function path(sid: string, tail: string): string {
  return `/s/${encodeURIComponent(sid)}/${tail}`;
}

async function reduce(sid: string, addr: string, op: 'stats' | 'topk'): Promise<ReduceEntry> {
  const r = await postView('reduce', path(sid, 'reduce'), {
    pointer: { kind: 'occurrence', address: addr },
    op,
    params: op === 'topk' ? { k: 3 } : {},
  });
  return r.ok
    ? { state: 'ok', result: r.env.data as ReduceResult }
    : { state: 'refused', refused: r.refused };
}

export const useLive = create<LiveStore>((set, get) => {
  let pass = 0;
  /** Summaries of the computed ports on screen (every computed port when the page does not
   *  report rows), one request at a time; a newer frame or a newer pass stops this one. A
   *  value outside the cursor's layer costs the server a layer re-entry, so rows off screen
   *  are not asked for. */
  async function summarize(frame: LiveFrame) {
    const sid = get().sid;
    if (!sid) return;
    const mine = ++pass;
    for (const port of frame.ports) {
      if (!port.computed) continue;
      if (get().frame?.seq !== frame.seq || pass !== mine) return;
      const visible = get().visible;
      const wantTopk = port.cursor || port.addr === get().expanded;
      if (visible && !visible.has(port.addr) && !wantTopk) continue;
      const key = summaryKey(frame.seq, port.addr);
      const have = get().summaries.get(key);
      if (have && (have.topk || !wantTopk)) continue;
      const stats = have?.stats ?? (await reduce(sid, port.addr, 'stats'));
      const topk = wantTopk ? await reduce(sid, port.addr, 'topk') : null;
      set((s) => ({
        summaries: lruSet(s.summaries, key, { stats, topk }, MAX_SUMMARIES),
      }));
    }
  }

  let scheduled = false;
  function resummarize() {
    if (scheduled) return;
    scheduled = true;
    queueMicrotask(() => {
      scheduled = false;
      const frame = get().frame;
      if (frame) void summarize(frame);
    });
  }

  return {
    sid: null,
    frame: null,
    timeline: null,
    refused: null,
    running: null,
    last: null,
    console: [],
    summaries: new Map(),
    expanded: null,
    visible: null,
    extra: new Map(),
    setSid(sid) {
      if (sid === get().sid) return;
      set({
        sid,
        frame: null,
        timeline: null,
        last: null,
        refused: null,
        summaries: new Map(),
        extra: new Map(),
      });
      if (sid) void get().refresh();
    },
    async refresh() {
      const sid = get().sid;
      if (!sid) return;
      const [frame, timeline] = await Promise.all([
        getView('live', path(sid, 'live')),
        getView('live-timeline', path(sid, 'live/timeline')),
      ]);
      if (get().sid !== sid) return;
      // A refused frame keeps the last one (or none): controls and console stay usable.
      if (!frame.ok) return set({ refused: frame.refused });
      const current = get().frame;
      if (current && current.seq > frame.env.data.seq) return; // a newer frame already landed
      set({
        frame: frame.env.data,
        extra: current?.seq === frame.env.data.seq ? get().extra : new Map(),
        timeline: timeline.ok ? timeline.env.data : null,
        refused: timeline.ok ? null : timeline.refused,
      });
      const where = frame.env.data.where;
      const cursor = useCursor.getState();
      if (where && cursor.sid === sid) cursor.go(where);
      void summarize(frame.env.data);
    },
    async op(name, args = {}) {
      const sid = get().sid;
      if (!sid || get().running) return;
      set({ running: name, refused: null });
      const r = await postView('op', path(sid, 'op'), {
        op: name,
        args,
        request_id: `${name}-${Date.now()}`,
      });
      if (r.ok) {
        const d = r.env.data;
        set({ running: null, last: outcomeOf(d.op, d.seq, d.result) });
      } else {
        set({ running: null, refused: r.refused });
      }
      await get().refresh();
    },
    async command(line) {
      const sid = get().sid;
      if (!sid || get().running) return;
      set({ running: line.split(/\s+/)[0] || 'console', refused: null });
      const r = await postView('console', path(sid, 'console'), { line });
      if (r.ok) {
        const d = r.env.data;
        set((s) => ({
          running: null,
          console: [...s.console, { seq: d.seq, line: d.line, text: d.text }].slice(-MAX_CONSOLE),
        }));
      } else {
        set({ running: null, refused: r.refused });
      }
      await get().refresh();
    },
    async interrupt() {
      const sid = get().sid;
      if (!sid) return;
      const r = await postView('interrupt', path(sid, 'interrupt'), {});
      if (!r.ok) set({ refused: r.refused });
    },
    expand(addr) {
      set({ expanded: addr });
      resummarize();
    },
    setVisible(addr, on) {
      const visible = new Set(get().visible ?? []);
      if (on === visible.has(addr)) {
        if (get().visible === null) set({ visible });
        return;
      }
      if (on) visible.add(addr);
      else visible.delete(addr);
      set({ visible });
      if (on) resummarize();
    },
    async more(field) {
      const sid = get().sid;
      const frame = get().frame;
      if (!sid || !frame) return;
      const have = get().extra.get(field);
      if (have && have.next === null) return;
      const [list, addr] = listOf(field);
      const offset = have?.next ?? shownIn(frame, field);
      const r = await getView('live-items', path(sid, 'live/items'), { list, offset, addr });
      if (get().frame?.seq !== frame.seq) return; // the frame moved on: its lists are stale
      const extra = new Map(get().extra);
      if (r.ok) {
        extra.set(field, {
          items: [...(have?.items ?? []), ...r.env.data.items],
          next: r.env.data.next,
          refused: null,
        });
      } else {
        extra.set(field, {
          items: have?.items ?? [],
          next: have?.next ?? offset,
          refused: r.refused,
        });
      }
      set({ extra });
    },
    onEvent(data) {
      const seq = typeof data.seq === 'number' ? data.seq : null;
      if (seq === null || get().running) return; // this page's own op refreshes on its answer
      if ((get().frame?.seq ?? -1) >= seq) return;
      void get().refresh();
    },
  };
});
