// The live session (`mdb serve --tiny|--model`): its frame and clock, the op in flight, the last
// op's outcome, the console log and per-port value summaries. Every action is one registry op
// (`POST /op`) or one REPL line (`POST /console`); the server runs them on one thread in order.
import { create } from 'zustand';
import { getView, postView } from '../api/client';
import type { LiveFrame, LiveTimeline, Refused, ReduceResult } from '../api/types.gen';
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

interface LiveStore {
  sid: string | null;
  frame: LiveFrame | null;
  timeline: LiveTimeline | null;
  /** A refusal of a request itself (busy, network, not live), not of an op. */
  refused: Refused | null;
  /** Op or console line this page is waiting for. */
  running: string | null;
  last: Outcome | null;
  console: ConsoleLine[];
  summaries: ReadonlyMap<string, Summary>;
  /** The row whose top entries are shown besides the cursor row. */
  expanded: string | null;
  setSid(sid: string | null): void;
  refresh(): Promise<void>;
  op(name: string, args?: Record<string, unknown>): Promise<void>;
  command(line: string): Promise<void>;
  interrupt(): Promise<void>;
  expand(addr: string | null): void;
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
  /** Summaries of every computed port of the frame, one request at a time; stops when a newer
   *  frame arrives. */
  async function summarize(frame: LiveFrame) {
    const sid = get().sid;
    if (!sid) return;
    for (const port of frame.ports) {
      if (!port.computed) continue;
      if (get().frame?.seq !== frame.seq) return;
      const key = summaryKey(frame.seq, port.addr);
      const have = get().summaries.get(key);
      const wantTopk = port.cursor || port.addr === get().expanded;
      if (have && (have.topk || !wantTopk)) continue;
      const stats = have?.stats ?? (await reduce(sid, port.addr, 'stats'));
      const topk = wantTopk ? await reduce(sid, port.addr, 'topk') : null;
      set((s) => ({
        summaries: lruSet(s.summaries, key, { stats, topk }, MAX_SUMMARIES),
      }));
    }
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
    setSid(sid) {
      if (sid === get().sid) return;
      set({ sid, frame: null, timeline: null, last: null, refused: null, summaries: new Map() });
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
      if (!frame.ok) return set({ refused: frame.refused });
      const current = get().frame;
      if (current && current.seq > frame.env.data.seq) return; // a newer frame already landed
      set({
        frame: frame.env.data,
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
      const frame = get().frame;
      if (frame) void summarize(frame);
    },
    onEvent(data) {
      const seq = typeof data.seq === 'number' ? data.seq : null;
      if (seq === null || get().running) return; // this page's own op refreshes on its answer
      if ((get().frame?.seq ?? -1) >= seq) return;
      void get().refresh();
    },
  };
});
