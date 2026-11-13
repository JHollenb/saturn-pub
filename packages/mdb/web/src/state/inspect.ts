// What the inspection and prediction views of the live session asked for and got: one slot per
// panel (symbols, reads, circuit, whatif, compare, forecast), the circuit records seen in this
// page, the selected circuit and node, and the pending seek. Every value comes from an op this
// page sent through the one op route (`useLive.run`); nothing here computes a number.
//
// The slow ops (`whatif`, `compare`, `circuit test`, `reads` with verdicts) move the session and
// restore it. They run only on an explicit action, one at a time (the live store refuses a
// second op while one runs), and Interrupt stays available while they run.
import { create } from 'zustand';
import type { Refused } from '../api/types.gen';
import { at, int, isRec, str, type Rec } from '../chrome/labelled';
import { useLive, type OpRun } from './live';

/** One panel's op: what it asked (`label`, the REPL-style line) and how it ended. */
export type Slot =
  | { state: 'idle' }
  | { state: 'running'; label: string }
  | {
      state: 'done';
      label: string;
      seq: number;
      result: Rec;
      /** The envelope state: `unresolved` when a part of the result is. */
      partial: string | null;
    }
  | { state: 'refused'; label: string; reason: string; type: string | null }
  | { state: 'interrupted'; label: string; where: string }
  | { state: 'request-refused'; label: string; refused: Refused };

export const IDLE: Slot = { state: 'idle' };

/** Classify an op answer for a panel. */
export function slotOf(label: string, run: OpRun | null): Slot {
  if (run === null) {
    return {
      state: 'request-refused',
      label,
      refused: { refused: 'busy', detail: 'another op of this page is running' },
    };
  }
  if (!run.ok) return { state: 'request-refused', label, refused: run.refused };
  const r = run.result;
  if (typeof r.interrupted === 'string') return { state: 'interrupted', label, where: r.interrupted };
  if (isRec(r.error) || r.state === 'refused') {
    return {
      state: 'refused',
      label,
      reason: str(r.reason) ?? run.reason ?? 'refused',
      type: str(at(r, 'error', 'type')),
    };
  }
  return {
    state: 'done',
    label,
    seq: run.seq,
    result: r,
    partial: run.state === 'unresolved' ? (run.reason ?? 'partly unresolved') : null,
  };
}

export type Tab = 'pipeline' | 'symbols' | 'graph' | 'predict';
export type PanelKey = 'symbols' | 'reads' | 'circuit' | 'test' | 'whatif' | 'compare' | 'forecast';

/** A circuit record as the `circuit` op returns it (build / show): the sealed record's body. */
export interface CircuitEntry {
  id: number;
  body: Rec;
  /** The last `circuit test` result for it, when one ran in this page or `show` carried it. */
  test: Rec | null;
  /** Node ids the last test ablated, by set (`circuit show`'s `ablated`, or the test's own
   *  `retained.nodes` / `complement.nodes`). */
  ablated: Rec | null;
}

/** A read the Symbols view pointed at: reader `t3.L1.h0` of the binding over `rows`. */
export interface FocusRead {
  reader: string;
  binding: string;
  rows: number[];
}

/** A seek that waits for the user's confirmation: the session moves when it runs. */
export interface PendingSeek {
  target: string;
  /** Where it came from (a definition site, a circuit node, an architecture cell). */
  from: string;
}

interface InspectStore {
  tab: Tab;
  slots: Record<PanelKey, Slot>;
  /** Arguments the symbols table was last asked with. */
  symbolsArgs: Record<string, unknown>;
  circuits: ReadonlyMap<number, CircuitEntry>;
  selected: number | null;
  node: string | null;
  overlay: boolean;
  focus: FocusRead | null;
  seek: PendingSeek | null;
  /** Per view: authority spelled out as text chips beside every number (off: the mark carries
   *  it, the label is in the tooltip). */
  labels: Record<Tab, boolean>;
  set(
    patch: Partial<
      Pick<InspectStore, 'tab' | 'selected' | 'node' | 'overlay' | 'focus' | 'seek' | 'labels'>
    >,
  ): void;
  /** Run one op for a panel and keep its answer in that panel's slot. */
  exec(key: PanelKey, name: string, args: Record<string, unknown>, label: string): Promise<Slot>;
  symbols(args: Record<string, unknown>): Promise<void>;
  build(consumer: string, top: number, depth: number): Promise<void>;
  show(id: number): Promise<void>;
  test(id: number): Promise<void>;
  confirmSeek(): Promise<void>;
  /** Forget everything (another live session). */
  reset(): void;
}

function circuitEntry(result: Rec, previous: CircuitEntry | undefined): CircuitEntry | null {
  const id = int(result.id);
  const body = at(result, 'record', 'body');
  if (id === null || !isRec(body)) return null;
  const tests = Array.isArray(result.tests) ? result.tests.filter(isRec) : [];
  return {
    id,
    body,
    test: tests[tests.length - 1] ?? previous?.test ?? null,
    ablated: isRec(result.ablated) ? result.ablated : (previous?.ablated ?? null),
  };
}

/** `{retained: [ids], complement: [ids]}` from a test result's sets. */
function ablatedOf(test: Rec): Rec {
  return {
    retained: at(test, 'retained', 'nodes') ?? [],
    complement: at(test, 'complement', 'nodes') ?? [],
  };
}

const FRESH = {
  tab: 'pipeline' as Tab,
  slots: {
    symbols: IDLE,
    reads: IDLE,
    circuit: IDLE,
    test: IDLE,
    whatif: IDLE,
    compare: IDLE,
    forecast: IDLE,
  },
  symbolsArgs: {},
  circuits: new Map<number, CircuitEntry>(),
  selected: null,
  node: null,
  overlay: true,
  focus: null,
  seek: null,
  labels: { pipeline: false, symbols: false, graph: false, predict: false } as Record<Tab, boolean>,
};

export const useInspect = create<InspectStore>((set, get) => ({
  ...FRESH,
  set: (patch) => set(patch),
  reset: () =>
    set({ ...FRESH, circuits: new Map(), slots: { ...FRESH.slots }, labels: { ...FRESH.labels } }),
  async exec(key, name, args, label) {
    set((s) => ({ slots: { ...s.slots, [key]: { state: 'running', label } } }));
    const slot = slotOf(label, await useLive.getState().run(name, args));
    set((s) => ({ slots: { ...s.slots, [key]: slot } }));
    return slot;
  },
  async symbols(args) {
    set({ symbolsArgs: args });
    const words = Object.entries(args).map(([k, v]) => `${k}=${String(v)}`);
    await get().exec('symbols', 'symbols', args, ['symbols', ...words].join(' '));
  },
  async build(consumer, top, depth) {
    const slot = await get().exec(
      'circuit',
      'circuit',
      { target: consumer, top, depth },
      `circuit ${consumer} top=${top} depth=${depth}`,
    );
    if (slot.state !== 'done') return;
    const entry = circuitEntry(slot.result, undefined);
    if (!entry) return;
    const circuits = new Map(get().circuits);
    circuits.set(entry.id, entry);
    set({ circuits, selected: entry.id, node: null });
  },
  async show(id) {
    const slot = await get().exec('circuit', 'circuit', { target: 'show', id }, `circuit show ${id}`);
    if (slot.state !== 'done') return;
    const entry = circuitEntry(slot.result, get().circuits.get(id));
    if (!entry) return;
    const circuits = new Map(get().circuits);
    circuits.set(entry.id, entry);
    set({ circuits, selected: entry.id });
  },
  async test(id) {
    const slot = await get().exec('test', 'circuit', { target: 'test', id }, `circuit test ${id}`);
    if (slot.state !== 'done' || !isRec(slot.result.test)) return;
    const have = get().circuits.get(id);
    if (!have) return;
    const circuits = new Map(get().circuits);
    circuits.set(id, { ...have, test: slot.result.test, ablated: ablatedOf(slot.result.test) });
    set({ circuits });
  },
  async confirmSeek() {
    const pending = get().seek;
    if (!pending) return;
    set({ seek: null });
    await useLive.getState().run('seek', { target: pending.target });
  },
}));
