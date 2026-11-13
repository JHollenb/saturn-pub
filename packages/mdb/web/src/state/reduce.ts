// Server-side reductions cached by (pointer digest, op, canonical params).
import { create } from 'zustand';
import { lruSet } from './lru';
import { postView } from '../api/client';
import type { PointerRef, Refused, ReduceResult } from '../api/types.gen';

export type ReduceOp = 'stats' | 'topk' | 'slice' | 'histogram' | 'tile' | 'project';
export type ReduceEntry =
  | { state: 'loading' }
  | { state: 'ok'; result: ReduceResult }
  | { state: 'refused'; refused: Refused };

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) =>
      a < b ? -1 : a > b ? 1 : 0,
    );
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonical(v)}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

export function reduceKey(sid: string, digest: string, op: ReduceOp, params: object): string {
  return `${sid}|${digest}|${op}|${canonical(params)}`;
}

export const MAX_REDUCE_ENTRIES = 64;

interface ReduceState {
  entries: ReadonlyMap<string, ReduceEntry>;
  run(sid: string, pointer: PointerRef, op: ReduceOp, params?: object): Promise<void>;
}

function put(
  set: (fn: (s: ReduceState) => Partial<ReduceState>) => void,
  key: string,
  entry: ReduceEntry,
) {
  set((s) => ({ entries: lruSet(s.entries, key, entry, MAX_REDUCE_ENTRIES) }));
}

export const useReduce = create<ReduceState>((set, get) => ({
  entries: new Map(),
  async run(sid, pointer, op, params = {}) {
    const key = reduceKey(sid, pointer.digest, op, params);
    const existing = get().entries.get(key);
    if (existing && existing.state !== 'refused') return;
    if (!pointer.spec) {
      put(set, key, {
        state: 'refused',
        refused: { refused: 'frontier', detail: pointer.reason },
      });
      return;
    }
    put(set, key, { state: 'loading' });
    const r = await postView('reduce', `/s/${encodeURIComponent(sid)}/reduce`, {
      pointer: pointer.spec,
      op,
      params,
    });
    const entry: ReduceEntry = r.ok
      ? { state: 'ok', result: r.env.data }
      : { state: 'refused', refused: r.refused };
    put(set, key, entry);
  },
}));
