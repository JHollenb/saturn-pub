// Test helper: a recorded op-route envelope (the views' fixtures/*.json, recorded from a real
// `mdb serve --tiny qwen2`) turned into what `useLive.run` hands a panel, and into its slot.
// Imported by tests only.
import type { OpResult } from '../api/types.gen';
import { slotOf, type Slot } from './inspect';
import type { OpRun } from './live';

export type Envelope = {
  state: 'resolved' | 'unresolved' | 'refused';
  reason?: string | null;
  data: OpResult;
};

export function runOf(body: unknown): OpRun {
  const env = body as Envelope;
  return {
    ok: true,
    op: env.data.op,
    seq: env.data.seq,
    result: env.data.result,
    state: env.state,
    reason: env.reason ?? null,
  };
}

export function slotFrom(body: unknown, label = 'op'): Slot {
  return slotOf(label, runOf(body));
}

/** The op result of a recorded envelope. */
export function resultOf<T = Record<string, unknown>>(body: unknown): T {
  return (body as Envelope).data.result as T;
}
