// Reading labelled values out of op results (the JSON `python -m mdb.repl --json` prints). Their numbers are labelled
// `{value, authority, proof?}`; a vector consumer gives a summary (`{max_abs, norm}`, each
// labelled); a value an op could not give is `{state: unresolved|refused, reason}`. Any field may
// be missing or malformed while the ops change, so every reader here returns a labelled `Num`
// (unresolved with the reason when the field is absent), never a bare number and never zero.
import type { Num } from '../api/types.gen';
import { asNum } from './evidence';

export type Rec = Record<string, unknown>;

export function isRec(value: unknown): value is Rec {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** `value[a][b]...`, or undefined when any step is missing. */
export function at(value: unknown, ...path: (string | number)[]): unknown {
  let here: unknown = value;
  for (const key of path) {
    if (Array.isArray(here) && typeof key === 'number') here = here[key];
    else if (isRec(here) && typeof key === 'string') here = here[key];
    else return undefined;
  }
  return here;
}

export function str(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

export function strs(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : [];
}

export function recs(value: unknown): Rec[] {
  return Array.isArray(value) ? value.filter(isRec) : [];
}

/** A structural integer (count, id, row) or null; never a measurement. */
export function int(value: unknown): number | null {
  return typeof value === 'number' && Number.isInteger(value) ? value : null;
}

export function absent(reason: string, ev: 'unresolved' | 'refused' = 'unresolved'): Num {
  return {
    v: null,
    ev,
    reason,
    unit: null,
    proof: null,
    ptr: null,
    note: null,
    recorded_authority: null,
  };
}

/** One labelled value of an op result. */
export type Val =
  | { kind: 'num'; n: Num }
  | { kind: 'summary'; parts: [string, Num][] }
  | { kind: 'absent'; n: Num };

const SUMMARY_KEYS = ['max_abs', 'norm', 'mean', 'min', 'max'];

/** Read a labelled value: a number record, a summary of number records, or an explicit
 *  unresolved/refused state. Anything else is `absent` with `what` saying which field. */
export function valOf(value: unknown, what = 'value'): Val {
  if (value === undefined || value === null) {
    return { kind: 'absent', n: absent(`the op result has no ${what}`) };
  }
  const n = asNum(value);
  if (n) return { kind: 'num', n };
  if (isRec(value)) {
    const state = str(value.state);
    if (state === 'unresolved' || state === 'refused') {
      return { kind: 'absent', n: absent(str(value.reason) ?? state, state) };
    }
    const parts: [string, Num][] = [];
    for (const key of SUMMARY_KEYS) {
      const part = asNum(value[key]);
      if (part) parts.push([key, part]);
    }
    if (parts.length) return { kind: 'summary', parts };
    // `{value: {...summary}, bound}` wrappers carry the summary one level down
    if ('value' in value && !('authority' in value)) return valOf(value.value, what);
  }
  return { kind: 'absent', n: absent(`${what} is not a labelled value`, 'refused') };
}

/** A scalar view of a value for drawing (bar length, edge width): the number itself, or a
 *  summary's norm. Null when there is none (absent): nothing is drawn, never a zero. */
export function magnitude(val: Val): { v: number; signed: boolean } | null {
  if (val.kind === 'num') return val.n.v === null ? null : { v: val.n.v, signed: true };
  if (val.kind === 'summary') {
    const norm = val.parts.find(([k]) => k === 'norm') ?? val.parts[0];
    return norm && norm[1].v !== null ? { v: norm[1].v, signed: false } : null;
  }
  return null;
}

/** Strip a selector and the `(unlisted)` suffix so the address names a whole stop that `seek`
 *  accepts: `t3.L1.p[h0,src=1]` -> `t3.L1.p`, `t3.M[4,9]` -> `t3.logits`. Null when the address
 *  is not an occurrence of a token row (a weight, a cache cell). */
export function seekable(addr: string): string | null {
  const plain = addr.replace(/\s*\(unlisted\)$/, '').trim();
  const m = /^t(\d+)\.(.+)$/.exec(plain);
  if (!m) return null;
  const stop = (m[2] ?? '').split('[')[0] ?? '';
  if (stop === 'M' || stop === 'logits') return `t${m[1]}.logits`;
  if (!/^(L\d+\.)?[A-Za-z_]+$/.test(stop)) return null;
  return `t${m[1]}.${stop}`;
}

/** `(row, layer)` of an address (`t3.L1.A[h0]` -> [3, 1]); layer null for a global stop. */
export function cellOf(addr: string): [number, number | null] | null {
  const m = /^t(\d+)\.(?:L(\d+)\.)?/.exec(addr);
  if (!m) return null;
  return [Number(m[1]), m[2] === undefined ? null : Number(m[2])];
}

/** The source row a read names (`t3.L1.p[h0,src=2]` -> 2), else null. */
export function srcOf(addr: string): number | null {
  const m = /src=(\d+)/.exec(addr);
  return m ? Number(m[1]) : null;
}
