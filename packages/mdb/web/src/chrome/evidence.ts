// Evidence classes and how each is drawn. Shape/pattern only: hue is reserved for roles.
import type { Ev, Num } from '../api/types.gen';

export const EV_CLASSES: readonly Ev[] = [
  'measured',
  'derived-exact',
  'derived-local',
  'bound',
  'supplied',
  'unresolved',
  'refused',
] as const;

export type MarkStyle = 'solid' | 'hollow' | 'hatched' | 'dashed-empty' | 'refused';

export function markStyle(ev: Ev): MarkStyle {
  switch (ev) {
    case 'measured':
      return 'solid';
    case 'derived-exact':
    case 'derived-local':
    case 'bound':
      return 'hollow';
    case 'supplied':
      return 'hatched';
    case 'unresolved':
      return 'dashed-empty';
    case 'refused':
      return 'refused';
  }
}

/** Classes whose number is shown with a chip naming the class (and its proof id). */
export function needsChip(ev: Ev): boolean {
  return markStyle(ev) === 'hollow';
}

export function isEv(value: unknown): value is Ev {
  return typeof value === 'string' && (EV_CLASSES as readonly string[]).includes(value);
}

/** True when `value` is a labelled number record (`{v, ev, ...}`): `v` is a number or null (an
 *  unresolved/refused number). Stored data that merely carries `ev` with a non-number `v` or a
 *  `vs` (`{ev: "measured", v: "hello"}`, `{ev: "measured", vs: "x"}`) is not a number. */
export function isNum(value: unknown): value is Num {
  if (typeof value !== 'object' || value === null || !('ev' in value)) return false;
  if (!isEv((value as { ev: unknown }).ev)) return false;
  if ('v' in value) {
    const v = (value as { v: unknown }).v;
    return typeof v === 'number' || v === null;
  }
  return false;
}

/** A labelled number in either record shape: the view shape `{v, ev, ...}` or the version-2
 *  record shape `{value, authority, unit?, proof?}`. Returns it as a `Num`, or null when `value`
 *  is neither (a bare number is not labelled). */
export function asNum(value: unknown): Num | null {
  if (isNum(value)) return value;
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  const r = value as Record<string, unknown>;
  if (!isEv(r.authority) || !('value' in r)) return null;
  if (r.value !== null && (typeof r.value !== 'number' || typeof r.value === 'boolean'))
    return null;
  return {
    v: r.value as number | null,
    ev: r.authority,
    unit: typeof r.unit === 'string' ? r.unit : null,
    proof: typeof r.proof === 'string' ? r.proof : null,
    ptr: typeof r.ptr === 'string' ? r.ptr : null,
    reason: typeof r.reason === 'string' ? r.reason : null,
    note: typeof r.note === 'string' ? r.note : null,
    recorded_authority: typeof r.recorded_authority === 'string' ? r.recorded_authority : null,
  };
}

/** The server's note on a number (e.g. why a bare legacy float is class `supplied`), or null. */
export function numNote(n: Num): string | null {
  return n.note || null;
}

/** The one number format of the views: 4 significant digits, exponent form below 1e-3 (and at
 *  1e6 and above), and a real minus sign (U+2212) so signs line up in tabular columns. The full
 *  value goes in the element's title (see NumView). */
export function formatValue(v: number): string {
  return minus(plainValue(v));
}

function plainValue(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (Number.isInteger(v) && Math.abs(v) < 1e15) return v.toString();
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
  return Number(v.toPrecision(4)).toString();
}

/** ASCII hyphen-minus to the minus sign (in the mantissa and the exponent). */
export function minus(text: string): string {
  return text.replace(/-/g, '\u2212');
}
