// The lifetime strip of a binding (the `symbols` op): one `t<row>:<marks>` group per reader row,
// one mark per layer. The op states what each mark means (`lifetime.legend` when present);
// otherwise the meanings its docstring gives are used and said to be the default.
import { isRec, str, type Rec } from '../../chrome/labelled';

export interface StripRow {
  row: number;
  marks: string[];
}

/** `"t3:## t4:#?"` -> rows; null when the text is not a strip. */
export function parseStrip(text: string | null): StripRow[] | null {
  if (!text) return null;
  const rows: StripRow[] = [];
  for (const group of text.trim().split(/\s+/)) {
    const m = /^t(\d+):(.*)$/.exec(group);
    if (!m) return null;
    rows.push({ row: Number(m[1]), marks: [...(m[2] ?? '')] });
  }
  return rows;
}

/** Meanings of the marks when the op gives no legend (its docstring). */
export const DEFAULT_LEGEND: Record<string, string> = {
  '#': 'some head reads it with mass above min_mass',
  '.': 'computed, every read below min_mass',
  '?': 'not computed yet',
};

/** The op's legend (`{mark: meaning}` or `[{mark, meaning}]`) or null. */
export function legendOf(lifetime: unknown): Record<string, string> | null {
  if (!isRec(lifetime)) return null;
  const raw = lifetime.legend;
  if (isRec(raw)) {
    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(raw)) if (typeof v === 'string') out[k] = v;
    return Object.keys(out).length ? out : null;
  }
  if (Array.isArray(raw)) {
    const out: Record<string, string> = {};
    for (const item of raw as unknown[]) {
      if (!isRec(item)) continue;
      const mark = str((item as Rec).mark) ?? str((item as Rec).char);
      const meaning = str((item as Rec).meaning);
      if (mark && meaning) out[mark] = meaning;
    }
    return Object.keys(out).length ? out : null;
  }
  return null;
}

/** How a mark is drawn: a filled cell, a light cell, an empty dashed cell, or the mark itself
 *  (an unknown mark is never guessed at). */
export function markClass(mark: string, legend: Record<string, string>): string {
  const meaning = (legend[mark] ?? '').toLowerCase();
  if (mark === '#') return 'life-read';
  if (mark === '?' || meaning.includes('not computed')) return 'life-pending';
  if (mark === '.' ) return 'life-below';
  if (meaning.includes('not part') || meaning.includes('not in')) return 'life-none';
  if (meaning.includes('below')) return 'life-below';
  return 'life-other';
}
