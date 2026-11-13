import { describe, expect, it } from 'vitest';
import { slotOf } from '../state/inspect';
import { formatValue } from './evidence';
import { cellOf, magnitude, seekable, srcOf, valOf } from './labelled';

describe('labelled op values', () => {
  it('reads a number record with its authority', () => {
    const v = valOf({ authority: 'derived-local', proof: 'P14', value: 1.5 });
    expect(v.kind).toBe('num');
    if (v.kind === 'num') expect([v.n.v, v.n.ev]).toEqual([1.5, 'derived-local']);
  });

  it('reads a summary (vector consumer), and the {value, bound} wrapper around one', () => {
    const summary = {
      max_abs: { authority: 'measured', value: 2 },
      norm: { authority: 'measured', value: -4 },
    };
    const v = valOf(summary);
    expect(v.kind).toBe('summary');
    expect(magnitude(v)).toEqual({ v: -4, signed: false });
    expect(valOf({ value: summary, bound: { authority: 'bound', value: 1 } }).kind).toBe('summary');
  });

  it('an unresolved, refused or missing value is absent with its reason, never zero', () => {
    const refused = valOf({ state: 'refused', reason: 'flips a token' });
    expect(refused.kind).toBe('absent');
    if (refused.kind === 'absent') expect([refused.n.ev, refused.n.reason]).toEqual(['refused', 'flips a token']);
    const missing = valOf(undefined, 'delta');
    if (missing.kind === 'absent') expect(missing.n.reason).toBe('the op result has no delta');
    expect(magnitude(missing)).toBeNull();
    expect(valOf('text').kind).toBe('absent');
  });

  it('maps node addresses to stops seek accepts', () => {
    expect(seekable('t3.L1.p[h0,src=1]')).toBe('t3.L1.p');
    expect(seekable('t3.M[4,9]')).toBe('t3.logits');
    expect(seekable('t3.M[4,9] (unlisted)')).toBe('t3.logits');
    expect(seekable('t3.L1.A[h0]')).toBe('t3.L1.A');
    expect(seekable('cache.L1.v[row=2]')).toBeNull();
    expect(cellOf('t3.L1.A[h0]')).toEqual([3, 1]);
    expect(cellOf('t3.logits')).toEqual([3, null]);
    expect(srcOf('t3.L1.p[h0,src=2]')).toBe(2);
    expect(srcOf('t3.L1.o')).toBeNull();
  });
});

describe('the number format of the views', () => {
  it('4 significant digits, exponent below 1e-3, a real minus sign', () => {
    expect(formatValue(-0.114013671875)).toBe('\u22120.114');
    expect(formatValue(0.16448974609375)).toBe('0.1645');
    expect(formatValue(1.388e-17)).toBe('1.388e\u221217');
    expect(formatValue(-0.00030954)).toBe('\u22123.095e\u22124');
    expect(formatValue(-3)).toBe('\u22123');
    expect(formatValue(12.5)).toBe('12.5');
  });
});

describe('slotOf', () => {
  const base = { ok: true as const, op: 'x', seq: 1, state: 'resolved' as const, reason: null };
  it('classifies done, refused, interrupted and request-refused answers', () => {
    expect(slotOf('x', { ...base, result: { state: 'resolved' } }).state).toBe('done');
    expect(
      slotOf('x', { ...base, state: 'unresolved', reason: 'part', result: { state: 'unresolved' } }),
    ).toMatchObject({ state: 'done', partial: 'part' });
    expect(
      slotOf('x', { ...base, result: { state: 'refused', reason: 'no', error: { type: 'refused' } } }),
    ).toMatchObject({ state: 'refused', reason: 'no', type: 'refused' });
    expect(slotOf('x', { ...base, result: { interrupted: 't3.L0.p' } })).toMatchObject({
      state: 'interrupted',
      where: 't3.L0.p',
    });
    expect(slotOf('x', null).state).toBe('request-refused');
    expect(
      slotOf('x', { ok: false, op: 'x', refused: { refused: 'over_cap', detail: 'big' } }),
    ).toMatchObject({ state: 'request-refused' });
  });
});
