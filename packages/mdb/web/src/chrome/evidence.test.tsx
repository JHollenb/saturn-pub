import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { EV_CLASSES, asNum, isNum } from './evidence';
import { EvidenceMark } from './EvidenceMark';
import { JsonTree } from './JsonTree';
import { RefusedView } from './NotAvailable';
import { axisStatus, tickInferred } from '../views/Timeline/axis';
import type { Num } from '../api/types.gen';

describe('labelled numbers in record bodies', () => {
  it('renders the version-2 {value, authority} shape with the same evidence mark', () => {
    const html = renderToStaticMarkup(
      <JsonTree
        value={{
          margin: { value: 0.25, authority: 'measured', unit: 'logit' },
          gap: { value: -1.5e-7, authority: 'derived-exact', proof: 'P-3' },
          prior: { value: 2, authority: 'supplied' },
        }}
      />,
    );
    expect(html).toContain('data-ev="measured"');
    expect(html).toContain('data-ev="derived-exact"');
    expect(html).toContain('derived-exact · P-3');
    expect(html).toContain('data-ev="supplied"');
    expect(html).not.toContain('refused');
    expect(html).toContain('0.25');
  });

  it('draws the {v, ev} shape and the {value, authority} shape identically', () => {
    const a = renderToStaticMarkup(
      <JsonTree
        value={{ v: 3.5, ev: 'measured', unit: null, proof: null, ptr: null, reason: null }}
      />,
    );
    const b = renderToStaticMarkup(<JsonTree value={{ value: 3.5, authority: 'measured' }} />);
    expect(a).toContain('data-ev="measured"');
    expect(b).toContain('data-ev="measured"');
    expect(a.match(/<circle/g)?.length).toBe(b.match(/<circle/g)?.length);
  });

  it('still refuses a bare number outside a structural key, and shows why', () => {
    const html = renderToStaticMarkup(
      <JsonTree value={{ score: 0.7, nested: { value: 1.5 }, layer: 2 }} />,
    );
    expect(html).not.toContain('0.7');
    expect(html).not.toContain('1.5');
    expect(html).toContain('number at &quot;score&quot; has no evidence class');
    expect(html).toContain('number at &quot;value&quot; has no evidence class');
    expect(html).toContain('data-structural');
  });

  it('isNum needs a numeric (or null) v: stored look-alikes are not numbers', () => {
    expect(isNum({ ev: 'measured', v: 'hello' })).toBe(false);
    expect(isNum({ ev: 'measured', v: { x: 1 } })).toBe(false);
    expect(isNum({ ev: 'measured', vs: 'oops' })).toBe(false);
    expect(isNum({ ev: 'bogus', v: 1 })).toBe(false);
    expect(isNum({ ev: 'measured', v: 1.5 })).toBe(true);
    expect(isNum({ ev: 'unresolved', v: null, reason: 'x' })).toBe(true);
    expect(asNum({ ev: 'measured', v: 'hello' })).toBeNull();
    const html = renderToStaticMarkup(<JsonTree value={{ a: { ev: 'measured', v: 'hello' } }} />);
    expect(html).not.toContain('data-ev="measured"');
  });

  it('asNum rejects a record whose authority is not an evidence class', () => {
    expect(asNum({ value: 1, authority: 'guessed' })).toBeNull();
    expect(asNum({ value: '1', authority: 'measured' })).toBeNull();
    expect(asNum({ value: 1 })).toBeNull();
    expect(asNum({ value: null, authority: 'unresolved', reason: 'not run' })?.ev).toBe(
      'unresolved',
    );
  });
});

describe('accessibility', () => {
  it.each(EV_CLASSES)(
    'the evidence mark for %s has an accessible name containing the class',
    (ev) => {
      const html = renderToStaticMarkup(<EvidenceMark ev={ev} />);
      expect(html).toMatch(new RegExp(`aria-label="[^"]*${ev}[^"]*"`));
      expect(html).toContain('role="img"');
    },
  );
});

describe('refusal messages', () => {
  it.each([
    'host_not_allowed',
    'origin_not_allowed',
    'over_limit',
    'bad_params',
    'web_build_absent',
  ])('says in words what %s means and still shows the code', (code) => {
    const html = renderToStaticMarkup(<RefusedView what="server" refused={{ refused: code }} />);
    expect(html).toContain(`<code>${code}</code>`);
    expect(html.replace(/<[^>]+>/g, '').length).toBeGreaterThan(code.length + 40);
  });
  it('keeps the detail of an unknown code', () => {
    const html = renderToStaticMarkup(
      <RefusedView what="x" refused={{ refused: 'weird', detail: 'because' }} />,
    );
    expect(html).toContain('because');
  });
});

describe('depth axis honesty', () => {
  const num = (ev: Num['ev'], v: number | null, unit = 'layers'): Num => ({
    v,
    ev,
    unit,
    proof: null,
    ptr: null,
    reason: v === null ? 'not recorded' : null,
    note: null,
    recorded_authority: null,
  });
  it('reads the layer status from the capabilities layers number', () => {
    expect(axisStatus(null)).toBe('stated');
    expect(axisStatus(num('measured', 2))).toBe('stated');
    expect(axisStatus(num('derived-local', 2, 'layers (at least)'))).toBe('inferred');
    expect(axisStatus(num('unresolved', null))).toBe('unknown');
  });
  it('draws a tick hollow exactly when the server marks it inferred', () => {
    const tick = { i: 1, label: 'L0', frag: 'L0', port: 'h', detail: null, addressable: true };
    expect(tickInferred({ ...tick, inferred: true })).toBe(true);
    expect(tickInferred({ ...tick, inferred: false })).toBe(false);
  });
});

describe('supplied numbers with a note', () => {
  it('shows the note in the tooltip and, in detail views, beside the value (hatched mark)', () => {
    const body = { value: 0.5, authority: 'supplied', note: 'bare float in a legacy record' };
    const n = asNum(body)!;
    const html = renderToStaticMarkup(<JsonTree value={{ w: body }} />);
    expect(n.ev).toBe('supplied');
    expect(html).toContain('title="supplied · bare float in a legacy record · value 0.5"');
    expect(html).toContain('> bare float in a legacy record</span>');
    expect(html).toContain('<line');
  });
});
