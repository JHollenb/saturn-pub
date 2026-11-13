import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { Num } from '../api/types.gen';
import { JsonTree } from './JsonTree';
import { Count, EvidenceMissing, NumView } from './NumView';

const base = {
  unit: null,
  proof: null,
  ptr: null,
  reason: null,
  note: null,
  recorded_authority: null,
};

describe('NumView evidence rule', () => {
  it('throws in development for a number without an evidence class', () => {
    expect(import.meta.env.DEV).toBe(true);
    const bare = { v: 3 } as unknown as Num;
    expect(() => renderToStaticMarkup(<NumView n={bare} />)).toThrow(EvidenceMissing);
    expect(() => renderToStaticMarkup(<NumView n={null} />)).toThrow(EvidenceMissing);
    const bogus = { v: 3, ev: 'guessed' } as unknown as Num;
    expect(() => renderToStaticMarkup(<NumView n={bogus} />)).toThrow(EvidenceMissing);
  });

  it('draws measured solid, derived hollow with a chip naming class and proof, supplied hatched', () => {
    const measured = renderToStaticMarkup(<NumView n={{ ...base, v: 0.5, ev: 'measured' }} />);
    expect(measured).toContain('data-ev="measured"');
    expect(measured).toContain('0.5');
    expect(measured).toContain('<circle');
    expect(measured).not.toContain('ev-chip');

    const derived = renderToStaticMarkup(
      <NumView n={{ ...base, v: 2, ev: 'derived-exact', proof: 'P-17' }} />,
    );
    expect(derived).toContain('fill="none"');
    expect(derived).toContain('ev-chip');
    expect(derived).toContain('derived-exact · P-17');

    const local = renderToStaticMarkup(<NumView n={{ ...base, v: 2, ev: 'derived-local' }} />);
    expect(local).toContain('derived-local');
    const bound = renderToStaticMarkup(<NumView n={{ ...base, v: 2, ev: 'bound' }} />);
    expect(bound).toContain('ev-chip');

    const supplied = renderToStaticMarkup(<NumView n={{ ...base, v: 1, ev: 'supplied' }} />);
    expect(supplied).toContain('<line');
  });

  it('never draws an unresolved number as zero', () => {
    const html = renderToStaticMarkup(
      <NumView n={{ ...base, v: null, ev: 'unresolved', reason: 'not reached' }} />,
    );
    expect(html).toContain('num-unresolved');
    expect(html).toContain('title="not reached"');
    expect(html).not.toMatch(/>\s*0\s*</);
    const refused = renderToStaticMarkup(
      <NumView n={{ ...base, v: null, ev: 'refused', reason: 'over cap' }} showReason />,
    );
    expect(refused).toContain('refused: over cap');
  });

  it('prints structural counts without a mark', () => {
    const html = renderToStaticMarkup(<Count n={42} unit="cuts" />);
    expect(html).toContain('42 cuts');
    expect(html).not.toContain('<svg');
  });

  it('refuses unlabelled numbers inside record bodies but prints structural ones', () => {
    const html = renderToStaticMarkup(
      <JsonTree
        value={{
          logit: 1.25,
          layer: 3,
          shape: [2, 4],
          effect: { ...base, v: 0.1, ev: 'measured' },
        }}
      />,
    );
    expect(html).toContain('number at &quot;logit&quot; has no evidence class');
    expect(html).not.toContain('1.25');
    expect(html).toContain('data-structural');
    expect(html).toContain('data-ev="measured"');
  });
});

// Static rule: only the evidence module formats numbers. Views print numbers through NumView
// (labelled) or Count (structural); no other file may call a number formatter.
const sources = import.meta.glob('../**/*.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

describe('number printing is confined to NumView', () => {
  it('has sources to scan', () => {
    expect(Object.keys(sources).length).toBeGreaterThan(20);
  });
  it.each(
    Object.entries(sources).filter(
      ([f]) => !f.includes('.test.') && !f.endsWith('/evidence.ts') && !f.endsWith('types.gen.ts'),
    ),
  )('%s uses no number formatter and prints no raw .v', (_file, text) => {
    expect(text).not.toMatch(/\.(toFixed|toPrecision|toExponential|toLocaleString)\(/);
    expect(text).not.toMatch(/Intl\.NumberFormat/);
    // `{x.v}` as JSX text (a prop such as `n={x.v}` passes it to NumView and is fine).
    expect(text).not.toMatch(/(?<!=)\{\s*[\w.?]+\.v\s*\}/);
  });
});
