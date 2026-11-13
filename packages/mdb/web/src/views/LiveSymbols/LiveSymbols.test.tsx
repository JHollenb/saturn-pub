// The live Symbols view against op responses a real `mdb serve --tiny qwen2` returned
// (fixtures/, recorded by the script named in fixtures/commands.json).
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { slotFrom as slot, type Envelope as Env } from '../../state/opFixture';
import { LiveSymbols, siteTarget, type LiveSymbolsProps } from './LiveSymbols';
import { legendOf, parseStrip } from './strip';
import declare from './fixtures/declare.json';
import readsInterrupted from './fixtures/reads_interrupted.json';
import readsT2 from './fixtures/reads_t2.json';
import symbolsMargin from './fixtures/symbols_margin.json';
import symbolsNone from './fixtures/symbols_none.json';
import symbolsRefused from './fixtures/symbols_refused.json';

const act = {
  run: () => undefined,
  declare: () => undefined,
  measure: () => undefined,
  openRead: () => undefined,
  seek: () => undefined,
  interrupt: () => undefined,
};

function render(over: Partial<LiveSymbolsProps>): string {
  return renderToStaticMarkup(
    <LiveSymbols
      slot={{ state: 'idle' }}
      reads={{ state: 'idle' }}
      args={{}}
      where="t3.logits"
      busy={false}
      declared={[]}
      act={act}
      {...over}
    />,
  );
}

function rowOf(html: string, symbol: string): string {
  const start = html.indexOf(`data-symbol="${symbol}"`);
  expect(start, symbol).toBeGreaterThan(-1);
  return html.slice(start, html.indexOf('</tr>', start));
}

describe('LiveSymbols', () => {
  it('draws every binding with status, sites, lifetime, labelled mass and readers', () => {
    const html = render({ slot: slot(symbolsMargin, 'symbols consumer=M[4,9]') });
    const result = (symbolsMargin as unknown as Env).data.result as {
      bindings: { symbol: string; status: string }[];
    };
    for (const b of result.bindings) {
      const row = rowOf(html, b.symbol);
      expect(row).toContain(`data-status="${b.status}"`);
      expect(row).toContain('data-testid="status-chip"');
    }
    const ctx = rowOf(html, 'ctx');
    expect(ctx).toContain('cache.L1.v[row=2]'); // a definition site, clickable
    expect(ctx).toContain('title="seek t2.L1.v (asks first: it moves the session)"');
    expect(ctx).toContain('class="lifetime"');
    expect(ctx).toContain('data-ev="derived-exact"'); // summed mass: exact arithmetic
    expect(ctx).toContain('data-ev="measured"'); // each reader's mass: measured
    expect(ctx).toMatch(/reads/); // count of reads and mean mass per read
    expect(ctx).toContain('mean per read');
    // the column says what the mass is: a sum over heads and layers, not a probability
    expect(html).toContain('mass (Σ heads, layers)');
    expect(html).toContain('not a probability');
  });

  it('always shows the Unbound row with its parts and bounds', () => {
    const html = render({ slot: slot(symbolsMargin) });
    const unbound = rowOf(html, 'Unbound');
    expect(unbound).toContain('data-part="embedding_mlp"');
    expect(unbound).toContain('data-part="remainder"');
    expect(unbound).toContain('data-ev="bound"');
    // an op result without the row still shows it, unresolved
    const s = slot(symbolsMargin);
    if (s.state !== 'done') throw new Error('fixture');
    const { unbound: _, ...rest } = s.result;
    void _;
    const bare = render({ slot: { ...s, result: rest } });
    const row = rowOf(bare, 'Unbound');
    expect(row).toContain('data-ev="unresolved"');
    expect(row).toContain('the op result has no Unbound row');
  });

  it('an unresolved table says so and shows no number where the op gave none', () => {
    const html = render({ slot: slot(symbolsNone, 'symbols'), where: 't3.L1.p' });
    expect(html).toContain('data-testid="symbols-unresolved"');
    const t0 = rowOf(html, 't0');
    expect(t0).toContain('aria-label="unresolved: no scalar consumer"');
    // the cursor's cell (t3, layer 1) is marked in the lifetime strip
    expect(t0).toMatch(/life-cell [a-z-]+ life-cursor/);
    expect(rowOf(html, 'Unbound')).toContain('data-ev="unresolved"');
  });

  it('a cursor that moved since the table marks it stale', () => {
    const html = render({ slot: slot(symbolsMargin), where: 't4.L0.h' });
    expect(html).toContain('data-testid="symbols-stale"');
  });

  it('refused, running and interrupted states are stated, never an empty panel', () => {
    const refused = render({ slot: slot(symbolsRefused, 'symbols consumer=M[4,9]') });
    expect(refused).toContain('data-testid="symbols-refused"');
    expect(refused).toContain('t3.logits has not been computed at t3.L1.p');
    const running = render({ slot: { state: 'running', label: 'symbols' }, busy: true });
    expect(running).toContain('data-testid="symbols-running"');
    expect(running).toMatch(/<button[^>]*>interrupt<\/button>/);
    const interrupted = slot(readsInterrupted, 'reads t3');
    expect(interrupted.state).toBe('interrupted');
    const html = render({ slot: slot(symbolsMargin), reads: interrupted });
    expect(html).toContain('data-testid="reads-interrupted"');
    expect(html).toContain('interrupted</strong> at <code>t3.logits</code>');
    expect(render({})).toContain('data-testid="symbols-idle"');
  });

  it('measured verdicts of a source row appear beside its readers, quoting the op', () => {
    const html = render({ slot: slot(symbolsMargin), reads: slot(readsT2, 'reads t2') });
    const t2 = rowOf(html, 't2');
    const reads = (readsT2 as unknown as Env).data.result as {
      reads: { name: string; causal: { verdict: string; reason?: string } }[];
    };
    for (const r of reads.reads) {
      if (!t2.includes(`>${r.name}<`)) continue;
      expect(t2).toContain(`data-verdict="${r.causal.verdict}"`);
      if (r.causal.reason) expect(t2).toContain(r.causal.reason);
    }
    expect(t2).toContain('data-verdict=');
    // other rows were not measured: they say so
    expect(rowOf(html, 't0')).toContain('verdict not measured');
  });

  it('lists the declared symbols from the ledger', () => {
    expect(slot(declare).state).toBe('done');
    const html = render({
      declared: [{ name: 'ctx', definitions: ['t1', 'cache.L1.v[row=2]'], rows: [1, 2] }],
    });
    expect(html).toContain('ctx = t1, cache.L1.v[row=2]');
  });
});

describe('lifetime strip and sites', () => {
  it('parses the strip and uses the op legend when it gives one', () => {
    expect(parseStrip('t3:#. t4:?#')).toEqual([
      { row: 3, marks: ['#', '.'] },
      { row: 4, marks: ['?', '#'] },
    ]);
    expect(parseStrip('garbage')).toBeNull();
    expect(legendOf({ legend: { '#': 'read', '-': 'not part of the binding' } })).toEqual({
      '#': 'read',
      '-': 'not part of the binding',
    });
    expect(legendOf({ strip: 't3:##' })).toBeNull();
  });

  it('maps definition sites to occurrences seek accepts', () => {
    expect(siteTarget('t0.L*.k')).toBe('t0.L0.k');
    expect(siteTarget('t1')).toBe('t1.L0.k');
    expect(siteTarget('cache.L1.v[row=2]')).toBe('t2.L1.v');
    expect(siteTarget('t2 (uncovered part)')).toBe('t2.L0.k');
    expect(siteTarget('lm_head')).toBeNull();
  });
});
