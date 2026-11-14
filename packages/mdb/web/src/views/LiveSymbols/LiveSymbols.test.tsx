// The live Symbols view against op responses a real `mdb serve --tiny qwen2` returned
// (fixtures/, recorded by the script named in fixtures/commands.json).
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { slotFrom as slot, type Envelope as Env } from '../../state/opFixture';
import { LiveSymbols, midTrunc, siteTarget, type LiveSymbolsProps } from './LiveSymbols';
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

/** A binding's compact row and, when its disclosure is open, its sub-row (sites, readers). */
function blockOf(html: string, symbol: string): string {
  const start = html.indexOf(`data-symbol="${symbol}"`);
  expect(start, symbol).toBeGreaterThan(-1);
  const next = html.indexOf('data-symbol=', start + 1);
  return html.slice(start, next > -1 ? next : html.indexOf('</tbody>', start));
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
    // the compact row: name, status, lifetime, mass, reads, mean per read, to-consumer
    const ctxRow = rowOf(html, 'ctx');
    expect(ctxRow).toContain('class="lifetime"');
    expect(ctxRow).toContain('data-ev="derived-exact"'); // summed mass: exact arithmetic
    // four right-aligned number columns, each labelled for the stacked narrow layout
    expect(ctxRow.match(/<td class="n" data-label="[^"]+">/g)?.length).toBe(4);
    expect(ctxRow).not.toContain('class="site');
    // the first binding's disclosure is open: sites and one line per reader
    expect(ctxRow).toContain('aria-expanded="true"');
    const ctx = blockOf(html, 'ctx');
    expect(ctx).toContain('cache.L1.v[row=2]'); // a definition site, clickable
    expect(ctx).toContain(
      'title="cache.L1.v[row=2]: seek t2.L1.v (asks first: it moves the session)"',
    );
    expect(ctx).toContain('data-ev="measured"'); // each reader's mass: measured
    expect(ctx).toMatch(/data-reader="t3\.L1\.h0"/);
    // the column says what the mass is: a sum over heads and layers, not a probability
    expect(html).toContain('mass (Σ heads, layers)');
    expect(html).toContain('not a probability');
    expect(html).toContain('mean per read');
    // a binding that is neither first nor under the cursor starts closed
    expect(rowOf(html, 't0')).toContain('aria-expanded="false"');
    expect(blockOf(html, 't0')).not.toContain('data-reader=');
    // ... and the binding under the cursor (t3 at t3.logits) starts open
    expect(rowOf(html, 't3')).toContain('aria-expanded="true"');
  });

  it('always shows the Unbound row with its parts and bounds', () => {
    const html = render({ slot: slot(symbolsMargin) });
    const unbound = blockOf(html, 'Unbound');
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
    // the row whose verdicts were measured opens by itself
    expect(rowOf(html, 't2')).toContain('aria-expanded="true"');
    const t2 = blockOf(html, 't2');
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
    expect(blockOf(html, 'ctx')).toContain('not measured');
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

  it('shortens long addresses in the middle (the full text goes in the title)', () => {
    expect(midTrunc('t3.L1.p[h0,src=1]')).toBe('t3.L1.p[h0,src=1]');
    const long = 't3.L1.p[h0,src=0|h0,src=1|h1,src=0|h1,src=3]';
    const short = midTrunc(long, 20);
    expect(short).toHaveLength(20);
    expect(short.startsWith('t3.L1.p[h')).toBe(true);
    expect(short.endsWith('src=3]')).toBe(true);
    expect(short).toContain('…');
  });

  it('lists named symbols from transplants with their measured delta and binding here', () => {
    const env = structuredClone(symbolsMargin) as unknown as Env;
    const result = env.data.result as Record<string, unknown>;
    result.named = [
      { symbol: 'subj', source: 't@subject.L1.v[kv0]', consumer: 't5.M[4,9]',
        delta: { value: -0.014526, authority: 'measured' }, verdict: 'effect', record: 'r1',
        here: { address: 't1.L1.v[kv0]', row: 1, how: 'its declaring prompt' } },
      { symbol: 'far', source: 't1.L0.v[kv0]', consumer: 't5.M[4,9]',
        delta: { value: 0.5, authority: 'measured' }, verdict: 'effect', record: 'r2',
        here: { state: 'refused', reason: 'absolute row t1; the prefix differs' } },
    ];
    const html = render({ slot: slot(env as never, 'symbols consumer=M[4,9]') });
    const named = html.slice(html.indexOf('data-testid="symbols-named"'));
    expect(named).toContain('data-named="subj"');
    expect(named).toContain('t@subject.L1.v[kv0]');
    expect(named).toContain('t1.L1.v[kv0]');
    expect(named).toContain('-0.014526');
    expect(named).toContain('refused: absolute row t1; the prefix differs');
    expect(render({ slot: slot(symbolsMargin, 'symbols') })).not.toContain('symbols-named');
  });

  it('maps definition sites to occurrences seek accepts', () => {
    expect(siteTarget('t0.L*.k')).toBe('t0.L0.k');
    expect(siteTarget('t1')).toBe('t1.L0.k');
    expect(siteTarget('cache.L1.v[row=2]')).toBe('t2.L1.v');
    expect(siteTarget('t2 (uncovered part)')).toBe('t2.L0.k');
    expect(siteTarget('lm_head')).toBeNull();
  });
});
