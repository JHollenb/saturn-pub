// Recorded mode for an engine-native project (records, no cuts): the stored bodies below are what
// a real `mdb serve --project` returned for a project written by
// `mdb engine tiny:qwen2 --project DIR --script FILE` (the /stored route, issue M20).
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { SessionSummary, StoredBody } from '../../api/types.gen';
import { StoredCircuit } from '../Circuit/CircuitView';
import { StoredSymbolTable } from '../LiveSymbols/LiveSymbols';
import { NoCuts, noCutsText } from './Recorded';
import circuitJson from './fixtures/circuit.json';
import sessionJson from './fixtures/session.json';
import tableJson from './fixtures/symbol_table.json';

const circuit = circuitJson as unknown as StoredBody;
const table = tableJson as unknown as StoredBody;
const session = sessionJson as unknown as SessionSummary;

describe('recorded symbol table', () => {
  const html = renderToStaticMarkup(<StoredSymbolTable result={table.body} />);

  it('draws every recorded binding', () => {
    const bindings = (table.body.bindings as { symbol: string }[]).map((b) => b.symbol);
    expect(bindings.length).toBeGreaterThan(0);
    for (const name of bindings) expect(html).toContain(`>${name}<`);
  });

  it('offers nothing that needs the live model', () => {
    expect(html).not.toContain('measure verdicts');
    expect(html).not.toContain('asks first: it moves the session');
    expect(html).not.toContain('open this read in the Circuits view');
    expect(html).toContain('as recorded: one page');
  });
});

describe('recorded circuit', () => {
  const html = renderToStaticMarkup(<StoredCircuit id={1} body={circuit.body} />);

  it('draws the same graph as the Live tab, rooted at the consumer', () => {
    expect(html).toContain('data-testid="stored-circuit"');
    expect(html).toContain('t7.M[2,3]');
    expect(html).toMatch(/data-testid="circuit-graph"><svg/);
    expect(html).toContain('data-kind="consumer" data-slot="0,0"');
    expect(html).toContain('arrows walk');
    expect(html).not.toContain('Enter asks to seek');
  });

  it('says why the remainders are not all checked', () => {
    expect(circuit.body.remainders_within).toBeNull();
    expect(html).toContain('remainders not all checked');
  });
});

describe('views that need cuts', () => {
  it('say "no retained cuts" with the session record count', () => {
    const html = renderToStaticMarkup(
      <NoCuts title="Timeline" what="timeline" session={session} />,
    );
    expect(html).toContain('<strong>no retained cuts</strong>');
    expect(html).toContain('this session holds 4 records and no state cuts');
    expect(noCutsText(undefined, 'graph')).toContain('holds records and no state cuts');
  });
});
