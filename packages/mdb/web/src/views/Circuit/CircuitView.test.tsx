// The circuit walk against `circuit` op responses a real `mdb serve --tiny qwen2` returned
// (fixtures/, recorded by the script named in fixtures/commands.json).
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { LiveLedger } from '../../api/types.gen';
import { isRec, type Rec } from '../../chrome/labelled';
import type { CircuitEntry, Slot } from '../../state/inspect';
import { resultOf, slotFrom } from '../../state/opFixture';
import { CircuitView, TestResult, type CircuitViewProps } from './CircuitView';
import {
  LABEL_W,
  LABEL_X,
  MIN_SCALE,
  NODE_R,
  ROW_H,
  flatten,
  layoutCircuit,
  naturalKey,
  scaleFor,
  walk,
  type CircuitLayout,
} from './layout';
import buildD2 from './fixtures/build_d2.json';
import buildK1 from './fixtures/build_k1.json';
import buildK2 from './fixtures/build_k2.json';
import buildO from './fixtures/build_o.json';
import buildRefused from './fixtures/build_refused.json';
import ledgerFx from './fixtures/ledger.json';
import showK2 from './fixtures/show_k2.json';
import testInterrupted from './fixtures/test_interrupted.json';
import testK2 from './fixtures/test_k2.json';
import testRefused from './fixtures/test_refused.json';

function bodyOf(fx: unknown): Rec {
  const body = (resultOf(fx) as { record: { body: Rec } }).record.body;
  return body;
}

function entryOf(fx: unknown): CircuitEntry {
  const r = resultOf(fx) as Rec & { id: number; tests?: Rec[] };
  return {
    id: r.id,
    body: bodyOf(fx),
    test: r.tests?.[r.tests.length - 1] ?? null,
    ablated: isRec(r.ablated) ? r.ablated : null,
  };
}

function positions(layout: CircuitLayout): Map<string, [number, number]> {
  return new Map(layout.placed.map((p) => [p.key, [p.x, p.y]]));
}

const act = {
  build: () => undefined,
  select: () => undefined,
  test: () => undefined,
  node: () => undefined,
  seek: () => undefined,
  overlay: () => undefined,
  interrupt: () => undefined,
};

const ledger = (ledgerFx as unknown as { data: LiveLedger }).data;

function render(over: Partial<CircuitViewProps>): string {
  return renderToStaticMarkup(
    <CircuitView
      ledger={ledger.circuits}
      circuits={new Map()}
      selected={null}
      node={null}
      focus={null}
      slot={{ state: 'idle' }}
      testSlot={{ state: 'idle' }}
      busy={false}
      overlay
      act={act}
      {...over}
    />,
  );
}

describe('circuit layout', () => {
  const k1 = bodyOf(buildK1);
  const k2 = bodyOf(buildK2);
  const d2 = bodyOf(buildD2);

  it('every circuit the op built keeps its +N more and remainder nodes', () => {
    for (const fx of [buildK1, buildK2, buildD2, buildO]) {
      const kinds = flatten(bodyOf(fx)).map((n) => n.kind);
      expect(kinds).toContain('more');
      expect(kinds).toContain('remainder');
      const html = renderToStaticMarkup(
        <CircuitView
          ledger={[]}
          circuits={new Map([[entryOf(fx).id, entryOf(fx)]])}
          selected={entryOf(fx).id}
          node={null}
          focus={null}
          slot={{ state: 'idle' }}
          testSlot={{ state: 'idle' }}
          busy={false}
          overlay
          act={act}
        />,
      );
      expect(html).toContain('data-kind="more"');
      expect(html).toContain('data-kind="remainder"');
    }
  });

  it('is deterministic: the same family gives the same positions', () => {
    expect(positions(layoutCircuit(k2, [k1, k2]))).toEqual(positions(layoutCircuit(k2, [k1, k2])));
  });

  it('a node keeps its position when K changes (top 1 → top 2) and when D changes', () => {
    const before = positions(layoutCircuit(k1, [k1]));
    const after = positions(layoutCircuit(k2, [k1, k2]));
    let shared = 0;
    for (const [key, xy] of before) {
      if (!after.has(key)) continue;
      shared += 1;
      expect(after.get(key), key).toEqual(xy);
    }
    expect(shared).toBeGreaterThanOrEqual(3); // consumer, embedding, +N more, remainder…
    // building the depth-2 circuit afterwards moves nothing the depth-3 one drew
    const k2Again = positions(layoutCircuit(k2, [k1, k2, d2]));
    expect(k2Again).toEqual(after);
    const d2Pos = positions(layoutCircuit(d2, [k1, k2, d2]));
    let sharedD = 0;
    for (const [key, xy] of d2Pos) {
      if (!after.has(key)) continue;
      sharedD += 1;
      expect(after.get(key), key).toEqual(xy);
    }
    expect(sharedD).toBeGreaterThan(3);
  });

  it('reserves each label box: no two labels overlap and no edge enters a label', () => {
    for (const fx of [buildK1, buildK2, buildD2, buildO]) {
      const layout = layoutCircuit(bodyOf(fx), [bodyOf(fx)]);
      const boxes = layout.placed.map((p) => ({
        id: p.node.id,
        x0: p.x - NODE_R,
        x1: p.x + LABEL_X + LABEL_W,
        y0: p.y - ROW_H / 2,
        y1: p.y + ROW_H / 2,
      }));
      for (const a of boxes)
        for (const b of boxes) {
          if (a.id >= b.id) continue;
          const apart = a.x1 <= b.x0 || b.x1 <= a.x0 || a.y1 <= b.y0 || b.y1 <= a.y0;
          expect(apart, `${a.id} / ${b.id}`).toBe(true);
        }
      for (const e of layout.edges) {
        // an edge runs from the end of its parent's label box to the child's node, so it spans
        // only the gap between two columns
        const from = e.from.x + LABEL_X + LABEL_W;
        const to = e.to.x - NODE_R;
        expect(to).toBeGreaterThan(from);
        for (const b of boxes) {
          const inGap = b.x1 <= from || b.x0 >= to;
          expect(inGap, `edge ${e.from.node.id}->${e.to.node.id} over ${b.id}`).toBe(true);
        }
      }
    }
  });

  it('fits the box down to the smallest readable scale, then scrolls', () => {
    expect(scaleFor('fit', 800, 1000)).toBe(1);
    expect(scaleFor('fit', 1000, 900)).toBeCloseTo(0.9);
    expect(scaleFor('fit', 1000, 300)).toBe(MIN_SCALE);
    expect(scaleFor(1.5, 1000, 300)).toBe(1.5);
    expect(scaleFor('fit', 1000, null)).toBe(1);
  });

  it('a +N more node is one node whatever N is', () => {
    const more = flatten(k1).find((n) => n.kind === 'more')!;
    expect(naturalKey(more)).toBe(`more|${more.address}`);
  });

  it('edges carry width from the exact contribution and a sign that is not colour alone', () => {
    const layout = layoutCircuit(k2, [k2]);
    expect(layout.edges.length).toBe(layout.placed.length - 1);
    for (const e of layout.edges) {
      if (e.width !== null) expect(e.width).toBeGreaterThanOrEqual(1);
      expect(['pos', 'neg', 'none']).toContain(e.sign);
    }
    expect(layout.edges.some((e) => e.sign === 'neg')).toBe(true);
    // a negative edge is dashed as well as coloured
    const e = entryOf(buildK2);
    const html = render({ circuits: new Map([[e.id, e]]), selected: e.id });
    expect(html).toMatch(/stroke-dasharray="6 3"[^>]*data-sign="neg"/);
  });

  it('the keyboard walk moves along edges and between siblings, and stops at the edges', () => {
    const layout = layoutCircuit(k2, [k2]);
    const root = layout.placed[0]!.node;
    expect(walk(layout, null, 'down')).toBe(root.id);
    expect(walk(layout, root.id, 'left')).toBe(root.id);
    const first = walk(layout, root.id, 'right')!;
    expect(layout.placed.find((p) => p.node.id === first)!.node.parent).toBe(root.id);
    const second = walk(layout, first, 'down')!;
    expect(second).not.toBe(first);
    expect(walk(layout, second, 'up')).toBe(first);
    expect(walk(layout, first, 'up')).toBe(first);
    expect(walk(layout, first, 'left')).toBe(root.id);
  });
});

describe('CircuitView', () => {
  it('lists the circuits of the run from the ledger', () => {
    const html = render({});
    for (const c of ledger.circuits) expect(html).toContain(`data-circuit="${c.id}"`);
    expect(html).toContain('data-testid="circuit-idle"');
  });

  it('draws the selected circuit with the closure, labelled values and a legend', () => {
    const e = entryOf(buildK2);
    const html = render({ circuits: new Map([[e.id, e]]), selected: e.id, node: 'n1' });
    expect(html).toContain('data-testid="circuit-graph"');
    expect(html).toContain('role="tree"');
    expect(html).toContain('data-testid="circuit-legend"');
    expect(html).toContain('closure');
    expect(html).toContain('data-ev="measured"');
    expect(html).toContain('data-ev="derived-exact"'); // the +N more Σ
    expect(html).toContain('data-testid="circuit-node-line"'); // the selected node
    expect(html).toContain('not tested in this page');
  });

  it('marks the nodes the last test ablated, by set, from circuit show', () => {
    const e = entryOf(showK2);
    expect(e.ablated).not.toBeNull();
    const html = render({ circuits: new Map([[e.id, e]]), selected: e.id });
    const retained = (e.ablated as { retained: string[] }).retained;
    for (const id of retained) {
      const at = html.indexOf(`data-node="${id}"`);
      expect(at, id).toBeGreaterThan(-1);
      expect(html.slice(at, html.indexOf('</g>', at))).toContain('✕R');
    }
    expect(html).toContain('✕C');
    expect(html).toContain('data-testid="test-result"');
  });

  it('a circuit test shows measured beside predicted, the exclusions, and a verdict quoting the op', () => {
    const test = (resultOf(testK2) as { test: Rec }).test;
    const html = renderToStaticMarkup(<TestResult test={test} />);
    expect(html).toContain('data-set="retained"');
    expect(html).toContain('data-set="complement"');
    expect(html).toContain('data-ev="derived-local"'); // predicted Δ
    expect(html).toContain('data-ev="measured"'); // measured Δ
    expect(html).toContain('data-ev="supplied"'); // the tolerance
    expect(html).toContain('data-testid="test-excluded"');
    expect(html).toContain('the embedding is ablated in neither set');
    expect(html).toContain(`>${String(test.verdict)}</span>`);
    expect(html).toContain(String(test.rule).replace(/</g, '&lt;').replace(/>/g, '&gt;'));
  });

  it('a test result missing a set says so instead of a blank row', () => {
    const html = renderToStaticMarkup(<TestResult test={{ verdict: 'x' }} />);
    expect(html).toMatch(/unresolved/);
    expect(html).toContain('retained');
  });

  it('the Test control is disabled with the M4 reason as its title', () => {
    const e = entryOf(buildK2);
    const html = render({ circuits: new Map([[e.id, e]]), selected: e.id, busy: false });
    const button = html.match(/<button[^>]*>circuit test \d+<\/button>/)?.[0] ?? '';
    expect(button).toContain('disabled');
    expect(button).toContain('title="circuit test is disabled: zero-ablating the complement says nothing on a real model (issue M4)"');
    expect(html).toContain('data-testid="test-disabled"');
  });

  it('refused, running and interrupted states are stated with their reason', () => {
    const refused = render({ slot: slotFrom(buildRefused, 'circuit M[4,9] top=2 depth=3') });
    expect(refused).toContain('data-testid="circuit-refused"');
    expect(refused).toContain('t3.logits has not been computed at t3.L1.p');
    const running: Slot = { state: 'running', label: 'circuit test 2' };
    const e = entryOf(buildK2);
    const busy = render({
      circuits: new Map([[e.id, e]]),
      selected: e.id,
      testSlot: running,
      busy: true,
    });
    expect(busy).toContain('data-testid="test-running"');
    expect(busy).toMatch(/<button[^>]*>interrupt<\/button>/);
    const interrupted = slotFrom(testInterrupted, 'circuit test 1');
    expect(interrupted.state).toBe('interrupted');
    const stopped = render({ circuits: new Map([[e.id, e]]), selected: e.id, testSlot: interrupted });
    expect(stopped).toContain('data-testid="test-interrupted"');
    const missing = slotFrom(testRefused, 'circuit test 9');
    expect(missing.state).toBe('refused');
    const html = render({ circuits: new Map([[e.id, e]]), selected: e.id, testSlot: missing });
    expect(html).toContain('no circuit 9');
  });
});
