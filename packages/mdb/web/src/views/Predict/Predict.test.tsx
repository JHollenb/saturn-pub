// Predictions in flight against `whatif`, `compare`, `forecast` and `verify` answers a real
// `mdb serve --tiny qwen2` returned, and the frames and ledgers around them (fixtures/, recorded
// by the script named in fixtures/commands.json).
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { LiveFrame, LiveLedger } from '../../api/types.gen';
import { valOf, type Rec } from '../../chrome/labelled';
import { resultOf, slotFrom } from '../../state/opFixture';
import { LiveView, type LiveViewProps } from '../Live/LiveView';
import { CompareResult, Predict, WhatIfResult, forecastsOf, type PredictProps } from './Predict';
import { SplitBar } from './SplitBar';
import compareCheckpoint from './fixtures/compare_checkpoint.json';
import compareEdits from './fixtures/compare_edits.json';
import compareRefused from './fixtures/compare_refused.json';
import forecastNext from './fixtures/forecast_next.json';
import forecastRefused from './fixtures/forecast_refused.json';
import frameForecasts from './fixtures/frame_forecasts.json';
import frameVerified from './fixtures/frame_verified.json';
import ledgerOpen from './fixtures/ledger_open.json';
import ledgerVerified from './fixtures/ledger_verified.json';
import verifyFirst from './fixtures/verify_first.json';
import whatifInterrupted from './fixtures/whatif_interrupted.json';
import whatifMargin from './fixtures/whatif_margin.json';
import whatifPort from './fixtures/whatif_port.json';
import whatifRefused from './fixtures/whatif_refused.json';
import whatifT4 from './fixtures/whatif_t4.json';

const act = {
  whatif: () => undefined,
  stage: () => undefined,
  compare: () => undefined,
  forecast: () => undefined,
  verify: () => undefined,
  interrupt: () => undefined,
};

const open = (ledgerOpen as unknown as { data: LiveLedger }).data.forecasts;
const verified = (ledgerVerified as unknown as { data: LiveLedger }).data.forecasts;

function render(over: Partial<PredictProps>): string {
  return renderToStaticMarkup(
    <Predict
      whatif={{ state: 'idle' }}
      compare={{ state: 'idle' }}
      forecast={{ state: 'idle' }}
      forecasts={[]}
      checkpoints={[]}
      port="L1.m"
      where="t3.logits"
      busy={false}
      act={act}
      {...over}
    />,
  );
}

function whatif(fx: unknown): string {
  return renderToStaticMarkup(<WhatIfResult result={resultOf(fx)} busy={false} act={act} />);
}

describe('what-if', () => {
  it('shows linear, exact with its bound, native and native − exact, each with its authority', () => {
    const html = whatif(whatifMargin);
    expect(html).toContain('sealed linear Δ');
    expect(html).toContain('data-ev="derived-local"'); // linear (P14)
    expect(html).toContain('data-ev="derived-exact"'); // exact
    expect(html).toContain('data-ev="bound"'); // its bound
    expect(html).toContain('data-ev="measured"'); // native
    expect(html).toContain('data-ev="supplied"'); // the edit's value
    expect(html).toContain('gradient of the consumer times the edit displacement; not the effect');
    expect(html).toContain(' within');
  });

  it('draws the error split as a labelled stacked bar whose terms are all listed', () => {
    const split = resultOf<{ split: { terms: { name: string }[] } }>(whatifMargin).split;
    const html = renderToStaticMarkup(<SplitBar split={split} />);
    for (const t of split.terms) expect(html).toContain(`data-term="${t.name}"`);
    expect(html).toContain('native delta = sum of the terms (float64)');
    expect(html).toContain('<svg');
  });

  it('states the session was restored and offers staging as a separate, explicit set', () => {
    const html = whatif(whatifMargin);
    expect(html).toContain('data-testid="whatif-session"');
    expect(html).toContain('The session was restored: unchanged at');
    expect(html).toMatch(/<button[^>]*>stage this edit<\/button>/);
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>stage this edit/);
  });

  it('a port consumer gives summaries (max_abs, norm), never one number', () => {
    const html = whatif(whatifPort);
    expect(html).toContain('max_abs');
    expect(html).toContain('norm');
  });

  it('a refused linear term (the edit flips a token) is shown with its reason', () => {
    const html = whatif(whatifT4);
    expect(html).toContain('the edit flips the sampled token of t4');
    expect(html).toContain('data-ev="refused"');
    const linear = resultOf<{ prediction: Rec }>(whatifT4).prediction.linear;
    expect(valOf(linear).kind).toBe('absent');
  });

  it('refused, running and interrupted what-ifs say so, never an empty panel', () => {
    const refused = render({ whatif: slotFrom(whatifRefused, 'whatif L1.zz') });
    expect(refused).toContain('data-testid="whatif-refused"');
    expect(refused).toContain('unknown port');
    const running = render({ whatif: { state: 'running', label: 'whatif L1.m' }, busy: true });
    expect(running).toContain('data-testid="whatif-running"');
    expect(running).toMatch(/<button[^>]*>interrupt<\/button>/);
    const stopped = render({ whatif: slotFrom(whatifInterrupted, 'whatif L1.m') });
    expect(stopped).toContain('data-testid="whatif-interrupted"');
    expect(stopped).toContain('t3.logits');
    expect(render({})).toContain('data-testid="whatif-idle"');
  });
});

describe('compare', () => {
  it('shows base, A, B, A+B with effects and the interaction, all measured', () => {
    const html = renderToStaticMarkup(<CompareResult result={resultOf(compareEdits)} />);
    for (const arm of ['base', 'A', 'B', 'A+B']) expect(html).toContain(`data-arm="${arm}"`);
    expect(html).toContain('(A+B) - A - B + base');
    expect(html).toContain('data-testid="compare-interaction"');
    expect(html).not.toContain('not run');
    expect(html).toContain('session unchanged');
  });

  it('a checkpoint side has no A+B: the arm says not run and the interaction gives the reason', () => {
    const html = renderToStaticMarkup(<CompareResult result={resultOf(compareCheckpoint)} />);
    expect(html).toContain('not run');
    expect(html).toContain('A+B needs both sides to be edits');
    expect(html).toContain('data-ev="refused"');
  });

  it('a refused compare quotes the op', () => {
    const html = render({ compare: slotFrom(compareRefused, 'compare nope') });
    expect(html).toContain('data-testid="compare-refused"');
    expect(html).toContain('no checkpoint');
  });
});

describe('forecasts', () => {
  it('the forecast op seals a record and refuses another row with its reason', () => {
    expect(slotFrom(forecastNext).state).toBe('done');
    const html = render({ forecast: slotFrom(forecastRefused, 'forecast t0.L0.p') });
    expect(html).toContain('data-testid="forecast-refused"');
    expect(html).toContain('forecasts stay in the cursor');
  });

  it('lists the open forecasts hollow (derived, bound) with a verify button each', () => {
    const html = render({ forecasts: open });
    for (const f of open) expect(html).toContain(`data-forecast="${f.id}"`);
    expect(html).toContain('verify reached (4 open)');
    expect(html).toContain('not measured');
    expect(html).not.toMatch(/data-ev="measured"/);
  });

  it('a verified forecast shows the native value, residual ≤ bound and the verdict', () => {
    expect(slotFrom(verifyFirst).state).toBe('done');
    const html = render({ forecasts: verified });
    const at = html.indexOf('data-forecast="1"');
    const row = html.slice(at, html.indexOf('</tr>', at));
    expect(row).toContain('data-status="closed"');
    expect(row).toContain('data-ev="measured"');
    expect(row).toContain('data-ev="bound"');
    expect(row).toContain('within-bound');
    expect(forecastsOf(verified, 't3.L0.a').map((f) => f.id)).toEqual([1]);
  });
});

describe('pipeline forecast column', () => {
  const act2 = {
    op: () => undefined,
    command: () => undefined,
    interrupt: () => undefined,
    expand: () => undefined,
    more: () => undefined,
    forecast: () => undefined,
    verify: () => undefined,
  };
  function pipeline(frame: unknown, forecasts: LiveLedger['forecasts'] | undefined): string {
    const p: LiveViewProps = {
      frame: (frame as { data: LiveFrame }).data,
      timeline: null,
      running: null,
      last: null,
      refused: null,
      summaries: new Map(),
      console: [],
      expanded: null,
      urlAddr: null,
      act: act2,
      forecasts,
    };
    return renderToStaticMarkup(<LiveView {...p} />);
  }
  function cell(html: string, id: number): string {
    const at = html.indexOf(`class="forecast" data-forecast="${id}"`);
    expect(at, `forecast ${id}`).toBeGreaterThan(-1);
    return html.slice(at, html.indexOf('</td>', at));
  }

  it('predicted values are hollow; a computed port offers verify; an uncomputed one waits', () => {
    const html = pipeline(frameForecasts, open);
    const a = cell(html, 1); // t3.L0.a, computed
    expect(a).toContain('data-ev="derived-exact"');
    expect(a).toMatch(/<button[^>]*>verify<\/button>/);
    const p = cell(html, 2); // t3.L0.p, not computed yet
    expect(p).toContain('awaiting native');
    expect(p).toContain('data-ev="bound"');
    expect(html).toMatch(/<button[^>]*title="forecast t3\.L0\.q[^"]*"[^>]*>forecast<\/button>/);
  });

  it('after verify the native value sits beside the prediction with residual and bound', () => {
    const html = pipeline(frameVerified, verified);
    const a = cell(html, 1);
    expect(a).toContain('data-ev="derived-exact"');
    expect(a).toContain('data-ev="measured"');
    expect(a).toContain('resid');
    expect(a).toContain('within-bound');
  });

  it('hides the authority chips unless the view shows labels; the mark and title keep them', () => {
    const p: LiveViewProps = {
      frame: (frameForecasts as unknown as { data: LiveFrame }).data,
      timeline: null,
      running: null,
      last: null,
      refused: null,
      summaries: new Map(),
      console: [],
      expanded: null,
      urlAddr: null,
      act: act2,
      forecasts: open,
      labels: false,
      onLabels: () => undefined,
    };
    const off = renderToStaticMarkup(<LiveView {...p} />);
    expect(off).toContain('class="live labels-off"');
    expect(off).toContain('data-testid="view-legend"');
    expect(off).toMatch(/<input type="checkbox"\/> show labels/);
    // the derived prediction's mark names its authority (and proof, when there is one)
    expect(off).toMatch(/aria-label="derived-exact[^"]* evidence"/);
    const on = renderToStaticMarkup(<LiveView {...p} labels />);
    expect(on).toContain('class="live labels-on"');
    expect(on).toMatch(/<input type="checkbox" checked=""\/> show labels/);
  });

  it('lists the frame\'s displays with where each reads at the cursor', () => {
    const frame = (frameForecasts as unknown as { data: LiveFrame }).data;
    const html = renderToStaticMarkup(
      <LiveView
        frame={{ ...frame, displays: [{ id: 1, address: 'L0.h' }, { id: 2, address: 't3.L0.a' }] }}
        timeline={null}
        running={null}
        last={null}
        refused={null}
        summaries={new Map()}
        console={[]}
        expanded={null}
        urlAddr={null}
        act={act2}
      />,
    );
    expect(html).toContain('data-testid="displays"');
    expect(html).toContain('data-display="1"');
    expect(html).toContain('→ t3.L0.h');
    expect(html).toContain('data-display="2"');
    expect(html).toMatch(/>undisplay<\/button>/);
  });

  it('no forecasts prop: no column', () => {
    expect(pipeline(frameForecasts, undefined)).not.toContain('class="forecast"');
  });
});
