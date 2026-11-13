// Live view states drawn from responses a real `mdb serve --tiny qwen2` returned (fixtures/,
// captured in-process from the live app: frames, clocks, op results and reductions).
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { LiveFrame, LiveTimeline, OpResult, ReduceResult, Refused } from '../../api/types.gen';
import { outcomeOf, summaryKey, type Summary } from '../../state/live';
import type { ReduceEntry } from '../../state/reduce';
import { LiveView, tokensOrText, type LiveViewProps } from './LiveView';
import breakpointOp from './fixtures/breakpoint_op.json';
import breakpoint from './fixtures/breakpoint.json';
import committed from './fixtures/committed.json';
import edited from './fixtures/edited.json';
import idle from './fixtures/idle.json';
import many from './fixtures/many.json';
import refusedOp from './fixtures/refused_op.json';

type Env<T> = { data: T };
const act = {
  op: () => undefined,
  command: () => undefined,
  interrupt: () => undefined,
  expand: () => undefined,
  more: () => undefined,
};

function entry(body: Env<ReduceResult> | Refused): ReduceEntry {
  return 'refused' in body
    ? { state: 'refused', refused: body as Refused }
    : { state: 'ok', result: (body as Env<ReduceResult>).data };
}

function summaries(
  frame: LiveFrame,
  layer: number,
  stats: Record<string, Env<ReduceResult> | Refused>,
  topk: [string, Env<ReduceResult>] | null,
): Map<string, Summary> {
  const out = new Map<string, Summary>();
  for (const [port, body] of Object.entries(stats)) {
    const addr = `t${frame.row}.L${layer}.${port}`;
    out.set(summaryKey(frame.seq, addr), {
      stats: entry(body),
      topk: topk && topk[0] === port ? entry(topk[1]) : null,
    });
  }
  return out;
}

function props(over: Partial<LiveViewProps>): LiveViewProps {
  return {
    frame: null,
    timeline: null,
    running: null,
    last: null,
    refused: null,
    summaries: new Map(),
    console: [],
    expanded: null,
    urlAddr: null,
    act,
    ...over,
  };
}

function render(p: LiveViewProps): string {
  return renderToStaticMarkup(<LiveView {...p} />);
}

/** The markup of one pipeline row. */
function row(html: string, port: string): string {
  const start = html.indexOf(`data-port="${port}"`);
  expect(start).toBeGreaterThan(-1);
  return html.slice(start, html.indexOf('</tr>', start));
}

function button(html: string, label: string): string {
  const match = html.match(new RegExp(`<button[^>]*>${label}</button>`));
  expect(match, label).not.toBeNull();
  return match![0];
}

const bpFrame = (breakpoint.frame as Env<LiveFrame>).data;
const bpTimeline = (breakpoint.timeline as Env<LiveTimeline>).data;
const edFrame = (edited.frame as Env<LiveFrame>).data;
const edTimeline = (edited.timeline as Env<LiveTimeline>).data;

describe('Live view', () => {
  it('idle: no run, motion disabled, nothing drawn as a value', () => {
    const html = render(
      props({
        frame: (idle.frame as Env<LiveFrame>).data,
        timeline: (idle.timeline as Env<LiveTimeline>).data,
      }),
    );
    expect(html).toContain('data-testid="no-run"');
    expect(html).toContain('no run');
    for (const label of ['step', 'next', 'finish', 'continue', 'reverse-step', 'interrupt']) {
      expect(button(html, label)).toContain('disabled');
    }
    expect(html).not.toContain('aria-label="pipeline"');
    expect(html).not.toContain('class="num"');
  });

  it('running: every control but interrupt is disabled', () => {
    const html = render(props({ frame: bpFrame, timeline: bpTimeline, running: 'continue' }));
    expect(html).toContain('data-running="yes"');
    expect(html).toContain('running continue');
    for (const label of ['step', 'next', 'finish', 'continue', 'reverse-step', 'commit', 'run']) {
      expect(button(html, label)).toContain('disabled');
    }
    expect(button(html, 'interrupt')).not.toContain('disabled');
  });

  it('stopped at a breakpoint: says so and marks the cursor port with its operator and proof', () => {
    const d = (breakpointOp as Env<OpResult>).data;
    const last = outcomeOf(d.op, d.seq, d.result);
    expect(last).toMatchObject({ kind: 'ok', stopped: 'breakpoint' });
    const html = render(
      props({
        frame: bpFrame,
        timeline: bpTimeline,
        last,
        summaries: summaries(bpFrame, 1, breakpoint.stats as Record<string, Env<ReduceResult>>, [
          'p',
          breakpoint.topk as Env<ReduceResult>,
        ]),
      }),
    );
    expect(html).toContain('data-stopped="breakpoint"');
    expect(html).toContain('continue stopped: breakpoint');
    expect(bpFrame.where).toBe('t3.L1.p');
    const p = row(html, 'p');
    expect(p).toContain('aria-current="step"');
    expect(p).toContain('softmax');
    expect(p).toMatch(/P\d+/);
    expect(p).toContain('data-ev="measured"'); // norm and top entries, measured
    expect(button(html, 'interrupt')).toContain('disabled');
    // the breakpoint is listed with its hit count
    expect(html).toContain('L1.p');
    expect(html).toMatch(/1<\/span> hits/);
  });

  it('a refused op shows the refusal and its reason, not a result', () => {
    const d = (refusedOp as Env<OpResult>).data;
    const last = outcomeOf(d.op, d.seq, d.result);
    expect(last.kind).toBe('refused');
    const html = render(props({ frame: bpFrame, timeline: bpTimeline, last }));
    expect(html).toContain('data-testid="state-refused"');
    expect(html).toContain('argument &#x27;n&#x27; must be an integer');
    expect(html).toContain('(bad-args)');
  });

  it('an edited value: the edit and its label, downstream marks, and no value past the cursor', () => {
    const html = render(
      props({
        frame: edFrame,
        timeline: edTimeline,
        summaries: summaries(edFrame, 0, edited.stats as Record<string, Env<ReduceResult>>, [
          'v',
          edited.topk as Env<ReduceResult>,
        ]),
        expanded: 't3.L0.v',
      }),
    );
    const v = row(html, 'v');
    expect(v).toContain('data-testid="edited"');
    expect(v).toContain('supplied');
    // the reduced norm of the edited value carries the supplied label (hatched mark)
    expect(v).toMatch(/class="num" data-ev="supplied"/);
    expect(v).toContain('<line');
    const s = row(html, 's');
    expect(s).toContain('data-testid="downstream"');
    expect(s).toContain('data-ev="derived-exact"');
    expect(s).toContain('ev-chip');
    expect(row(html, 'h')).not.toContain('data-testid="downstream"');
    // past the cursor: nothing computed, nothing claimed
    for (const port of ['o', 'r', 'm', 'h_out']) {
      const r = row(html, port);
      expect(r).toContain('not computed yet');
      expect(r).not.toContain('class="num"');
      expect(r).not.toContain('data-testid="edited"');
    }
    // the clock marks the edited stop
    expect(html).toMatch(/data-edited="yes"[^>]*>L0\.v</);
    // the checkpoint is listed once, durable (stored in the project), and can be restarted
    const checkpoints = html.slice(html.indexOf('aria-label="checkpoints"'));
    expect(checkpoints.match(/<tr>/g)).toHaveLength(1);
    expect(checkpoints).toContain('durable');
    // the edit chip carries its full label as a title (it wraps, never clipped)
    expect(v).toMatch(/class="chip live-mark"[^>]*title="edited \(edit by v\) · supplied"/);
  });

  it('after commit every computed port shows its reduced value (none unresolved)', () => {
    const frame = (committed.frame as Env<LiveFrame>).data;
    expect(frame.where).toBe('t3.L0.h_out');
    const html = render(
      props({
        frame,
        timeline: (committed.timeline as Env<LiveTimeline>).data,
        summaries: summaries(frame, 0, committed.stats as Record<string, Env<ReduceResult>>, null),
      }),
    );
    const computed = frame.ports.filter((p) => p.computed);
    expect(computed).toHaveLength(19);
    for (const p of computed) {
      const r = row(html, p.port);
      expect(r, p.port).toMatch(/class="num" data-ev="(measured|supplied|derived-exact)"/);
      expect(r, p.port).not.toContain('num-unresolved');
    }
  });

  it('a frame that shortened its lists says how many more and offers the rest', () => {
    const frame = (many.frame as Env<LiveFrame>).data;
    const fields = frame.more.map((m) => m.field);
    expect(fields).toContain('committed');
    expect(fields).toContain('breakpoints');
    const html = render(props({ frame, timeline: null }));
    for (const field of ['committed', 'breakpoints']) {
      const block = html.slice(html.indexOf(`data-testid="more-${field}"`));
      const count = frame.more.find((m) => m.field === field)!.count;
      expect(block).toMatch(new RegExp(`^[^]*?>${count}</span> more`));
      expect(block.slice(0, 400)).toContain('show more');
    }
    // committed edits are listed one per line with their row
    expect(html.match(/data-testid="committed-edit"/g)).toHaveLength(
      (frame.edits as { committed: unknown[] }).committed.length,
    );
    // a port's upstream count is the total, not the few listed
    const h = row(html, 'h');
    const port = frame.ports.find((p) => p.port === 'h')!;
    expect(port.edit!.upstream.length).toBeLessThan(port.edit!.upstream_total);
    expect(h).toContain(`>${port.edit!.upstream_total}</span> edits`);
    expect(h).toMatch(/and \d+ more"/);
    // a long breakpoint condition is clipped in the table with the whole text as a title
    expect(html).toMatch(/class="mono faint live-clip" title="norm &gt; 9{240}"/);
  });

  it('pages fetched with "more" are listed after the frame’s own items', () => {
    const frame = (many.frame as Env<LiveFrame>).data;
    const page = (many.committed_page as { data: { items: Record<string, unknown>[] } }).data;
    const bps = (many.breakpoints_page as { data: { items: Record<string, unknown>[] } }).data;
    const extra = new Map([
      ['committed', { items: page.items, next: null, refused: null }],
      ['breakpoints', { items: bps.items, next: null, refused: null }],
    ]);
    const html = render(props({ frame, timeline: null, extra }));
    const shown = (frame.edits as { committed: unknown[] }).committed.length;
    expect(html.match(/data-testid="committed-edit"/g)).toHaveLength(shown + page.items.length);
    const block = html.slice(html.indexOf('data-testid="more-committed"'));
    expect(block.slice(0, 200)).toContain('all shown');
    const table = html.slice(html.indexOf('aria-label="breakpoints and watches"'));
    expect(table.slice(0, table.indexOf('</table>')).match(/<tr /g)).toHaveLength(
      frame.breakpoints.length + bps.items.length,
    );
  });

  it('a refused frame leaves the controls and the console usable, with the refusal beside them', () => {
    // The answer the server gave before this fix for an over-cap frame (the audit's 12-overcap).
    const refused: Refused = {
      refused: 'over_cap',
      state: 'refused',
      detail: null,
      bytes: 69586,
      cap: 65536,
    };
    const html = render(props({ frame: null, refused }));
    expect(html).toContain('over_cap');
    for (const label of ['step', 'next', 'finish', 'continue', 'reverse-step']) {
      expect(button(html, label)).not.toContain('disabled');
    }
    expect(html).toContain('aria-label="prompt"');
    expect(html).toContain('aria-label="console"');
    expect(html).toContain('aria-label="command"');
    expect(button(html, 'interrupt')).toContain('disabled');
  });

  it('labels each number by its authority and never renders a missing value as 0', () => {
    const stats = { ...(edited.stats as Record<string, Env<ReduceResult> | Refused>) };
    stats.p = edited.uncomputed as Refused;
    const html = render(
      props({
        frame: edFrame,
        timeline: edTimeline,
        summaries: summaries(edFrame, 0, stats, null),
      }),
    );
    for (const ev of ['measured', 'supplied', 'derived-exact']) {
      expect(html).toContain(`data-ev="${ev}"`);
    }
    const p = row(html, 'p');
    expect(p).toContain('num-unresolved');
    expect(p).toContain('has not been computed');
    expect(p).not.toMatch(/>\s*0\s*</);
    // the staged/committed edit value is an op-labelled number ({value, authority})
    const staged = html.slice(html.indexOf('data-testid="staged-edit"'));
    expect(staged.slice(0, staged.indexOf('</div>'))).toMatch(/data-ev="supplied"[^>]*>.*0\.25/);
    // a port whose summary has not arrived shows a placeholder, not a number
    expect(row(html, 'q')).toContain('…');
    expect(row(html, 'q')).not.toContain('class="num"');
  });

  it('a pasted link to another address offers `until` instead of moving the model', () => {
    const html = render(props({ frame: bpFrame, timeline: bpTimeline, urlAddr: 't3.L1.o' }));
    expect(html).toContain('data-testid="url-addr"');
    expect(html).toContain('t3.L1.o');
  });
});

describe('live helpers', () => {
  it('classifies interrupted, unresolved and refused results', () => {
    expect(
      outcomeOf('continue', 4, {
        interrupted: 't9.L0.q',
        state: 'unresolved',
        reason: 'interrupted at t9.L0.q',
      }),
    ).toEqual({ kind: 'interrupted', op: 'continue', seq: 4, where: 't9.L0.q' });
    expect(outcomeOf('print', 2, { state: 'unresolved', reason: 'x' }).kind).toBe('unresolved');
    expect(outcomeOf('step', 3, { where: 't3.L0.a', state: 'resolved' })).toMatchObject({
      kind: 'ok',
      stopped: null,
    });
  });

  it('reads token ids or prompt text', () => {
    expect(tokensOrText('1 5 9 3')).toEqual([1, 5, 9, 3]);
    expect(tokensOrText('1,5, 9')).toEqual([1, 5, 9]);
    expect(tokensOrText('Weather: cold')).toBe('Weather: cold');
  });
});
