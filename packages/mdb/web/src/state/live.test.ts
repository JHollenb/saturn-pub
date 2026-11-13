// The live store against recorded server answers (a stubbed fetch replays them).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import breakpointOp from '../views/Live/fixtures/breakpoint_op.json';
import breakpoint from '../views/Live/fixtures/breakpoint.json';
import many from '../views/Live/fixtures/many.json';
import { summaryKey, useLive } from './live';

type Call = { url: string; body: unknown };

function stub(calls: Call[]) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const body = init?.body ? JSON.parse(String(init.body)) : null;
    calls.push({ url, body });
    let answer: unknown;
    if (url.endsWith('/live')) answer = frameAnswer;
    else if (url.includes('/live/items?')) answer = many.committed_page;
    else if (url.endsWith('/live/timeline')) answer = breakpoint.timeline;
    else if (url.endsWith('/op')) answer = breakpointOp;
    else if (url.endsWith('/reduce')) {
      const address = (body as { pointer: { address: string } }).pointer.address;
      const port = address.split('.').pop() ?? '';
      const stats = breakpoint.stats as Record<string, unknown>;
      answer =
        (body as { op: string }).op === 'topk'
          ? breakpoint.topk
          : (stats[port] ?? { refused: 'unresolved', state: 'refused', detail: 'not in fixture' });
    } else answer = { refused: 'not_found', state: 'refused' };
    return new Response(JSON.stringify(answer), { status: 200 });
  });
}

const settle = () => new Promise((r) => setTimeout(r, 0));
let frameAnswer: unknown = breakpoint.frame;

describe('live store', () => {
  let calls: Call[];
  beforeEach(() => {
    calls = [];
    vi.stubGlobal('fetch', stub(calls));
    frameAnswer = breakpoint.frame;
    useLive.setState({
      sid: null,
      frame: null,
      running: null,
      last: null,
      refused: null,
      summaries: new Map(),
      visible: null,
      extra: new Map(),
    });
  });
  afterEach(() => vi.unstubAllGlobals());

  it('an op posts once through /op, records its outcome, then re-fetches frame and clock', async () => {
    useLive.getState().setSid('qwen2-x');
    await settle();
    calls.length = 0;
    await useLive.getState().op('continue');
    const urls = calls.map((c) => c.url);
    expect(urls[0]).toBe('/api/v1/s/qwen2-x/op');
    expect(calls[0]?.body).toMatchObject({ op: 'continue', args: {} });
    expect(urls).toContain('/api/v1/s/qwen2-x/live');
    expect(urls).toContain('/api/v1/s/qwen2-x/live/timeline');
    const s = useLive.getState();
    expect(s.running).toBeNull();
    expect(s.last).toMatchObject({ kind: 'ok', stopped: 'breakpoint' });
    expect(s.frame?.where).toBe('t3.L1.p');
  });

  it('summarises computed ports one at a time through /reduce occurrence pointers', async () => {
    useLive.getState().setSid('qwen2-x');
    for (let i = 0; i < 60; i++) await settle();
    const reduces = calls.filter((c) => c.url.endsWith('/reduce'));
    const frame = useLive.getState().frame!;
    const computed = frame.ports.filter((p) => p.computed);
    expect(reduces.length).toBe(computed.length + 1); // stats for each, topk at the cursor
    for (const r of reduces) {
      expect((r.body as { pointer: { kind: string } }).pointer.kind).toBe('occurrence');
    }
    const p = useLive.getState().summaries.get(summaryKey(frame.seq, 't3.L1.p'));
    expect(p?.stats.state).toBe('ok');
    expect(p?.topk?.state).toBe('ok');
    expect(frame.ports.some((x) => !x.computed)).toBe(true);
  });

  it('an event for a frame it already holds does not re-fetch; a newer one does', async () => {
    useLive.getState().setSid('qwen2-x');
    for (let i = 0; i < 60; i++) await settle();
    const seq = useLive.getState().frame!.seq;
    calls.length = 0;
    useLive.getState().onEvent({ op: 'step', seq });
    await settle();
    expect(calls.filter((c) => c.url.endsWith('/live'))).toHaveLength(0);
    useLive.getState().onEvent({ op: 'step', seq: seq + 1 });
    await settle();
    expect(calls.filter((c) => c.url.endsWith('/live'))).toHaveLength(1);
  });

  it('asks only for the rows on screen when the page reports them', async () => {
    useLive.setState({ visible: new Set() });
    useLive.getState().setSid('qwen2-x');
    for (let i = 0; i < 60; i++) await settle();
    const asked = () =>
      calls
        .filter((c) => c.url.endsWith('/reduce'))
        .map((c) => (c.body as { pointer: { address: string } }).pointer.address);
    expect(new Set(asked())).toEqual(new Set(['t3.L1.p'])); // only the cursor row (stats, topk)
    useLive.getState().setVisible('t3.L1.q', true);
    for (let i = 0; i < 20; i++) await settle();
    expect(asked().filter((a) => a === 't3.L1.q')).toHaveLength(1);
    expect(asked()).not.toContain('t3.L1.h');
  });

  it('"more" fetches the next page of a list from where the frame stopped', async () => {
    frameAnswer = many.frame;
    useLive.setState({ visible: new Set() });
    useLive.getState().setSid('qwen2-x');
    for (let i = 0; i < 20; i++) await settle();
    const frame = useLive.getState().frame!;
    const shown = (frame.edits as { committed: unknown[] }).committed.length;
    await useLive.getState().more('committed');
    const url = calls.map((c) => c.url).find((u) => u.includes('/live/items'))!;
    expect(url).toBe(`/api/v1/s/qwen2-x/live/items?list=committed&offset=${shown}`);
    const extra = useLive.getState().extra.get('committed')!;
    expect(extra.items).toHaveLength(many.committed_page.data.items.length);
    expect(extra.next).toBeNull();
    calls.length = 0;
    await useLive.getState().more('committed'); // complete: nothing more to ask
    expect(calls).toHaveLength(0);
  });

  it('a refused frame is reported and the store still sends ops', async () => {
    frameAnswer = { refused: 'over_cap', state: 'refused', bytes: 69586, cap: 65536 };
    useLive.getState().setSid('qwen2-x');
    for (let i = 0; i < 5; i++) await settle();
    expect(useLive.getState().frame).toBeNull();
    expect(useLive.getState().refused?.refused).toBe('over_cap');
    calls.length = 0;
    await useLive.getState().op('step');
    expect(calls[0]?.url).toBe('/api/v1/s/qwen2-x/op');
  });

  it('does not send a second op while one is in flight', async () => {
    useLive.setState({ sid: 'qwen2-x', running: 'continue' });
    await useLive.getState().op('step');
    expect(calls).toHaveLength(0);
  });
});
