// The live store against recorded server answers (a stubbed fetch replays them).
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import breakpointOp from '../views/Live/fixtures/breakpoint_op.json';
import breakpoint from '../views/Live/fixtures/breakpoint.json';
import { summaryKey, useLive } from './live';

type Call = { url: string; body: unknown };

function stub(calls: Call[]) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const body = init?.body ? JSON.parse(String(init.body)) : null;
    calls.push({ url, body });
    let answer: unknown;
    if (url.endsWith('/live')) answer = breakpoint.frame;
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

describe('live store', () => {
  let calls: Call[];
  beforeEach(() => {
    calls = [];
    vi.stubGlobal('fetch', stub(calls));
    useLive.setState({ sid: null, frame: null, running: null, last: null, summaries: new Map() });
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

  it('does not send a second op while one is in flight', async () => {
    useLive.setState({ sid: 'qwen2-x', running: 'continue' });
    await useLive.getState().op('step');
    expect(calls).toHaveLength(0);
  });
});
