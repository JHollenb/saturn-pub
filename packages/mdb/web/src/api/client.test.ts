import { describe, expect, it } from 'vitest';
import { accept, getView, postView, SchemaError, type Result } from './client';
import type { Frame } from './types.gen';
import { CAP_BYTES, SCHEMAS } from './types.gen';

function respond(status: number, body: unknown) {
  return async () =>
    new Response(typeof body === 'string' ? body : JSON.stringify(body), {
      status,
      headers: { 'content-type': 'application/json' },
    });
}

describe('Envelope / refused handling', () => {
  it('accepts an envelope of the requested view', () => {
    const r = accept(
      'hello',
      { schema: SCHEMAS.hello, state: 'resolved', data: { x: 1 }, digest: 'd', bytes: 7 },
      200,
    );
    expect(r.ok).toBe(true);
    if (r.ok) expect((r.env.data as unknown as { x: number }).x).toBe(1);
  });

  it('rejects an envelope of another view', () => {
    expect(() =>
      accept(
        'frame',
        { schema: SCHEMAS.hello, state: 'resolved', data: {}, digest: 'd', bytes: 2 },
        200,
      ),
    ).toThrow(SchemaError);
    expect(() =>
      accept('frame', { schema: SCHEMAS.frame, state: 'resolved', data: {} }, 200),
    ).toThrow(SchemaError);
    expect(() => accept('frame', [1, 2], 200)).toThrow(SchemaError);
  });

  it('surfaces the cap refusal with its byte count (never a truncated body)', async () => {
    const r = await getView(
      'graph',
      '/s/tiny/graph',
      { zoom: 'session' },
      respond(413, { refused: 'over_cap', state: 'refused', bytes: 70009, cap: CAP_BYTES }),
    );
    expect(r).toEqual({
      ok: false,
      status: 413,
      refused: { refused: 'over_cap', state: 'refused', bytes: 70009, cap: 65536 },
    });
  });

  it('treats other refusals as states, not exceptions', async () => {
    const r = await postView(
      'reduce',
      '/s/tiny/reduce',
      { pointer: { kind: 'blob', ref: 'x' }, op: 'stats', params: {} },
      respond(422, { refused: 'unresolved', state: 'refused', detail: 'not materialised' }),
    );
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.refused.refused).toBe('unresolved');
  });

  it('maps a non-JSON body and a network failure to refusals', async () => {
    const bad = await getView('hello', '/hello', undefined, respond(502, '<html>bad gateway'));
    expect(bad).toMatchObject({ ok: false, status: 502, refused: { refused: 'bad_response' } });
    const down = await getView('hello', '/hello', undefined, async () => {
      throw new TypeError('fetch failed');
    });
    expect(down).toMatchObject({ ok: false, status: 0, refused: { refused: 'network' } });
  });

  it('builds query strings without empty params', async () => {
    let seen = '';
    await getView(
      'symbols',
      '/s/a b/symbols',
      { at: 't1.L0.h[" hot"=5012]', q: '', page: 0 },
      async (u) => {
        seen = u;
        return new Response(
          JSON.stringify({
            schema: SCHEMAS.symbols,
            state: 'resolved',
            data: {},
            digest: 'd',
            bytes: 2,
          }),
        );
      },
    );
    expect(seen).toBe('/api/v1/s/a b/symbols?at=t1.L0.h%5B%22%20hot%22%3D5012%5D&page=0');
  });
});

describe('typed boundary', () => {
  it('ties each route to its generated view type (a wrong pairing does not compile)', async () => {
    const ok = await getView('hello', '/hello', undefined, respond(200, 'x'));
    expect(ok.ok).toBe(false);
    // @ts-expect-error a `hello` route cannot be read as a Frame
    const wrong: Result<Frame> = await getView('hello', '/hello', undefined, respond(200, 'x'));
    expect(wrong.ok).toBe(false);
  });

  it('requires `state` on every envelope and refusal, and a reason when not resolved', () => {
    const base = { schema: SCHEMAS.frame, data: {}, digest: 'd', bytes: 2 };
    expect(() => accept('frame', base, 200)).toThrow(SchemaError); // no state
    expect(accept('frame', { ...base, state: 'resolved' }, 200).ok).toBe(true);
    expect(() => accept('frame', { ...base, state: 'refused' }, 200)).toThrow(SchemaError);
    expect(() => accept('frame', { ...base, state: 'unresolved', reason: '' }, 200)).toThrow(
      SchemaError,
    );
    expect(accept('frame', { ...base, state: 'unresolved', reason: 'not recorded' }, 200).ok).toBe(
      true,
    );
    expect(() => accept('frame', { ...base, state: 'weird' }, 200)).toThrow(SchemaError);
    expect(accept('frame', { refused: 'bad_request', state: 'refused', detail: 'x' }, 400).ok).toBe(
      false,
    );
    expect(() => accept('frame', { refused: 'bad_request' }, 400)).toThrow(SchemaError);
  });

  it('turns a malformed envelope into a visible refusal state, not an exception', async () => {
    const r = await getView(
      'frame',
      '/s/tiny/frame',
      { addr: 't0.L0.h' },
      respond(200, { schema: 'nope', state: 'resolved' }),
    );
    expect(r).toMatchObject({ ok: false, refused: { refused: 'bad_envelope' } });
    const arr = await getView('frame', '/s/tiny/frame', undefined, respond(200, [1]));
    expect(arr).toMatchObject({ ok: false, refused: { refused: 'bad_envelope' } });
  });

  it('passes the abort signal to fetch', async () => {
    const controller = new AbortController();
    let seen: AbortSignal | null | undefined;
    await getView(
      'hello',
      '/hello',
      undefined,
      async (_u, init) => {
        seen = init?.signal;
        return new Response('{}');
      },
      { signal: controller.signal },
    );
    expect(seen).toBe(controller.signal);
  });
});
