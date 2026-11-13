import { describe, expect, it } from 'vitest';
import { getViewUrl, SchemaError, type Result } from '../api/client';
import { SCHEMAS } from '../api/types.gen';
import { lruSet } from './lru';
import { RequestPool } from './loader';
import { MAX_DIGESTS, MAX_URLS, useRecords } from './records';
import { MAX_REDUCE_ENTRIES, useReduce } from './reduce';

/** A fake fetch whose responses the test releases in any order. It ignores the abort signal on
 *  purpose, to show that a stale answer is dropped by the pool, not by the network layer. */
function controlledFetch() {
  const pending = new Map<string, (body: unknown) => void>();
  const signals = new Map<string, AbortSignal | undefined>();
  const fetchImpl = (input: string, init?: RequestInit) =>
    new Promise<Response>((resolve) => {
      signals.set(input, init?.signal ?? undefined);
      pending.set(input, (body) => resolve(new Response(JSON.stringify(body))));
    });
  const frameBody = (addr: string) => ({
    schema: SCHEMAS.frame,
    state: 'resolved',
    data: { addr },
    digest: `d-${addr}`,
    bytes: 10,
  });
  return {
    fetchImpl,
    signals,
    release: (u: string, addr: string) => pending.get(u)?.(frameBody(addr)),
  };
}

describe('request pool', () => {
  it('delivers only to current subscribers when responses arrive out of order', async () => {
    const net = controlledFetch();
    const pool = new RequestPool();
    const seen: string[] = [];
    const start = (u: string) => (signal: AbortSignal) =>
      getViewUrl('frame', u, { signal }, net.fetchImpl);
    const onResult = (r: Result<unknown>) => {
      if (r.ok) seen.push((r.env.data as { addr: string }).addr);
    };
    // View shows cursor A, then the cursor moves to B before A answers.
    const cancelA = pool.subscribe('/a', start('/a'), onResult, () => undefined);
    cancelA();
    expect(net.signals.get('/a')?.aborted).toBe(true);
    const cancelB = pool.subscribe('/b', start('/b'), onResult, () => undefined);
    net.release('/b', 'B');
    net.release('/a', 'A'); // the stale answer arrives after the fresh one
    await new Promise((r) => setTimeout(r, 0));
    expect(seen).toEqual(['B']);
    cancelB();
    expect(pool.size).toBe(0);
  });

  it('also drops the stale answer when it arrives first', async () => {
    const net = controlledFetch();
    const pool = new RequestPool();
    const seen: string[] = [];
    const onResult = (r: Result<unknown>) => {
      if (r.ok) seen.push((r.env.data as { addr: string }).addr);
    };
    const cancelA = pool.subscribe(
      '/a',
      (signal) => getViewUrl('frame', '/a', { signal }, net.fetchImpl),
      onResult,
      () => undefined,
    );
    pool.subscribe(
      '/b',
      (signal) => getViewUrl('frame', '/b', { signal }, net.fetchImpl),
      onResult,
      () => undefined,
    );
    cancelA();
    net.release('/a', 'A');
    net.release('/b', 'B');
    await new Promise((r) => setTimeout(r, 0));
    expect(seen).toEqual(['B']);
  });

  it('shares one request between subscribers and aborts only when the last leaves', () => {
    const net = controlledFetch();
    const pool = new RequestPool();
    let starts = 0;
    const start = (signal: AbortSignal) => {
      starts += 1;
      return getViewUrl('frame', '/a', { signal }, net.fetchImpl);
    };
    const one = pool.subscribe(
      '/a',
      start,
      () => undefined,
      () => undefined,
    );
    const two = pool.subscribe(
      '/a',
      start,
      () => undefined,
      () => undefined,
    );
    expect(starts).toBe(1);
    one();
    expect(net.signals.get('/a')?.aborted).toBe(false);
    two();
    expect(net.signals.get('/a')?.aborted).toBe(true);
  });

  it('does not refetch a settled key that is subscribed again after the cache holds it', async () => {
    const net = controlledFetch();
    const pool = new RequestPool();
    const results: boolean[] = [];
    pool.subscribe(
      '/a',
      (signal) => getViewUrl('frame', '/a', { signal }, net.fetchImpl),
      (r) => results.push(r.ok),
      () => undefined,
    );
    net.release('/a', 'A');
    await new Promise((r) => setTimeout(r, 0));
    expect(results).toEqual([true]);
    expect(pool.size).toBe(0);
  });
});

describe('bounded caches', () => {
  it('lruSet evicts the least recently set key and keeps the map immutable', () => {
    const a = lruSet(new Map<string, number>(), 'a', 1, 2);
    const b = lruSet(a, 'b', 2, 2);
    const c = lruSet(lruSet(b, 'a', 1, 2), 'c', 3, 2); // touching a makes b the oldest
    expect([...c.keys()]).toEqual(['a', 'c']);
    expect([...a.keys()]).toEqual(['a']);
  });

  it('records caches stay within their bounds', () => {
    for (let i = 0; i < MAX_URLS + 40; i++) {
      useRecords
        .getState()
        .put(`/u${i}`, { schema: 's', state: 'resolved', data: i, digest: `d${i}`, bytes: 1 });
    }
    expect(useRecords.getState().byDigest.size).toBe(MAX_DIGESTS);
    expect(useRecords.getState().byUrl.size).toBe(MAX_URLS);
    // The newest entry is present; the oldest was evicted.
    const state = useRecords.getState();
    expect(state.byDigest.get(state.byUrl.get(`/u${MAX_URLS + 39}`) ?? '')).toBeDefined();
    expect(state.byUrl.has('/u0')).toBe(false);
  });

  it('reduce entries stay within their bound', async () => {
    for (let i = 0; i < MAX_REDUCE_ENTRIES + 10; i++) {
      await useReduce.getState().run(
        'tiny',
        {
          digest: `p${i}`,
          addr: null,
          spec: null,
          via: [],
          status: 'frontier',
          reason: 'not retained',
          shape: null,
          dtype: null,
        },
        'stats',
      );
    }
    expect(useReduce.getState().entries.size).toBe(MAX_REDUCE_ENTRIES);
  });

  it('SchemaError is still exported for callers that validate', () => {
    expect(new SchemaError('x')).toBeInstanceOf(Error);
  });
});
