// View records cached by digest: one entry per distinct record body, however many URLs name it.
// Both maps are LRU-bounded; an evicted entry is simply fetched again.
import { useEffect, useState } from 'react';
import { create } from 'zustand';
import { getViewUrl, url, type View, type ViewTypes } from '../api/client';
import type { Envelope, Refused } from '../api/types.gen';
import { lruSet } from './lru';
import { RequestPool } from './loader';

type Params = Record<string, string | number | null | undefined>;

export const MAX_DIGESTS = 96;
export const MAX_URLS = 256;

interface RecordsState {
  byDigest: ReadonlyMap<string, Envelope<unknown>>;
  byUrl: ReadonlyMap<string, string>;
  put(key: string, env: Envelope<unknown>): void;
  touch(key: string): void;
}

export const useRecords = create<RecordsState>((set) => ({
  byDigest: new Map(),
  byUrl: new Map(),
  put(key, env) {
    set((s) => ({
      byDigest: lruSet(s.byDigest, env.digest, env, MAX_DIGESTS),
      byUrl: lruSet(s.byUrl, key, env.digest, MAX_URLS),
    }));
  },
  touch(key) {
    set((s) => {
      const digest = s.byUrl.get(key);
      const env = digest === undefined ? undefined : s.byDigest.get(digest);
      if (digest === undefined || !env) return s;
      return {
        byDigest: lruSet(s.byDigest, digest, env, MAX_DIGESTS),
        byUrl: lruSet(s.byUrl, key, digest, MAX_URLS),
      };
    });
  },
}));

export const pool = new RequestPool();

export type Loaded<T> =
  | { state: 'loading' }
  | { state: 'ok'; env: Envelope<T> }
  | { state: 'refused'; refused: Refused; status: number }
  | { state: 'error'; message: string };

/** Fetch a view (or reuse its cached envelope); `path === null` means "not applicable".
 *  The effect depends only on the URL string, the view name and whether the cache holds it:
 *  never on an object that is rebuilt each render. */
export function useView<V extends View>(
  view: V,
  path: string | null,
  params?: Params,
): Loaded<ViewTypes[V]> {
  const key = path === null ? null : url(path, params);
  const cached = useRecords((s) => (key ? s.byDigest.get(s.byUrl.get(key) ?? '') : undefined));
  const hasCached = cached !== undefined;
  const [result, setResult] = useState<{ key: string; value: Loaded<ViewTypes[V]> } | null>(null);
  useEffect(() => {
    if (!key) return;
    if (hasCached) {
      useRecords.getState().touch(key);
      return;
    }
    return pool.subscribe(
      key,
      (signal) => getViewUrl(view, key, { signal }),
      (r) => {
        if (r.ok) {
          useRecords.getState().put(key, r.env);
          setResult({ key, value: { state: 'ok', env: r.env as Envelope<ViewTypes[V]> } });
        } else {
          setResult({ key, value: { state: 'refused', refused: r.refused, status: r.status } });
        }
      },
      (error) => setResult({ key, value: { state: 'error', message: String(error) } }),
    );
  }, [key, view, hasCached]);
  if (cached) return { state: 'ok', env: cached as Envelope<ViewTypes[V]> };
  if (result && result.key === key) return result.value;
  return { state: 'loading' };
}
