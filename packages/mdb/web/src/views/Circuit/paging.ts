// Child pages of a circuit record. A circuit too big for one record keeps its root and first
// levels; a group that did not fit is a node with `paged` (its children WERE examined, they are
// stored as a sha256-checked page). `usePages` fetches a page when the user expands or walks into
// such a node, from the live op (`circuit show ID node=NODEID`) or the recorded route
// (`/stored/{record}?node=NODEID`), and `spliceBody` puts the fetched subtrees into the body the
// graph lays out. Nothing is computed here: a failed fetch is shown, never a partial subtree.
import { useCallback, useRef, useState } from 'react';
import { getViewUrl, url } from '../../api/client';
import { at, int, isRec, recs, str, type Rec } from '../../chrome/labelled';
import { slotOf, type Slot } from '../../state/inspect';
import { useLive } from '../../state/live';

export type PageSource =
  | { kind: 'live'; id: number }
  | { kind: 'stored'; sid: string; record: string };

export type PageState = { state: 'loading' } | { state: 'failed'; reason: string };

/** An integer inside a `{value, authority}` claim (`paged.children`, `ledger_id`), or null. */
export function claimInt(claim: unknown): number | null {
  return int(at(claim, 'value'));
}

/** `{children, descendants}` of a paged node, or null for one that lists its children. */
export function pagedCounts(paged: Rec | null): { children: number; descendants: number } | null {
  if (!paged) return null;
  return {
    children: claimInt(paged.children) ?? 0,
    descendants: claimInt(paged.descendants) ?? 0,
  };
}

/** `body` with every fetched page in place: a node still marked `paged` whose id has a page
 *  becomes that page's subtree (its children, which may be paged again), keeping the old
 *  pointer as `was_paged`. With no pages `body` itself is returned (same object). */
export function spliceBody(body: Rec, pages: ReadonlyMap<string, Rec>): Rec {
  if (!pages.size || !isRec(body.root)) return body;
  return { ...body, root: splice(body.root, pages) };
}

function splice(node: Rec, pages: ReadonlyMap<string, Rec>): Rec {
  const id = str(node.id);
  const page = id !== null && isRec(node.paged) ? pages.get(id) : undefined;
  const here: Rec = page ? { ...page, was_paged: node.paged } : node;
  const kids = recs(here.children);
  return kids.length ? { ...here, children: kids.map((k) => splice(k, pages)) } : here;
}

function reasonOf(slot: Slot): string {
  switch (slot.state) {
    case 'refused':
      return slot.reason;
    case 'request-refused':
      return slot.refused.detail ?? slot.refused.refused;
    case 'interrupted':
      return `interrupted at ${slot.where}`;
    default:
      return 'no answer';
  }
}

export async function fetchPage(source: PageSource, node: string): Promise<Rec | string> {
  if (source.kind === 'live') {
    const label = `circuit show ${source.id} node=${node}`;
    const slot = slotOf(
      label,
      await useLive.getState().run('circuit', { target: 'show', id: source.id, node }),
    );
    if (slot.state !== 'done') return reasonOf(slot);
    const page = at(slot.result, 'page', 'node');
    return isRec(page) ? page : 'the answer holds no page';
  }
  const path = `/s/${encodeURIComponent(source.sid)}/stored/${encodeURIComponent(source.record)}`;
  const got = await getViewUrl('stored', url(path, { node }));
  if (!got.ok) return got.refused.detail ?? got.refused.refused;
  const page = at(got.env.data.body, 'node');
  return isRec(page) ? page : 'the answer holds no page';
}

/** The pages fetched for one circuit, the state of those in flight or failed, and `load(node)`
 *  (a no-op for a node already loaded or loading). Keyed by the circuit: remount per circuit. */
export function usePages(source: PageSource) {
  const [pages, setPages] = useState<ReadonlyMap<string, Rec>>(new Map());
  const [status, setStatus] = useState<ReadonlyMap<string, PageState>>(new Map());
  const busy = useRef(new Set<string>());
  const key = source.kind === 'live' ? `live:${source.id}` : `${source.sid}:${source.record}`;
  const load = useCallback(
    async (node: string) => {
      if (busy.current.has(node)) return;
      busy.current.add(node);
      setStatus((s) => new Map(s).set(node, { state: 'loading' }));
      const got = await fetchPage(source, node);
      busy.current.delete(node);
      if (typeof got === 'string') {
        setStatus((s) => new Map(s).set(node, { state: 'failed', reason: got }));
        return;
      }
      setPages((p) => new Map(p).set(node, got));
      setStatus((s) => {
        const next = new Map(s);
        next.delete(node);
        return next;
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the source's identity is `key`
    [key],
  );
  return { pages, status, load };
}
