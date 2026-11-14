// A paged circuit: the body and one child page below are what a real `mdb serve --project`
// returned for a 40-layer tiny Qwen3 circuit (`circuit M[4,9] top=8 depth=3`: 371 nodes, 66 KiB,
// over the record bound before paging), read from /stored/{record} and /stored/{record}?node=.
import { renderToStaticMarkup } from 'react-dom/server';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SCHEMAS, type StoredBody } from '../../api/types.gen';
import { isRec, recs, type Rec } from '../../chrome/labelled';
import { StoredCircuit } from './CircuitView';
import { flatten, layoutCircuit, walk } from './layout';
import { fetchPage, pagedCounts, spliceBody } from './paging';
import rootJson from '../Recorded/fixtures/circuit_paged.json';
import pageJson from '../Recorded/fixtures/circuit_paged_page.json';

const stored = rootJson as unknown as StoredBody;
const pageStored = pageJson as unknown as StoredBody;
const pageNode = pageStored.body.node as Rec;
const NODE = pageStored.node as string;

function pagedIds(body: Rec): string[] {
  return flatten(body)
    .filter((n) => n.paged !== null)
    .map((n) => n.id);
}

afterEach(() => vi.unstubAllGlobals());

describe('a circuit stored in pages', () => {
  it('has a root page with paged groups whose children are counted, not missing', () => {
    const ids = pagedIds(stored.body);
    expect(ids).toContain(NODE);
    const node = flatten(stored.body).find((n) => n.id === NODE)!;
    expect(node.children).toEqual([]);
    const counts = pagedCounts(node.paged)!;
    expect(counts.children).toBe(recs(pageNode.children).length);
    expect(counts.descendants).toBeGreaterThan(counts.children);
    expect(stored.ledger_id).toBe(1);
  });

  it('draws a paged mark on each group and says it is not +N more', () => {
    const html = renderToStaticMarkup(
      <StoredCircuit
        id={stored.ledger_id!}
        body={stored.body}
        source={{ kind: 'stored', sid: 's', record: stored.id }}
      />,
    );
    expect(html.match(/data-testid="paged-mark"/g)?.length).toBe(pagedIds(stored.body).length);
    expect(html).toContain('not +N more');
  });

  it('puts a fetched page into the body: the group gains its children, the rest is untouched', () => {
    const before = flatten(stored.body);
    const body = spliceBody(stored.body, new Map([[NODE, pageNode]]));
    const after = flatten(body);
    const group = after.find((n) => n.id === NODE)!;
    expect(group.paged).toBeNull();
    expect(group.children.length).toBe(recs(pageNode.children).length);
    expect(after.length).toBeGreaterThan(before.length);
    // every other paged group is still paged
    expect(pagedIds(body)).toEqual(pagedIds(stored.body).filter((id) => id !== NODE));
    // the keyboard walk goes right from the group into its first child once loaded
    const layout = layoutCircuit(body, [body]);
    expect(walk(layout, NODE, 'right')).toBe(group.children[0]);
    expect(spliceBody(stored.body, new Map())).toBe(stored.body);
  });

  it('walking right from an unloaded paged group stays on it (the fetch is the caller)', () => {
    const layout = layoutCircuit(stored.body, [stored.body]);
    expect(walk(layout, NODE, 'right')).toBe(NODE);
  });
});

describe('fetching a page', () => {
  const envelope = (data: unknown) =>
    new Response(
      JSON.stringify({ schema: SCHEMAS.stored, state: 'resolved', data, digest: 'd', bytes: 1 }),
    );

  it('reads /stored/{record}?node= and returns the page subtree', async () => {
    const seen: string[] = [];
    vi.stubGlobal('fetch', async (u: string) => {
      seen.push(u);
      return envelope(pageStored);
    });
    const got = await fetchPage({ kind: 'stored', sid: 's1', record: stored.id }, NODE);
    expect(seen).toEqual([`/api/v1/s/s1/stored/${stored.id}?node=${NODE}`]);
    expect(isRec(got) && got.id).toBe(NODE);
  });

  it('shows a refused page as its reason, never a partial subtree', async () => {
    vi.stubGlobal(
      'fetch',
      async () =>
        new Response(
          JSON.stringify({
            refused: 'page_refused',
            state: 'refused',
            detail: 'page abc refused: the page differs from the pointer sha256',
          }),
          { status: 409 },
        ),
    );
    const got = await fetchPage({ kind: 'stored', sid: 's1', record: stored.id }, NODE);
    expect(typeof got).toBe('string');
    expect(got).toContain('differs from the pointer');
  });
});
