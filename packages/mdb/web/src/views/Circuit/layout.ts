// Layout and keyboard walk of a circuit (the `circuit` op's record body: a node tree under
// `root`). The consumer sits at the left; each depth is one column to its right (layer
// components, then heads or neurons, then source token rows).
//
// Positions are sticky slots keyed by a node's natural key (kind and address), computed over the
// circuit's family: every circuit of the same consumer this page holds, taken in id order. The
// first circuit that shows a node gives it the first free slot of its column at or below its
// parent's slot; later circuits never move it. So a node keeps its position when K or D change
// (a new circuit only fills free slots), and the same family always gives the same layout: no
// randomness, no dependence on render history.
import { isRec, magnitude, recs, str, valOf, type Rec } from '../../chrome/labelled';

export const PAD = 16;
export const NODE_R = 7;
/** Width reserved right of each node for its two label lines (name, value): no edge or other
 *  label enters it. Edges leave a node from the end of its label. */
export const LABEL_W = 136;
/** Room between one column's labels and the next column's nodes, where the edges run. */
export const EDGE_GAP = 56;
export const LABEL_X = NODE_R + 5;
export const COL_W = LABEL_X + LABEL_W + EDGE_GAP + NODE_R;
/** Two label lines (10 px name above the node's centre, 10 px value below) plus a gap. */
export const ROW_H = 34;
export const MAX_LABEL = 21;
/** Smallest scale the graph is drawn at to fit its box (10 px labels at 0.8 = 8 px). */
export const MIN_SCALE = 0.8;

export interface CNode {
  id: string;
  name: string;
  kind: string;
  address: string;
  value: unknown;
  note: string | null;
  proof: string | null;
  depth: number;
  parent: string | null;
  children: string[];
  /** The raw node record (for the Inspector). */
  raw: Rec;
}

/** The nodes of a circuit body, parents before children (breadth first). A node without an id
 *  gets its tree path, so a malformed record still draws. */
export function flatten(body: unknown): CNode[] {
  const root = isRec(body) ? body.root : undefined;
  if (!isRec(root)) return [];
  const out: CNode[] = [];
  const queue: [Rec, number, string | null, string][] = [[root, 0, null, '0']];
  while (queue.length) {
    const [raw, depth, parent, path] = queue.shift()!;
    const id = str(raw.id) ?? `p${path}`;
    const kids = recs(raw.children);
    const node: CNode = {
      id,
      name: str(raw.name) ?? str(raw.address) ?? id,
      kind: str(raw.kind) ?? 'node',
      address: str(raw.address) ?? '',
      value: raw.value,
      note: str(raw.note),
      proof: str(raw.proof),
      depth,
      parent,
      children: [],
      raw: Object.fromEntries(Object.entries(raw).filter(([k]) => k !== 'children')),
    };
    out.push(node);
    kids.forEach((kid, i) => queue.push([kid, depth + 1, id, `${path}.${i}`]));
  }
  const byId = new Map(out.map((n) => [n.id, n]));
  for (const n of out) if (n.parent) byId.get(n.parent)?.children.push(n.id);
  return out;
}

/** The identity of a node across circuits of one consumer: kind and address (and the name,
 *  which tells apart nodes that share an address, e.g. a head and its remainder). A `+N more`
 *  node is one node whatever N is. */
export function naturalKey(n: CNode): string {
  return n.kind === 'more' ? `more|${n.address}` : `${n.kind}|${n.address}|${n.name}`;
}

export interface Slot {
  col: number;
  slot: number;
}

/** Sticky slots for a family of circuit bodies (taken in the order given: id order). */
export function familySlots(bodies: unknown[]): Map<string, Slot> {
  const slots = new Map<string, Slot>();
  const used = new Map<number, Set<number>>();
  for (const body of bodies) {
    const nodes = flatten(body);
    const keyOf = new Map(nodes.map((n) => [n.id, naturalKey(n)]));
    for (const n of nodes) {
      const key = keyOf.get(n.id)!;
      if (slots.has(key)) continue;
      const parent = n.parent ? slots.get(keyOf.get(n.parent) ?? '') : undefined;
      const col = n.depth;
      const taken = used.get(col) ?? new Set<number>();
      let slot = parent?.slot ?? 0;
      while (taken.has(slot)) slot += 1;
      taken.add(slot);
      used.set(col, taken);
      slots.set(key, { col, slot });
    }
  }
  return slots;
}

export interface Placed {
  node: CNode;
  key: string;
  x: number;
  y: number;
  col: number;
  slot: number;
  label: string;
}

export interface Edge {
  from: Placed;
  to: Placed;
  /** Stroke width from the exact contribution's size; null when the child has no value. */
  width: number | null;
  sign: 'pos' | 'neg' | 'none';
}

export interface CircuitLayout {
  placed: Placed[];
  edges: Edge[];
  width: number;
  height: number;
}

function truncate(text: string): string {
  return text.length <= MAX_LABEL ? text : `${text.slice(0, MAX_LABEL - 1)}…`;
}

/** Lay out `body` on its family's slots (`family` must contain `body`). */
export function layoutCircuit(body: unknown, family: unknown[]): CircuitLayout {
  const slots = familySlots(family.includes(body) ? family : [...family, body]);
  const nodes = flatten(body);
  let maxCol = 0;
  let maxSlot = 0;
  for (const s of slots.values()) {
    maxCol = Math.max(maxCol, s.col);
    maxSlot = Math.max(maxSlot, s.slot);
  }
  const placed: Placed[] = nodes.map((node) => {
    const key = naturalKey(node);
    const s = slots.get(key) ?? { col: node.depth, slot: 0 };
    return {
      node,
      key,
      col: s.col,
      slot: s.slot,
      x: PAD + NODE_R + s.col * COL_W,
      y: PAD + ROW_H / 2 + s.slot * ROW_H,
      label: truncate(node.name),
    };
  });
  const at = new Map(placed.map((p) => [p.node.id, p]));
  const sizes = placed
    .filter((p) => p.node.parent !== null)
    .map((p) => magnitude(valOf(p.node.value)))
    .filter((m): m is { v: number; signed: boolean } => m !== null)
    .map((m) => Math.abs(m.v));
  const largest = Math.max(0, ...sizes);
  const edges: Edge[] = [];
  for (const p of placed) {
    const parent = p.node.parent ? at.get(p.node.parent) : undefined;
    if (!parent) continue;
    const m = magnitude(valOf(p.node.value));
    edges.push({
      from: parent,
      to: p,
      width: m === null ? null : 1 + (largest > 0 ? (5 * Math.abs(m.v)) / largest : 0),
      sign: m === null || !m.signed ? 'none' : m.v < 0 ? 'neg' : 'pos',
    });
  }
  return {
    placed,
    edges,
    width: PAD * 2 + maxCol * COL_W + NODE_R + LABEL_X + LABEL_W,
    height: PAD * 2 + (maxSlot + 1) * ROW_H,
  };
}

export type WalkKey = 'up' | 'down' | 'left' | 'right';

/** The node a key moves to from `id`: up/down the previous/next sibling (by position), left
 *  the parent (towards the consumer), right the first child (along an edge). Stays put at an
 *  edge of the tree; with no `id`, the consumer. */
export function walk(layout: CircuitLayout, id: string | null, key: WalkKey): string | null {
  const at = new Map(layout.placed.map((p) => [p.node.id, p]));
  const here = id ? at.get(id) : undefined;
  if (!here) return layout.placed[0]?.node.id ?? null;
  const bySlot = (ids: string[]) =>
    ids
      .map((i) => at.get(i))
      .filter((p): p is Placed => !!p)
      .sort((a, b) => a.slot - b.slot);
  if (key === 'left') return here.node.parent ?? here.node.id;
  if (key === 'right') return bySlot(here.node.children)[0]?.node.id ?? here.node.id;
  const parent = here.node.parent ? at.get(here.node.parent) : undefined;
  const siblings = parent ? bySlot(parent.node.children) : [here];
  const i = siblings.findIndex((p) => p.node.id === here.node.id);
  const next = siblings[key === 'up' ? i - 1 : i + 1];
  return next?.node.id ?? here.node.id;
}

/** The circuits of one consumer, in id order: a circuit's layout family. */
export function familyOf<T extends { id: number; body: Rec }>(all: T[], consumer: string | null): T[] {
  return all
    .filter((c) => str(c.body.consumer) === consumer)
    .sort((a, b) => a.id - b.id);
}

/** The scale the graph is drawn at: `fit` fits the box's width (never above 1, never below
 *  MIN_SCALE: past that the graph scrolls), a number is a zoom the user chose. */
export function scaleFor(zoom: 'fit' | number, layoutWidth: number, boxWidth: number | null): number {
  if (zoom !== 'fit') return zoom;
  if (boxWidth === null || boxWidth <= 0) return 1;
  return Math.max(MIN_SCALE, Math.min(1, boxWidth / layoutWidth));
}
