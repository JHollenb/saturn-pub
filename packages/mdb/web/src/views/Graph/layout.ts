// Deterministic clock layout: a node that sits at a clock cell is drawn at that cell; nodes
// without a cell go in lanes (one per role, server-assigned) below, in server order.
import type { CircuitOverlay, GraphNode } from '../../api/types.gen';

export const PAD = 28;
/** Height of one label line; nodes in one cell stack at this pitch so labels never overlap. */
export const LH = 20;
/** Estimated width of one label character at font-size 10 (monospace-ish upper bound). */
export const CHAR_W = 6;
export const MAX_LABEL = 22;
/** Gap from a node centre to the left edge of its label. */
export const LABEL_DX = 11;
export const NODE_R = 9;

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Placed {
  node: GraphNode;
  cx: number;
  cy: number;
  /** Label as drawn (truncated with an ellipsis); the full text stays in the node title. */
  label: string;
  labelBox: Box;
  /** Marker (circle or square) bounding box. */
  markBox: Box;
}

export interface LegendLine {
  id: string;
  text: string;
  full: string;
  box: Box;
}

export interface Layout {
  placed: Placed[];
  legend: LegendLine[];
  width: number;
  height: number;
  /** Top of the node area (below the legend). */
  top: number;
}

export function truncate(text: string, max = MAX_LABEL): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}\u2026`;
}

function byKey<T>(items: T[], key: (item: T) => string): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const k = key(item);
    const group = groups.get(k);
    if (group) group.push(item);
    else groups.set(k, [item]);
  }
  return groups;
}

/** Deterministic layout. Nodes at a clock cell go to that cell's column; several nodes in one
 *  cell stack vertically at label pitch; each row is as tall as its fullest cell. Nodes without a
 *  cell go in lanes (server-assigned), wrapped into a grid of the same column pitch. Overlay
 *  labels are one line each in a legend band above the nodes, never at node positions. The same
 *  input always yields the same coordinates (no randomness, no dependence on previous renders). */
export function layout(nodes: GraphNode[], overlays: CircuitOverlay[] = []): Layout {
  const labels = nodes.map((n) => truncate(n.label));
  const longest = Math.max(4, ...labels.map((l) => l.length));
  const pitch = LABEL_DX + longest * CHAR_W + 24;
  const cell = nodes.filter((n) => n.x !== null && n.y !== null);
  const xs = [...new Set(cell.map((n) => n.x ?? 0))].sort((a, b) => a - b);
  const ys = [...new Set(cell.map((n) => n.y ?? 0))].sort((a, b) => a - b);
  const groups = byKey(cell, (n) => `${n.x},${n.y}`);
  const rowTop = new Map<number, number>();
  let y = 0;
  for (const row of ys) {
    rowTop.set(row, y);
    const fullest = Math.max(...xs.map((x) => groups.get(`${x},${row}`)?.length ?? 0));
    y += fullest * LH + 10;
  }
  const lanes = byKey(
    nodes.filter((n) => n.x === null || n.y === null),
    (n) => String(n.lane),
  );
  const laneIds = [...lanes.keys()].map(Number).sort((a, b) => a - b);
  const cols = Math.max(xs.length, 3);
  const slots = new Map<string, { col: number; y: number }>();
  for (const lane of laneIds) {
    const members = lanes.get(String(lane)) ?? [];
    members.forEach((n, i) => slots.set(n.id, { col: i % cols, y: y + Math.floor(i / cols) * LH }));
    y += Math.ceil(members.length / cols) * LH + 10;
  }
  const legendLines = overlays.map((o) => ({
    id: o.id,
    full: `${o.label} \u00b7 ${o.verdict}`,
  }));
  const top = PAD + (legendLines.length ? legendLines.length * LH + 8 : 0);
  const legendWidth = Math.max(0, ...legendLines.map((l) => Math.min(l.full.length, 48) * CHAR_W));
  const legend: LegendLine[] = legendLines.map((l, i) => ({
    id: l.id,
    full: l.full,
    text: truncate(l.full, 48),
    box: {
      x: PAD + 30,
      y: PAD + i * LH - LH / 2,
      w: Math.min(l.full.length, 48) * CHAR_W,
      h: LH,
    },
  }));
  const placedAt = (node: GraphNode, i: number, cx: number, cy: number): Placed => ({
    node,
    cx,
    cy,
    label: labels[i] ?? '',
    labelBox: {
      x: cx + LABEL_DX,
      y: cy - LH / 2,
      w: (labels[i] ?? '').length * CHAR_W,
      h: LH,
    },
    markBox: { x: cx - NODE_R, y: cy - NODE_R, w: 2 * NODE_R, h: 2 * NODE_R },
  });
  const cellIndex = new Map<string, number>();
  const placed = nodes.map((node, i) => {
    if (node.x !== null && node.y !== null) {
      const key = `${node.x},${node.y}`;
      const k = cellIndex.get(key) ?? 0;
      cellIndex.set(key, k + 1);
      return placedAt(
        node,
        i,
        PAD + xs.indexOf(node.x) * pitch + NODE_R,
        top + (rowTop.get(node.y) ?? 0) + k * LH + LH / 2,
      );
    }
    const slot = slots.get(node.id) ?? { col: 0, y: 0 };
    return placedAt(node, i, PAD + slot.col * pitch + NODE_R, top + slot.y + LH / 2);
  });
  const width = Math.max(
    240,
    PAD * 2 + Math.max(xs.length, laneIds.length ? cols : 0) * pitch,
    PAD * 2 + 30 + legendWidth,
  );
  const height = top + Math.max(y, LH) + PAD;
  return { placed, legend, width, height, top };
}

/** Convex hull (monotone chain) of points, for overlay outlines. */
export function hull(points: [number, number][]): [number, number][] {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: [number, number][] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2]!, lower[lower.length - 1]!, p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper: [number, number][] = [];
  for (const p of [...pts].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2]!, upper[upper.length - 1]!, p) <= 0)
      upper.pop();
    upper.push(p);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

export function overlayPath(overlay: CircuitOverlay, at: Map<string, Placed>): string | null {
  const pad = 14;
  const pts: [number, number][] = [];
  for (const id of overlay.nodes) {
    const p = at.get(id);
    if (!p) continue;
    pts.push(
      [p.cx - pad, p.cy - pad],
      [p.cx + pad, p.cy - pad],
      [p.cx - pad, p.cy + pad],
      [p.cx + pad, p.cy + pad],
    );
  }
  if (!pts.length) return null;
  const h = hull(pts);
  return `M${h.map(([x, y]) => `${x},${y}`).join('L')}Z`;
}

// Server roles (views/graph.py `_role`) -> role hue tokens.
const ROLE_VARS: Record<string, string> = {
  symbol: '--role-symbol',
  cursor: '--role-input',
  readout: '--role-output',
  carrier: '--role-intervention',
  state: '--role-state',
  cache: '--role-cache',
};

export function roleColor(role: string): string {
  return `var(${ROLE_VARS[role] ?? '--role-other'})`;
}

export const EDGE_DASH: Record<string, string | undefined> = {
  dataflow: undefined,
  read: '1 3',
  kv_write: '6 2 1 2',
  attribution: '8 3',
  intervention: undefined,
  declared: '3 3',
  unknown: '1 5',
};
