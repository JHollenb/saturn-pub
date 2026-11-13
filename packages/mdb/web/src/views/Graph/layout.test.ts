import { describe, expect, it } from 'vitest';
import type { Envelope, GraphView } from '../../api/types.gen';
import qwenSession from './fixtures/qwen-session.json';
import qwenSlice from './fixtures/qwen-slice.json';
import tinySession from './fixtures/tiny-session.json';
import tinySlice from './fixtures/tiny-slice.json';
import { layout, truncate, MAX_LABEL, type Box } from './layout';

const FIXTURES: [string, GraphView][] = [
  ['tiny slice', (tinySlice as unknown as Envelope<GraphView>).data],
  ['tiny session', (tinySession as unknown as Envelope<GraphView>).data],
  ['qwen slice', (qwenSlice as unknown as Envelope<GraphView>).data],
  ['qwen session', (qwenSession as unknown as Envelope<GraphView>).data],
];

function overlap(a: Box, b: Box): boolean {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

describe('graph layout', () => {
  it.each(FIXTURES)('%s: no two label boxes (or label and marker) overlap', (_name, view) => {
    const { placed, legend, width, height } = layout(view.nodes, view.overlays);
    const labels = [...placed.map((p) => p.labelBox), ...legend.map((l) => l.box)];
    const marks = placed.map((p) => p.markBox);
    for (let i = 0; i < labels.length; i++) {
      for (let j = i + 1; j < labels.length; j++) {
        expect(overlap(labels[i]!, labels[j]!), `labels ${i} and ${j}`).toBe(false);
      }
      for (let j = 0; j < marks.length; j++) {
        if (placed[j]?.labelBox === labels[i]) continue;
        expect(overlap(labels[i]!, marks[j]!), `label ${i} over marker ${j}`).toBe(false);
      }
    }
    for (const b of [...labels, ...marks]) {
      expect(b.x).toBeGreaterThanOrEqual(0);
      expect(b.y).toBeGreaterThanOrEqual(0);
      expect(b.x + b.w).toBeLessThanOrEqual(width);
      expect(b.y + b.h).toBeLessThanOrEqual(height);
    }
  });

  it.each(FIXTURES)('%s: markers never overlap each other', (_name, view) => {
    const { placed } = layout(view.nodes, view.overlays);
    for (let i = 0; i < placed.length; i++) {
      for (let j = i + 1; j < placed.length; j++) {
        expect(overlap(placed[i]!.markBox, placed[j]!.markBox)).toBe(false);
      }
    }
  });

  it.each(FIXTURES)(
    '%s: deterministic, and independent of unrelated input state',
    (_name, view) => {
      const a = layout(view.nodes, view.overlays);
      const b = layout(structuredClone(view.nodes), structuredClone(view.overlays));
      expect(b).toEqual(a);
      expect(a.placed.map((p) => p.node.id)).toEqual(view.nodes.map((n) => n.id));
    },
  );

  it('truncates long labels with an ellipsis and keeps the node text intact', () => {
    const long = 'x'.repeat(60);
    expect(truncate(long).length).toBe(MAX_LABEL);
    expect(truncate(long).endsWith('…')).toBe(true);
    expect(truncate('short')).toBe('short');
  });
});
