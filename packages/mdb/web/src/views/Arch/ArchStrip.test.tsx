// The architecture strip against a frame and a clock a real `mdb serve --tiny qwen2` returned
// (fixtures/, recorded by the script named in fixtures/commands.json): two stepped rows, three
// prefilled rows, an edit committed on t3 and one staged on t4, two checkpoints.
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import type { LiveFrame, LiveTimeline } from '../../api/types.gen';
import { slotFrom } from '../../state/opFixture';
import { SeekLine } from '../Live/Live';
import { ArchStrip, cellState, lanes, laneOf } from './ArchStrip';
import frameFx from './fixtures/frame.json';
import seekBack from './fixtures/seek_back.json';
import seekRefused from './fixtures/seek_refused.json';
import timelineFx from './fixtures/timeline.json';

const frame = (frameFx as unknown as { data: LiveFrame }).data;
const timeline = (timelineFx as unknown as { data: LiveTimeline }).data;

function render(circuit: string[] = [], busy = false): string {
  return renderToStaticMarkup(
    <ArchStrip frame={frame} timeline={timeline} circuit={circuit} busy={busy} seek={() => undefined} />,
  );
}

function cellOf(html: string, key: string): string {
  const at = html.indexOf(`data-cell="${key}"`);
  expect(at, key).toBeGreaterThan(-1);
  const start = html.lastIndexOf('<button', at);
  return html.slice(start, html.indexOf('</button>', at));
}

describe('ArchStrip', () => {
  it('has one lane per layer plus the row input and output, from the clock', () => {
    const ls = lanes(timeline.stops, 2);
    expect(ls.map((l) => l.key)).toEqual(['in', 'L0', 'L1', 'out']);
    expect(ls[0]!.stops).toEqual(['start', 'emb']);
    expect(ls[3]!.stops).toEqual(['fn', 'logits', 'sample']);
    expect(laneOf('t4.L0.p')).toEqual({ row: 4, lane: 'L0' });
    expect(laneOf('t3.logits')).toEqual({ row: 3, lane: 'out' });
    expect(laneOf('t3.M[4,9]')).toEqual({ row: 3, lane: 'out' });
    expect(laneOf('lm_head')).toBeNull();
  });

  it('draws materialised cells full or partial and unmaterialised cells empty, never zero', () => {
    const html = render();
    expect(cellOf(html, '3|L1')).toContain('data-state="full"');
    expect(cellOf(html, '3|out')).toContain('data-state="full"');
    expect(cellOf(html, '4|L0')).toContain('data-state="partial"');
    expect(cellOf(html, '4|L1')).toContain('data-state="empty"');
    expect(cellOf(html, '4|L1')).toContain('not computed (no value)');
    // prefilled rows: cache only on layers, nothing at the input or output
    expect(cellOf(html, '0|L0')).toContain('data-state="cache"');
    expect(cellOf(html, '0|in')).toContain('data-state="empty"');
    expect(html).not.toMatch(/data-state="empty"[^>]*>0</);
    expect(cellState(null, timeline.stops, lanes(timeline.stops, 2)[1]!)).toBe('cache');
  });

  it('marks the cursor, edits and checkpoints in text as well as style', () => {
    const html = render();
    const cursor = cellOf(html, '4|L0');
    expect(cursor).toContain('arch-cursor');
    expect(cursor).toContain('cursor here');
    expect(cursor).toContain('✎'); // the staged L0.v edit
    expect(cursor).toContain('⚑'); // checkpoint mid at t4.L0.p
    expect(cellOf(html, '3|L1')).toContain('edit L1.m');
    expect(cellOf(html, '3|out')).toContain('checkpoint here');
  });

  it("marks the selected circuit's nodes and the rows they read", () => {
    const html = render(['t3.M[4,9]', 't3.L1.A[h0]', 't3.L1.p[h0,src=1]']);
    expect(cellOf(html, '3|out')).toContain('◆');
    expect(cellOf(html, '3|L1')).toContain('◆');
    expect(cellOf(html, '1|L1')).toContain('○');
    expect(cellOf(html, '1|L1')).toContain('read by the selected circuit');
  });

  it("expands the current layer's port program, computed ports solid, the rest empty", () => {
    const html = render();
    expect(html).toContain('port program of t4.L0');
    for (const p of frame.ports) {
      expect(html).toContain(`title="${p.addr}: ${p.computed ? 'computed' : 'not computed yet'}`);
    }
  });

  it('every cell is a button that asks before seeking; a busy page disables them', () => {
    expect(render()).toContain('click: seek t4.L0.h (asks first)');
    expect(render([], true)).toMatch(/<button[^>]*disabled=""[^>]*data-cell=/);
  });

  it('no run yet: says so', () => {
    const html = renderToStaticMarkup(
      <ArchStrip frame={frame} timeline={null} circuit={[]} busy={false} seek={() => undefined} />,
    );
    expect(html).toContain('architecture: no run yet');
  });
});

describe('seek confirmation', () => {
  it('names the move and where it came from before anything runs', () => {
    const html = renderToStaticMarkup(
      <SeekLine
        seek={{ target: 't3.L1.p', from: 'architecture cell t3 L1' }}
        where="t4.L0.p"
        busy={false}
        onConfirm={() => undefined}
        onCancel={() => undefined}
      />,
    );
    expect(html).toContain('data-testid="seek-line"');
    expect(html).toContain('seek t3.L1.p</code> moves the session from <code>t4.L0.p');
    expect(html).toContain('architecture cell t3 L1');
    expect(html).toMatch(/>cancel<\/button>/);
  });

  it('the seek op answers resolved with its moves, or refused with the reason', () => {
    const back = slotFrom(seekBack);
    expect(back.state).toBe('done');
    const refused = slotFrom(seekRefused);
    expect(refused.state).toBe('refused');
    if (refused.state === 'refused') expect(refused.reason).toContain('t0 was prefilled');
  });
});
