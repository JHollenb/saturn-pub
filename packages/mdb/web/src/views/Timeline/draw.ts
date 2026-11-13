// Canvas drawing for the timeline grid. Pure: everything comes from the block and the options.
// Reached cells are solid (measured); a measured zero is a small solid dot; cells the recording
// never reached are EMPTY with a dashed outline (unresolved), never drawn as zero. No numbers
// are drawn on the canvas; the hover line prints values through NumView.
import { max as d3max } from 'd3-array';
import { scaleLinear } from 'd3-scale';
import type { TimelineBlock } from '../../api/types.gen';
import { tickInferred } from './axis';

export interface Geometry {
  cw: number;
  ch: number;
  left: number;
  top: number;
}

export interface Palette {
  ink: string;
  faint: string;
  cell: string;
  cursor: string;
  symbol: string;
  bg: string;
}

export function readPalette(el: Element): Palette {
  const css = getComputedStyle(el);
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  return {
    ink: v('--ink', '#222'),
    faint: v('--ink-3', '#999'),
    cell: v('--cell', '#333'),
    cursor: v('--cursor', '#000'),
    symbol: v('--role-symbol', '#2f6fb3'),
    bg: v('--panel', '#fff'),
  };
}

export function cellAt(g: Geometry, block: TimelineBlock, px: number, py: number) {
  const x = Math.floor((px - g.left) / g.cw) + block.win.x0;
  const y = Math.floor((py - g.top) / g.ch) + block.win.y0;
  if (px < g.left || py < g.top || x >= block.win.x1 || y >= block.win.y1) return null;
  const nx = block.axes[0]?.ticks.length ?? 0;
  const ny = block.axes[1]?.ticks.length ?? 0;
  return x < nx && y < ny ? { x, y } : null;
}

export function drawTimeline(
  ctx: CanvasRenderingContext2D,
  block: TimelineBlock,
  g: Geometry,
  p: Palette,
  cursor: { x: number | null; y: number | null },
  focus: { x: number; y: number } | null = null,
) {
  const { win } = block;
  const yAxis = block.axes[1];
  const nx = Math.min(win.x1, block.axes[0]?.ticks.length ?? 0);
  const ny = Math.min(win.y1, yAxis?.ticks.length ?? 0);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.font = '11px ui-monospace, Menlo, monospace';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = p.faint;
  for (let y = win.y0; y < ny; y++) {
    const tick = yAxis?.ticks[y];
    const cy = g.top + (y - win.y0 + 0.5) * g.ch;
    // An inferred tick carries a hollow mark (derived), so it is never read as a measured layer.
    const inferred = !!tick && tickInferred(tick);
    if (inferred) {
      ctx.beginPath();
      ctx.arc(7, cy, 2.6, 0, Math.PI * 2);
      ctx.strokeStyle = p.faint;
      ctx.stroke();
    }
    ctx.fillText(tick?.label ?? '', inferred ? 13 : 4, cy);
  }
  const listed = new Map(block.cells.map((c) => [`${c.x},${c.y}`, c]));
  const alpha = scaleLinear()
    .domain([0, Math.max(1, d3max(block.cells, (c) => c.v ?? 0) ?? 1)])
    .range([0.3, 1]);
  for (let y = win.y0; y < ny; y++) {
    for (let x = win.x0; x < nx; x++) {
      const px = g.left + (x - win.x0) * g.cw;
      const py = g.top + (y - win.y0) * g.ch;
      const cell = listed.get(`${x},${y}`);
      if (!cell || cell.v === null) {
        ctx.save();
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = p.faint;
        ctx.globalAlpha = 0.45;
        ctx.strokeRect(px + 2.5, py + 2.5, g.cw - 5, g.ch - 5);
        ctx.restore();
        continue;
      }
      ctx.fillStyle = p.cell;
      if (cell.v > 0) {
        ctx.globalAlpha = alpha(cell.v);
        ctx.fillRect(px + 2, py + 2, g.cw - 4, g.ch - 4);
        ctx.globalAlpha = 1;
      } else {
        ctx.beginPath();
        ctx.arc(px + g.cw / 2, py + g.ch / 2, 1.8, 0, Math.PI * 2);
        ctx.fill();
      }
      drawGlyphs(ctx, cell.glyphs, px, py, g, p);
    }
  }
  if (cursor.x !== null && cursor.y !== null) {
    const px = g.left + (cursor.x - win.x0) * g.cw;
    const py = g.top + (cursor.y - win.y0) * g.ch;
    ctx.strokeStyle = p.cursor;
    ctx.lineWidth = 2;
    ctx.strokeRect(px + 1, py + 1, g.cw - 2, g.ch - 2);
    ctx.lineWidth = 1;
  }
  if (focus) {
    // Keyboard focus ring: a dashed outline outside the cell, distinct from the solid cursor.
    const px = g.left + (focus.x - win.x0) * g.cw;
    const py = g.top + (focus.y - win.y0) * g.ch;
    ctx.save();
    ctx.setLineDash([3, 2]);
    ctx.strokeStyle = p.ink;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(px + 0.25, py + 0.25, g.cw - 0.5, g.ch - 0.5);
    ctx.restore();
  }
}

function drawGlyphs(
  ctx: CanvasRenderingContext2D,
  glyphs: string[],
  px: number,
  py: number,
  g: Geometry,
  p: Palette,
) {
  if (glyphs.includes('selected')) {
    ctx.strokeStyle = p.bg;
    ctx.strokeRect(px + 4.5, py + 4.5, g.cw - 9, g.ch - 9);
  }
  if (glyphs.includes('fork')) {
    ctx.fillStyle = p.ink;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px + 6, py);
    ctx.lineTo(px, py + 6);
    ctx.fill();
  }
  if (glyphs.includes('symbol')) {
    ctx.fillStyle = p.symbol;
    ctx.beginPath();
    ctx.arc(px + g.cw - 4, py + 4, 2.6, 0, Math.PI * 2);
    ctx.fill();
  }
  if (glyphs.includes('record')) {
    ctx.fillStyle = p.ink;
    ctx.fillRect(px + g.cw - 6, py + g.ch - 6, 4, 4);
  }
}
