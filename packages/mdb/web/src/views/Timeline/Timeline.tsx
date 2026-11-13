// Timeline: a canvas grid over the first two server-supplied clock axes. No axis name is assumed;
// cell addresses are composed from the server's tick fragments.
import { useEffect, useMemo, useRef, useState } from 'react';
import { branchOf, composeAddress, originOf } from '../../api/address';
import type { Frame, TimelineBlock } from '../../api/types.gen';
import { OmittedNote, Show } from '../../chrome/Loaded';
import { EvidenceMark } from '../../chrome/EvidenceMark';
import { Count, NumView } from '../../chrome/NumView';
import { useCursor } from '../../state/cursor';
import { useSession } from '../../state/session';
import { useView } from '../../state/records';
import { useUi } from '../../state/ui';
import { axisStatus } from './axis';
import { BranchRail } from './BranchRail';
import { cellAt, drawTimeline, readPalette, type Geometry } from './draw';

const CH = 18;
const LEFT = 64;
const TOP = 4;

export function Timeline() {
  const sid = useCursor((s) => s.sid);
  const addr = useCursor((s) => s.addr);
  const lens = useUi((s) => s.lens);
  const set = useUi((s) => s.set);
  const branch = branchOf(addr);
  const base = sid ? `/s/${encodeURIComponent(sid)}` : null;
  const block = useView('timeline', base && `${base}/timeline`, { lens, branch });
  const frame = useView('frame', base && addr ? `${base}/frame` : null, { addr });
  if (!sid) return null;
  return (
    <section className="panel" aria-label="timeline">
      <h2>
        Timeline
        <BranchRail sid={sid} />
        <span style={{ flex: 1 }} />
        {block.state === 'ok' && (
          <select
            aria-label="lens"
            value={lens}
            onChange={(e) => set({ lens: e.target.value })}
            title={block.env.data.lenses.find((l) => l.id === lens)?.note}
          >
            {block.env.data.lenses.map((l) => (
              <option key={l.id} value={l.id} title={l.note}>
                {l.label}
                {l.hasData ? '' : ' (no data)'}
              </option>
            ))}
          </select>
        )}
      </h2>
      <div className="panel-body">
        <Show what="timeline" loaded={block}>
          {(data, env) => (
            <>
              <Grid block={data} frame={frame.state === 'ok' ? frame.env.data : null} />
              <OmittedNote env={env} />
            </>
          )}
        </Show>
      </div>
    </section>
  );
}

function Grid({ block, frame }: { block: TimelineBlock; frame: Frame | null }) {
  const addr = useCursor((s) => s.addr);
  const go = useCursor((s) => s.go);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const [focusCell, setFocusCell] = useState<{ x: number; y: number } | null>(null);
  const [focused, setFocused] = useState(false);
  const [xAxis, yAxis] = block.axes;
  const layers = useSession((s) => s.caps?.layers ?? null);
  const depth = axisStatus(layers);
  const nx = Math.min(block.win.x1, xAxis?.ticks.length ?? 0) - block.win.x0;
  const ny = Math.min(block.win.y1, yAxis?.ticks.length ?? 0) - block.win.y0;
  const cw = Math.max(14, Math.min(28, Math.floor(900 / Math.max(1, nx))));
  const g: Geometry = useMemo(() => ({ cw, ch: CH, left: LEFT, top: TOP }), [cw]);
  const cursor = {
    x: frame && xAxis ? (frame.pos[xAxis.id] ?? null) : null,
    y: frame && yAxis ? (frame.pos[yAxis.id] ?? null) : null,
  };
  const width = LEFT + nx * cw + 4;
  const height = TOP + ny * CH + 4;
  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext('2d');
    if (!el || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = width * dpr;
    el.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawTimeline(ctx, block, g, readPalette(el), cursor, focused ? focusCell : null);
  });
  const listed = useMemo(
    () => new Map(block.cells.map((c) => [`${c.x},${c.y}`, c])),
    [block.cells],
  );
  const probe = hover ?? (focused ? focusCell : null);
  const hovered = probe ? listed.get(`${probe.x},${probe.y}`) : undefined;
  const origin = addr ? originOf(addr) : '';
  const start = { x: cursor.x ?? block.win.x0, y: cursor.y ?? block.win.y0 };
  const pick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return cellAt(g, block, e.clientX - r.left, e.clientY - r.top);
  };
  return (
    <div>
      <TokenStrip block={block} cw={cw} cursorX={cursor.x} origin={origin} cursorY={cursor.y} />
      <div style={{ overflow: 'auto' }}>
        <canvas
          ref={canvas}
          style={{ width, height, display: 'block', cursor: 'pointer' }}
          tabIndex={0}
          role="application"
          data-own-keys=""
          aria-label={`${xAxis?.label ?? ''} by ${yAxis?.label ?? ''} grid; arrow keys move the focus cell, Enter goes to its address`}
          onFocus={() => {
            setFocused(true);
            setFocusCell((c) => c ?? start);
          }}
          onBlur={() => setFocused(false)}
          onKeyDown={(e) => {
            if (e.metaKey || e.ctrlKey || e.altKey) return;
            const cur = focusCell ?? start;
            const step: Record<string, [number, number]> = {
              ArrowLeft: [-1, 0],
              ArrowRight: [1, 0],
              ArrowUp: [0, -1],
              ArrowDown: [0, 1],
            };
            const d = step[e.key];
            if (d) {
              e.preventDefault();
              setFocusCell({
                x: Math.min(block.win.x0 + nx - 1, Math.max(block.win.x0, cur.x + d[0])),
                y: Math.min(block.win.y0 + ny - 1, Math.max(block.win.y0, cur.y + d[1])),
              });
            } else if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              const a = composeAddress(block.axes, cur.x, cur.y, origin);
              if (a) go(a);
            }
          }}
          onMouseMove={(e) => setHover(pick(e))}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => {
            const c = pick(e);
            const a = c && composeAddress(block.axes, c.x, c.y, origin);
            if (a) go(a);
          }}
        />
      </div>
      {(yAxis?.note || layers) && (
        <div className="faint" data-testid="axis-note" data-status={depth}>
          {depth !== 'stated' && (
            <EvidenceMark ev={depth === 'inferred' ? 'derived-local' : 'unresolved'} />
          )}{' '}
          {yAxis?.label}: {yAxis?.note}
          {layers && (
            <>
              {' '}
              layers <NumView n={layers} showReason />
            </>
          )}
          {depth === 'inferred' && ' (layer count inferred; hollow ticks are inferred)'}
          {depth === 'unknown' && ' (layer count unknown)'}
        </div>
      )}
      <div className="muted" style={{ minHeight: 20 }} aria-live="polite">
        {probe ? (
          <>
            <code>
              {composeAddress(block.axes, probe.x, probe.y, origin) ??
                `${yAxis?.ticks[probe.y]?.label ?? ''}: ${yAxis?.ticks[probe.y]?.detail ?? 'no address'}`}
            </code>{' '}
            {hovered ? (
              <>
                <NumView
                  n={{
                    v: hovered.v,
                    ev: hovered.ev,
                    unit: block.unit,
                    proof: null,
                    ptr: null,
                    reason: null,
                    note: null,
                    recorded_authority: null,
                  }}
                />{' '}
                {hovered.glyphs.join(' ')}
              </>
            ) : (
              <NumView n={block.fill} showReason />
            )}
          </>
        ) : (
          <span className="faint">
            {xAxis?.label} × {yAxis?.label}; <Count n={block.cells.length} /> reached cells; empty
            dashed cells: {block.fill.reason}
          </span>
        )}
      </div>
    </div>
  );
}

function TokenStrip({
  block,
  cw,
  cursorX,
  cursorY,
  origin,
}: {
  block: TimelineBlock;
  cw: number;
  cursorX: number | null;
  cursorY: number | null;
  origin: string;
}) {
  const go = useCursor((s) => s.go);
  if (!block.tokens) return null;
  return (
    <div style={{ display: 'flex', marginLeft: LEFT, height: 18 }} aria-label="tokens">
      {block.tokens.map((t) => (
        <button
          key={t.x}
          type="button"
          className="row-button mono"
          style={{
            width: cw,
            flex: 'none',
            fontSize: 9,
            textAlign: 'center',
            overflow: 'hidden',
            border: t.ev === 'unresolved' ? '1px dashed var(--ink-3)' : undefined,
            fontWeight: t.x === cursorX ? 700 : 400,
          }}
          title={
            t.id === null
              ? (t.reason ?? 'token unresolved')
              : `token id ${String(t.id)}${t.text ? ` ${JSON.stringify(t.text)}` : ''}`
          }
          onClick={() => {
            const ys = block.axes[1]?.ticks ?? [];
            const y = cursorY ?? ys.findIndex((tick) => tick.addressable);
            const a = composeAddress(block.axes, t.x, Math.max(0, y), origin);
            if (a) go(a);
          }}
        >
          {t.id === null ? ' ' : <Count n={t.id} />}
        </button>
      ))}
    </div>
  );
}
