// Render one reduce result. Every value goes through NumView with the class the server gave.
import { max as d3max } from 'd3-array';
import { scaleBand, scaleLinear } from 'd3-scale';
import { useEffect, useRef, useState } from 'react';
import type { Ev, HistogramResult, Num, ReduceResult, TileResult } from '../../api/types.gen';
import { Count, NumView } from '../../chrome/NumView';
import { bytesOf, decodeF32 } from './decode';

const MAX_SHOWN = 64;

function n(
  v: number | undefined,
  ev: Ev,
  unit: string | null = null,
  proof: string | null = null,
): Num {
  if (v === undefined || !Number.isFinite(v)) {
    return {
      v: null,
      ev: 'unresolved',
      unit,
      proof,
      ptr: null,
      reason: 'non-finite value',
      note: null,
      recorded_authority: null,
    };
  }
  return { v, ev, unit, proof, ptr: null, reason: null, note: null, recorded_authority: null };
}

export function ReduceResultView({ r }: { r: ReduceResult }) {
  const head = (
    <div className="faint">
      {r.op} of [{r.shape.map(String).join('×')}] {r.dtype}
      {r.note ? ` · ${r.note}` : ''}
    </div>
  );
  switch (r.op) {
    case 'stats':
      return (
        <div>
          {head}
          <table>
            <tbody>
              {(['count', 'finite', 'min', 'max', 'mean', 'std', 'absmax', 'l2'] as const).map(
                (k) => (
                  <tr key={k}>
                    <td className="muted">{k}</td>
                    <td>
                      <NumView n={r[k]} showReason />
                    </td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
      );
    case 'topk':
      return (
        <div>
          {head}
          <div className="faint">ranked by {r.score}</div>
          <table>
            <tbody>
              {r.entries.map((e, i) => (
                <tr key={i}>
                  <td className="mono">[{e.index.map(String).join(',')}]</td>
                  <td>
                    <NumView n={e.v} showReason />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case 'slice':
    case 'project': {
      const values = decodeF32(r.data);
      const ev = r.ev;
      return (
        <div>
          {head}
          <div className="faint">
            out [{r.outShape.map(String).join('×')}]{' '}
            {r.op === 'project' && (
              <>
                · direction norm <NumView n={r.directionNorm} />
              </>
            )}
          </div>
          <table>
            <tbody>
              {Array.from(values.slice(0, MAX_SHOWN)).map((v, i) => (
                <tr key={i}>
                  <td className="mono faint">
                    <Count n={i} />
                  </td>
                  <td>
                    <NumView
                      n={n(v, ev, null, r.op === 'project' ? 'dot product with direction' : null)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {values.length > MAX_SHOWN && (
            <div className="faint">
              +<Count n={values.length - MAX_SHOWN} /> more values (narrow the selector)
            </div>
          )}
        </div>
      );
    }
    case 'histogram':
      return (
        <div>
          {head}
          <Histogram r={r} />
        </div>
      );
    case 'tile':
      return (
        <div>
          {head}
          <Tile r={r} />
        </div>
      );
  }
}

function Histogram({ r }: { r: HistogramResult }) {
  const [hover, setHover] = useState<number | null>(null);
  const w = 260;
  const h = 80;
  const y = scaleLinear()
    .domain([0, Math.max(1, d3max(r.counts) ?? 1)])
    .range([0, h]);
  const x = scaleBand<number>()
    .domain(r.counts.map((_, i) => i))
    .range([0, w])
    .paddingInner(0.08);
  const edges = r.edges.vs;
  return (
    <div>
      <svg
        width={w}
        height={h}
        role="img"
        aria-label="histogram"
        onMouseLeave={() => setHover(null)}
      >
        {r.counts.map((c, i) => (
          <rect
            key={i}
            x={x(i) ?? 0}
            y={h - y(c)}
            width={Math.max(1, x.bandwidth())}
            height={y(c)}
            fill={hover === i ? 'var(--ink)' : 'var(--ink-2)'}
            onMouseEnter={() => setHover(i)}
          />
        ))}
      </svg>
      <div className="muted" style={{ minHeight: 20 }}>
        {hover !== null ? (
          <>
            [<NumView n={n(edges[hover], r.edges.ev, null, r.edges.proof)} bare />,{' '}
            <NumView n={n(edges[hover + 1], r.edges.ev, null, r.edges.proof)} bare />){' '}
            <Count n={r.counts[hover] ?? 0} /> values
          </>
        ) : (
          <>
            <Count n={r.bins} /> bins · <Count n={r.nonfinite} /> non-finite
          </>
        )}
      </div>
    </div>
  );
}

function Tile({ r }: { r: TileResult }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const scale = Math.max(1, Math.floor(256 / Math.max(r.w, r.h, 1)));
  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx || !r.w || !r.h) return;
    const px = bytesOf(r.data);
    const img = ctx.createImageData(r.w, r.h);
    for (let i = 0; i < px.length; i++) {
      const b = px[i] ?? 0;
      // Byte 0 = no finite value: transparent (empty), never drawn as the domain minimum.
      const grey = b === 0 ? 0 : Math.round(((b - 1) / 254) * 255);
      img.data[i * 4] = grey;
      img.data[i * 4 + 1] = grey;
      img.data[i * 4 + 2] = grey;
      img.data[i * 4 + 3] = b === 0 ? 0 : 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [r]);
  const [lo, hi] = r.domain.vs;
  return (
    <div>
      <canvas
        ref={ref}
        width={r.w}
        height={r.h}
        style={{
          width: r.w * scale,
          height: r.h * scale,
          imageRendering: 'pixelated',
          border: '1px dashed var(--ink-3)',
          display: 'block',
        }}
        aria-label="tile"
      />
      <div className="muted">
        {r.scale} · black <NumView n={n(lo, r.domain.ev, null, r.domain.proof)} bare /> → white{' '}
        <NumView n={n(hi, r.domain.ev, null, r.domain.proof)} bare /> · {r.agg} · level{' '}
        <Count n={r.level} /> of [{r.full.map(String).join('×')}] · <Count n={r.nonfinite} /> empty
        blocks
      </div>
    </div>
  );
}
