// Graph: the recorded state/action graph around the cursor, as SVG. Node colour = role; node
// fill pattern = evidence class; edge dash = edge type; hulls = circuit overlays with verdicts.
import { useId, useMemo, useState } from 'react';
import type { GraphView } from '../../api/types.gen';
import { OmittedNote, Show } from '../../chrome/Loaded';
import { Count, NumView } from '../../chrome/NumView';
import { markStyle } from '../../chrome/evidence';
import { useCursor } from '../../state/cursor';
import { useView } from '../../state/records';
import { useUi } from '../../state/ui';
import {
  EDGE_DASH,
  LABEL_DX,
  LH,
  PAD,
  layout,
  overlayPath,
  roleColor,
  type Placed,
} from './layout';

// Room for the overlay's labelled interaction number beside its legend line.
const OVERLAY_NUM_W = 260;

export function Graph() {
  const sid = useCursor((s) => s.sid);
  const addr = useCursor((s) => s.addr);
  const zoom = useUi((s) => s.zoom);
  const set = useUi((s) => s.set);
  const [slice, setSlice] = useState<string | null>(null);
  const [depth, setDepth] = useState(3);
  const path = sid ? `/s/${encodeURIComponent(sid)}/graph` : null;
  const loaded = useView('graph', path, { addr, zoom, depth, k: 48, slice });
  // One value, one key: highlight by the server's value_key spelling of the cursor.
  const frame = useView('frame', sid && addr ? `/s/${encodeURIComponent(sid)}/frame` : null, {
    addr,
  });
  const key = frame.state === 'ok' ? (frame.env.data.key ?? frame.env.data.addr) : addr;
  if (!sid) return null;
  return (
    <section className="panel panel-col" aria-label="graph">
      <h2>
        Graph
        <select
          aria-label="zoom"
          value={zoom}
          onChange={(e) => set({ zoom: e.target.value as 'slice' | 'session' })}
        >
          <option value="slice">slice</option>
          <option value="session">session</option>
        </select>
        {loaded.state === 'ok' && loaded.env.data.slices.length > 0 && zoom === 'slice' && (
          <select
            aria-label="slice"
            value={slice ?? ''}
            onChange={(e) => setSlice(e.target.value || null)}
          >
            <option value="">slice at cursor</option>
            {loaded.env.data.slices.map((s) => (
              <option key={s.id} value={s.id}>
                {s.component} {s.complete ? '' : '(gaps)'}
              </option>
            ))}
          </select>
        )}
        <label className="muted">
          depth{' '}
          <select
            aria-label="depth"
            value={depth}
            onChange={(e) => setDepth(Number(e.target.value))}
          >
            {[1, 2, 3, 4, 6].map((d) => (
              <option key={d} value={d}>
                {String(d)}
              </option>
            ))}
          </select>
        </label>
      </h2>
      <div className="panel-body panel-col-body">
        <Show what="graph" loaded={loaded}>
          {(g, env) => (
            <>
              <GraphSvg view={g} cursor={key} />
              <div className="faint panel-foot">
                {g.authority} · <Count n={g.nodes.length} /> nodes · <Count n={g.edges.length} />{' '}
                edges
                {g.pruned.map((p) => (
                  <span key={p.group} className="chip">
                    {p.group} +<Count n={p.count} />
                  </span>
                ))}
              </div>
              <OmittedNote env={env} />
            </>
          )}
        </Show>
      </div>
    </section>
  );
}

function GraphSvg({ view, cursor }: { view: GraphView; cursor: string | null }) {
  const go = useCursor((s) => s.go);
  const hatch = `hatch-${useId().replace(/:/g, '')}`;
  const { placed, legend, width, height } = useMemo(
    () => layout(view.nodes, view.overlays),
    [view.nodes, view.overlays],
  );
  const at = useMemo(() => new Map(placed.map((p) => [p.node.id, p])), [placed]);
  const svgWidth = Math.max(
    width,
    ...legend.map((l) =>
      view.overlays.find((o) => o.id === l.id)?.interaction
        ? l.box.x + l.box.w + 8 + OVERLAY_NUM_W
        : 0,
    ),
  );
  if (!view.nodes.length) return <div className="na">The graph at this address has no nodes.</div>;
  return (
    <div className="panel-scroll">
      <svg
        width={svgWidth}
        height={height}
        role="img"
        aria-label="state graph"
        style={{ display: 'block' }}
      >
        <defs>
          <pattern
            id={hatch}
            width="4"
            height="4"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <line x1="0" y1="0" x2="0" y2="4" stroke="var(--ink-2)" strokeWidth="1.2" />
          </pattern>
          <marker
            id={`${hatch}-arrow`}
            viewBox="0 0 6 6"
            refX="6"
            refY="3"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path d="M0,0L6,3L0,6Z" fill="var(--ink-3)" />
          </marker>
        </defs>
        {view.overlays.map((o) => {
          const d = overlayPath(o, at);
          const first = o.nodes.map((id) => at.get(id)).find(Boolean);
          if (!d || !first) return null;
          return (
            <g key={o.id} aria-label={`overlay ${o.label}`}>
              <path d={d} fill="none" stroke="var(--ink-2)" strokeDasharray="5 3" />
            </g>
          );
        })}
        {legend.map((l) => {
          const interaction = view.overlays.find((o) => o.id === l.id)?.interaction;
          return (
            <g key={l.id}>
              <title>{l.full}</title>
              <line
                x1={PAD}
                x2={PAD + 24}
                y1={l.box.y + LH / 2}
                y2={l.box.y + LH / 2}
                stroke="var(--ink-2)"
                strokeDasharray="5 3"
              />
              <text x={l.box.x} y={l.box.y + LH / 2 + 3.5} fontSize={10} fill="var(--ink-2)">
                {l.text}
              </text>
              {interaction ? (
                <foreignObject
                  x={l.box.x + l.box.w + 8}
                  y={l.box.y}
                  width={OVERLAY_NUM_W}
                  height={LH}
                >
                  <span className="overlay-num" style={{ fontSize: 10 }}>
                    interaction <NumView n={interaction} />
                  </span>
                </foreignObject>
              ) : null}
            </g>
          );
        })}
        {view.edges.map((e) => {
          const a = at.get(e.source);
          const b = at.get(e.target);
          if (!a || !b) return null;
          return (
            <line
              key={e.id}
              x1={a.cx}
              y1={a.cy}
              x2={b.cx}
              y2={b.cy}
              stroke="var(--ink-3)"
              strokeWidth={e.type === 'intervention' ? 2.2 : 1}
              strokeDasharray={EDGE_DASH[e.type]}
              markerEnd={`url(#${hatch}-arrow)`}
            >
              <title>{`${e.type} ${e.relation} (${e.evidence})`}</title>
            </line>
          );
        })}
        {placed.map((p) => (
          <NodeMark
            key={p.node.id}
            p={p}
            hatch={hatch}
            active={!!cursor && p.node.addr === cursor}
            onGo={go}
          />
        ))}
      </svg>
    </div>
  );
}

function NodeMark({
  p,
  hatch,
  active,
  onGo,
}: {
  p: Placed;
  hatch: string;
  active: boolean;
  onGo: (a: string) => void;
}) {
  const { node, cx, cy } = p;
  const activate = () => node.addr && onGo(node.addr);
  const color = roleColor(node.role);
  const style = node.ev ? markStyle(node.ev) : 'hollow';
  const fill = style === 'solid' ? color : style === 'hatched' ? `url(#${hatch})` : 'var(--panel)';
  const dash = style === 'dashed-empty' ? '2 2' : undefined;
  const title = `${node.label}\n${node.kind} · role ${node.role}${node.ev ? ` · ${node.ev}` : ''}${node.addr ? `\n${node.addr}` : ''}`;
  const clickable = !!node.addr;
  const common = {
    stroke: color,
    strokeWidth: active ? 3 : 1.6,
    strokeDasharray: dash,
    fill,
    style: { cursor: clickable ? 'pointer' : 'default' },
    onClick: activate,
  };
  return (
    <g
      tabIndex={clickable ? 0 : undefined}
      role={clickable ? 'button' : undefined}
      aria-label={clickable ? title.split('\n')[0] : undefined}
      onKeyDown={(e) => {
        if (clickable && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          activate();
        }
      }}
    >
      <title>{title}</title>
      {node.kind === 'aggregate' ? (
        <rect
          x={cx - 8}
          y={cy - 8}
          width={16}
          height={16}
          rx={4}
          {...common}
          fill="var(--panel-2)"
        />
      ) : node.kind === 'action' ? (
        <rect x={cx - 6} y={cy - 6} width={12} height={12} {...common} />
      ) : (
        <circle cx={cx} cy={cy} r={7} {...common} />
      )}
      <text x={cx + LABEL_DX} y={cy + 3.5} fontSize={10} fill="var(--ink-2)">
        {p.label}
      </text>
    </g>
  );
}
