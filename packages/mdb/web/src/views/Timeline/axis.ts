// Honesty about the depth axis. `Capabilities.layers` is a labelled number: measured when the
// layer count was recorded, derived-local (unit "layers (at least)") when inferred, unresolved when
// unknown (then the server sends no layer ticks at all). The UI only draws ticks the server sent;
// a tick the server marks `inferred` is drawn hollow (derived), never as measured.
import type { AxisTick, Num } from '../../api/types.gen';

export type LayerStatus = 'stated' | 'inferred' | 'unknown';

export function axisStatus(layers?: Num | null): LayerStatus {
  if (!layers) return 'stated';
  if (layers.ev === 'measured') return 'stated';
  if (layers.ev === 'unresolved' || layers.ev === 'refused' || layers.v === null) return 'unknown';
  return 'inferred';
}

export function tickInferred(tick: AxisTick): boolean {
  return tick.inferred;
}
