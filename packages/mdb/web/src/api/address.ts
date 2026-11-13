// The cursor is one occurrence address; the URL fragment is `#/<session>/<address>`.
// The address text is opaque here (the server parses it); the codec must round-trip any text,
// including `~program` suffixes and token selectors such as `" hot"=5012`.
import type { ClockAxis } from './types.gen';

/** Detail selection carried in the fragment after the address: `?record=<id>&symbol=<id>`. */
export interface Detail {
  record: string | null;
  symbol: string | null;
}

export interface Location extends Detail {
  sid: string;
  addr: string | null;
}

export const NO_DETAIL: Detail = { record: null, symbol: null };

/** `#/<session>[/<address>][?record=<id>][&symbol=<id>]`. Session, address and ids are each
 *  encodeURIComponent-ed, so a raw `/` or `?` or `&` never occurs inside one of them and the
 *  address text is recovered byte for byte. */
export function formatFragment(
  loc: { sid: string; addr: string | null } & Partial<Detail>,
): string {
  const sid = encodeURIComponent(loc.sid);
  const base = loc.addr ? `#/${sid}/${encodeURIComponent(loc.addr)}` : `#/${sid}`;
  const parts: string[] = [];
  if (loc.record) parts.push(`record=${encodeURIComponent(loc.record)}`);
  if (loc.symbol) parts.push(`symbol=${encodeURIComponent(loc.symbol)}`);
  return parts.length ? `${base}?${parts.join('&')}` : base;
}

export function parseFragment(hash: string): Location | null {
  const body = hash.replace(/^#\/?/, '');
  if (!body) return null;
  const q = body.indexOf('?');
  const path = q < 0 ? body : body.slice(0, q);
  const queryText = q < 0 ? '' : body.slice(q + 1);
  const cut = path.indexOf('/');
  try {
    const detail: Detail = { record: null, symbol: null };
    for (const part of queryText.split('&')) {
      const eq = part.indexOf('=');
      if (eq < 0) continue;
      const name = part.slice(0, eq);
      const value = decodeURIComponent(part.slice(eq + 1));
      if (name === 'record') detail.record = value || null;
      else if (name === 'symbol') detail.symbol = value || null;
    }
    if (cut < 0) {
      const sid = decodeURIComponent(path);
      return sid ? { sid, addr: null, ...detail } : null;
    }
    const sid = decodeURIComponent(path.slice(0, cut));
    const addr = decodeURIComponent(path.slice(cut + 1));
    return sid ? { sid, addr: addr || null, ...detail } : null;
  } catch {
    return null;
  }
}

/** Split `origin:` from an address (`b2:t3.L1.h` -> `b2:`). Quoted selectors may hold `:`. */
export function originOf(addr: string): string {
  const quote = addr.indexOf('"');
  const colon = addr.indexOf(':');
  if (colon < 0 || (quote >= 0 && quote < colon)) return '';
  const head = addr.slice(0, colon);
  return /^(@[A-Za-z0-9_-]+(\+[A-Za-z0-9_-]+)*|[A-Za-z_][A-Za-z0-9_-]*)$/.test(head)
    ? `${head}:`
    : '';
}

/** The branch named by an address origin, when it is a branch (not `@cut`). */
export function branchOf(addr: string | null): string | null {
  if (!addr) return null;
  const origin = originOf(addr);
  return origin && !origin.startsWith('@') ? origin.slice(0, -1) : null;
}

/** Address of grid cell (x, y) from server-supplied tick fragments; no axis name is assumed.
 *  Null when either tick is a boundary no occurrence address names (`addressable: false`). */
export function composeAddress(
  axes: ClockAxis[],
  x: number,
  y: number,
  origin = '',
): string | null {
  const tx = axes[0]?.ticks[x];
  const ty = axes[1]?.ticks[y];
  if (!tx || !ty) throw new Error(`cell ${x},${y} is outside the axes`);
  if (!tx.addressable || !ty.addressable) return null;
  const parts = [tx.frag, ty.frag].filter((p) => p);
  const port = ty.port ?? tx.port ?? 'h';
  return `${origin}${parts.join('.')}.${port}`;
}
