// Typed client for `mdb serve`. Every response is either an Envelope whose `schema` must match
// the requested view, or a Refused body. Refusal is a first-class state, never an exception.
import type {
  Capabilities,
  Envelope,
  Event,
  Frame,
  GraphView,
  Hello,
  RecordDetail,
  RecordPage,
  ReduceResult,
  Refused,
  SessionSummary,
  SymbolDetail,
  SymbolPage,
  TimelineBlock,
  BranchTree,
} from './types.gen';
import { SCHEMAS } from './types.gen';

export type View = keyof typeof SCHEMAS;

/** The data type each route's envelope carries. A function that names a route gets this type and
 *  nothing else, so pairing a route with the wrong view type is a compile error. */
export interface ViewTypes {
  hello: Hello;
  sessions: SessionSummary[];
  capabilities: Capabilities;
  frame: Frame;
  timeline: TimelineBlock;
  branches: BranchTree;
  symbols: SymbolPage;
  symbol: SymbolDetail;
  graph: GraphView;
  records: RecordPage;
  record: RecordDetail;
  reduce: ReduceResult;
  op: unknown;
  event: Event;
}
// Compile-time check that the map covers exactly the generated schema names.
type Assert<T extends true> = T;
export type _RoutesCovered = Assert<
  [View] extends [keyof ViewTypes] ? ([keyof ViewTypes] extends [View] ? true : false) : false
>;

export type Result<T> =
  | { ok: true; env: Envelope<T>; status: number }
  | { ok: false; refused: Refused; status: number };

export interface RequestOptions {
  signal?: AbortSignal;
}

/** The server answered with a body that is neither the requested envelope nor a refusal. */
export class SchemaError extends Error {}

export const API = '/api/v1';

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

function query(params?: Record<string, string | number | null | undefined>): string {
  if (!params) return '';
  const parts = Object.entries(params)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`);
  return parts.length ? `?${parts.join('&')}` : '';
}

export function isRefused(body: unknown): body is Refused {
  return typeof body === 'object' && body !== null && typeof (body as Refused).refused === 'string';
}

const STATES = ['resolved', 'unresolved', 'refused'];

/** Validate one decoded body against the view it was requested as. Throws SchemaError when the
 *  body is neither a well-formed refusal nor a well-formed envelope of that view. `state` is
 *  required on both; an envelope that is not `resolved` must give a `reason`. */
export function accept<V extends View>(
  view: V,
  body: unknown,
  status: number,
): Result<ViewTypes[V]> {
  if (isRefused(body)) {
    if (body.state !== 'refused') {
      throw new SchemaError(`${view}: refusal ${body.refused} lacks state "refused"`);
    }
    return { ok: false, refused: body, status };
  }
  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    throw new SchemaError(`${view}: response is not a JSON object`);
  }
  const env = body as Partial<Envelope<unknown>>;
  if (env.schema !== SCHEMAS[view]) {
    throw new SchemaError(`${view}: expected schema ${SCHEMAS[view]}, got ${String(env.schema)}`);
  }
  if (!('data' in env) || typeof env.digest !== 'string' || typeof env.bytes !== 'number') {
    throw new SchemaError(`${view}: envelope lacks data, digest or bytes`);
  }
  if (typeof env.state !== 'string' || !STATES.includes(env.state)) {
    throw new SchemaError(`${view}: envelope lacks a valid state (got ${String(env.state)})`);
  }
  if (env.state !== 'resolved' && (typeof env.reason !== 'string' || !env.reason)) {
    throw new SchemaError(`${view}: envelope is ${env.state} but gives no reason`);
  }
  return { ok: true, env: env as Envelope<ViewTypes[V]>, status };
}

async function decode<V extends View>(view: V, response: Response): Promise<Result<ViewTypes[V]>> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return {
      ok: false,
      refused: { refused: 'bad_response', detail: `HTTP ${response.status}: not JSON` },
      status: response.status,
    };
  }
  try {
    return accept(view, body, response.status);
  } catch (error) {
    if (!(error instanceof SchemaError)) throw error;
    return {
      ok: false,
      refused: { refused: 'bad_envelope', detail: error.message },
      status: response.status,
    };
  }
}

export function url(path: string, params?: Record<string, string | number | null | undefined>) {
  return `${API}${path}${query(params)}`;
}

/** GET a fully built URL (from `url()`). */
export async function getViewUrl<V extends View>(
  view: V,
  fullUrl: string,
  options: RequestOptions = {},
  fetchImpl: FetchLike = fetch,
): Promise<Result<ViewTypes[V]>> {
  let response: Response;
  try {
    response = await fetchImpl(fullUrl, options.signal ? { signal: options.signal } : undefined);
  } catch (error) {
    return { ok: false, refused: { refused: 'network', detail: String(error) }, status: 0 };
  }
  return decode(view, response);
}

export function getView<V extends View>(
  view: V,
  path: string,
  params?: Record<string, string | number | null | undefined>,
  fetchImpl: FetchLike = fetch,
  options: RequestOptions = {},
): Promise<Result<ViewTypes[V]>> {
  return getViewUrl(view, url(path, params), options, fetchImpl);
}

export async function postView<V extends View>(
  view: V,
  path: string,
  body: unknown,
  fetchImpl: FetchLike = fetch,
  options: RequestOptions = {},
): Promise<Result<ViewTypes[V]>> {
  let response: Response;
  try {
    response = await fetchImpl(url(path), {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      ...(options.signal ? { signal: options.signal } : {}),
    });
  } catch (error) {
    return { ok: false, refused: { refused: 'network', detail: String(error) }, status: 0 };
  }
  return decode(view, response);
}
