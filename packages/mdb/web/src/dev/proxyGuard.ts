// The Vite dev server's proxy to `mdb serve` (development only; the built bundle is served by
// `mdb serve` itself and needs none of this). `mdb serve` refuses any Origin that is not its own,
// and the browser's Origin names the Vite port, so the proxy rewrites it, but only for a request
// from the dev page itself. Any other Origin (another site, a DNS-rebound name) is refused here,
// so the proxy never turns a cross-site request into one the server would accept. A request
// without Origin is forwarded unchanged with its Sec-Fetch-Site header, which the server checks.

export const DEV_PORT = 5173;
export const TARGET = 'http://127.0.0.1:8765';
const LOOPBACK = ['127.0.0.1', 'localhost', '[::1]'];

export type Verdict =
  | { action: 'forward' }
  | { action: 'rewrite'; origin: string }
  | { action: 'refuse'; reason: string };

type Headers = Record<string, string | string[] | undefined>;

function one(value: string | string[] | undefined): string | undefined | null {
  if (Array.isArray(value)) return value.length === 1 ? value[0] : null; // null: duplicated
  return value;
}

/** What the dev proxy does with a request, from its headers. */
export function verdict(headers: Headers, devPort = DEV_PORT, target = TARGET): Verdict {
  const own = LOOPBACK.map((name) => `${name}:${devPort}`);
  const host = one(headers.host);
  if (host === null || !host || !own.includes(host.trim().toLowerCase())) {
    return { action: 'refuse', reason: `Host ${String(host)} is not this dev server` };
  }
  const origin = one(headers.origin);
  if (origin === undefined) return { action: 'forward' };
  if (origin !== null && own.map((h) => `http://${h}`).includes(origin.trim().toLowerCase())) {
    return { action: 'rewrite', origin: new URL(target).origin };
  }
  return { action: 'refuse', reason: `Origin ${String(origin)} is not this dev server` };
}

interface ProxyRequest {
  setHeader(name: string, value: string): void;
  destroy(): void;
}
interface IncomingRequest {
  headers: Headers;
}
interface ServerResponse {
  headersSent?: boolean;
  writeHead(status: number, headers?: Record<string, string>): unknown;
  end(body?: string): unknown;
}
interface Socket {
  end(data?: string): unknown;
}
interface Proxy {
  on(
    event: 'proxyReq',
    fn: (p: ProxyRequest, req: IncomingRequest, res: ServerResponse) => void,
  ): unknown;
  on(
    event: 'proxyReqWs',
    fn: (p: ProxyRequest, req: IncomingRequest, socket: Socket) => void,
  ): unknown;
}

/** `configure` for Vite's `server.proxy['/api']` (http-proxy events). */
export function guardDevProxy(proxy: Proxy): void {
  proxy.on('proxyReq', (outgoing, req, res) => {
    const v = verdict(req.headers);
    if (v.action === 'rewrite') outgoing.setHeader('origin', v.origin);
    if (v.action !== 'refuse') return;
    outgoing.destroy();
    if (!res.headersSent) {
      res.writeHead(403, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({ refused: 'origin_not_allowed', state: 'refused', detail: v.reason }),
      );
    }
  });
  proxy.on('proxyReqWs', (outgoing, req, socket) => {
    const v = verdict(req.headers);
    if (v.action === 'rewrite') outgoing.setHeader('origin', v.origin);
    if (v.action !== 'refuse') return;
    outgoing.destroy();
    socket.end('HTTP/1.1 403 Forbidden\r\n\r\n');
  });
}
