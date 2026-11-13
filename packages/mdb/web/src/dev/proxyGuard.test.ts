import { describe, expect, it } from 'vitest';
import { guardDevProxy, verdict } from './proxyGuard';

describe('dev proxy origin guard', () => {
  it('rewrites the dev page’s own Origin to the server’s', () => {
    for (const host of ['localhost:5173', '127.0.0.1:5173', '[::1]:5173']) {
      expect(verdict({ host, origin: `http://${host}` })).toEqual({
        action: 'rewrite',
        origin: 'http://127.0.0.1:8765',
      });
    }
  });

  it('forwards a request without Origin unchanged (the server checks Sec-Fetch-Site)', () => {
    expect(verdict({ host: 'localhost:5173' })).toEqual({ action: 'forward' });
  });

  it('refuses any other Origin, a foreign Host, and duplicated headers', () => {
    for (const headers of [
      { host: 'localhost:5173', origin: 'http://evil.example' },
      { host: 'localhost:5173', origin: 'http://localhost:5174' },
      { host: 'localhost:5173', origin: 'https://localhost:5173' },
      { host: 'localhost:5173', origin: 'null' },
      { host: 'evil.example:5173', origin: 'http://evil.example:5173' }, // DNS rebinding
      { host: 'evil.example:5173' },
      { host: 'localhost:5173', origin: ['http://localhost:5173', 'http://evil.example'] },
    ]) {
      expect(verdict(headers).action).toBe('refuse');
    }
  });

  it('acts on the proxied request: rewrite, or destroy and answer 403', () => {
    const handlers: Record<string, (...a: never[]) => void> = {};
    guardDevProxy({ on: (event: string, fn: (...a: never[]) => void) => (handlers[event] = fn) });
    const set: string[] = [];
    let destroyed = false;
    const outgoing = {
      setHeader: (k: string, v: string) => set.push(`${k}=${v}`),
      destroy: () => (destroyed = true),
    };
    const answers: unknown[] = [];
    const res = { writeHead: (s: number) => answers.push(s), end: (b?: string) => answers.push(b) };
    const call = handlers.proxyReq as unknown as (o: unknown, r: unknown, s: unknown) => void;
    call(outgoing, { headers: { host: 'localhost:5173', origin: 'http://localhost:5173' } }, res);
    expect(set).toEqual(['origin=http://127.0.0.1:8765']);
    expect(destroyed).toBe(false);
    call(outgoing, { headers: { host: 'localhost:5173', origin: 'http://evil.example' } }, res);
    expect(destroyed).toBe(true);
    expect(answers[0]).toBe(403);
    let ended = '';
    const ws = handlers.proxyReqWs as unknown as (o: unknown, r: unknown, s: unknown) => void;
    ws(
      outgoing,
      { headers: { host: 'localhost:5173', origin: 'http://evil.example' } },
      {
        end: (d: string) => (ended = d),
      },
    );
    expect(ended).toContain('403');
  });
});
