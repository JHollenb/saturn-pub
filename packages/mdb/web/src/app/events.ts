// The session event socket. A recorded project only sends hello + heartbeats; the indicator
// shows whether the server is reachable, not that a model is running. On the live session each
// completed op arrives as an `op` event (no tensors); the live store re-fetches what is stale.
import { useEffect, useState } from 'react';
import { API } from '../api/client';
import { SCHEMAS } from '../api/types.gen';
import { useLive } from '../state/live';

export type Link = 'connecting' | 'open' | 'closed';

export function useEvents(sid: string | null): Link {
  const [link, setLink] = useState<Link>('connecting');
  useEffect(() => {
    if (!sid || typeof WebSocket === 'undefined') return;
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(
      `${proto}://${window.location.host}${API}/s/${encodeURIComponent(sid)}/events`,
    );
    setLink('connecting');
    ws.onmessage = (m) => {
      try {
        const body = JSON.parse(String(m.data)) as {
          schema?: string;
          data?: { type?: string; data?: Record<string, unknown> };
        };
        setLink(body.schema === SCHEMAS.event ? 'open' : 'closed');
        if (body.data?.type === 'op' && body.data.data && useLive.getState().sid === sid) {
          useLive.getState().onEvent(body.data.data);
        }
      } catch {
        setLink('closed');
      }
    };
    ws.onclose = () => setLink('closed');
    ws.onerror = () => setLink('closed');
    return () => ws.close();
  }, [sid]);
  return link;
}
