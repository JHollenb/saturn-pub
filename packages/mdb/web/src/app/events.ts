// The session event socket. A recorded project only sends hello + heartbeats; the indicator
// shows whether the server is reachable, not that a model is running.
import { useEffect, useState } from 'react';
import { API } from '../api/client';
import { SCHEMAS } from '../api/types.gen';

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
        const body = JSON.parse(String(m.data)) as { schema?: string };
        setLink(body.schema === SCHEMAS.event ? 'open' : 'closed');
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
