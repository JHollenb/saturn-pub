// Render a `Loaded<T>`: loading / refused / error states are explicit, never blank.
import type { ReactNode } from 'react';
import type { Envelope } from '../api/types.gen';
import type { Loaded } from '../state/records';
import { RefusedView } from './NotAvailable';
import { Count } from './NumView';

export function Show<T>({
  what,
  loaded,
  children,
}: {
  what: string;
  loaded: Loaded<T>;
  children: (data: T, env: Envelope<T>) => ReactNode;
}) {
  if (loaded.state === 'loading') return <div className="faint">loading {what}…</div>;
  if (loaded.state === 'refused') return <RefusedView what={what} refused={loaded.refused} />;
  if (loaded.state === 'error') {
    return (
      <div className="na">
        <strong>{what}: client error</strong> {loaded.message}
      </div>
    );
  }
  return <>{children(loaded.env.data, loaded.env)}</>;
}

/** The `omitted` list of an envelope: what a bounded view left out, and how much. */
export function OmittedNote({ env }: { env: Envelope<unknown> }) {
  if (!env.omitted?.length) return null;
  return (
    <div className="faint" role="note">
      left out:{' '}
      {env.omitted.map((o) => (
        <span key={o.field} className="chip" title={o.reason ?? ''}>
          {o.field} <Count n={o.count} />
        </span>
      ))}
    </div>
  );
}
