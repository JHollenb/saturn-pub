// The state line of a panel's op: what it ran, and whether it is running (with Interrupt), was
// refused (with the op's reason), interrupted (where the cursor was left) or partly unresolved.
// A panel always says one of these; it is never an empty box.
import { RefusedView } from '../../chrome/NotAvailable';
import type { Slot } from '../../state/inspect';

export function SlotState({
  slot,
  what,
  onInterrupt,
  idle,
}: {
  slot: Slot;
  what: string;
  onInterrupt: () => void;
  /** What to say before the panel ran anything. */
  idle: string;
}) {
  switch (slot.state) {
    case 'idle':
      return (
        <div className="faint" data-testid={`${what}-idle`}>
          {idle}
        </div>
      );
    case 'running':
      return (
        <div className="slot-running" role="status" data-testid={`${what}-running`}>
          <span data-testid="state-running">running {slot.label}…</span>{' '}
          <button type="button" onClick={onInterrupt} title="Ctrl-C the running op">
            interrupt
          </button>
        </div>
      );
    case 'refused':
      return (
        <div className="na" role="note" data-testid={`${what}-refused`} data-ev="refused">
          <strong>
            {slot.label}: refused{slot.type ? ` (${slot.type})` : ''}
          </strong>{' '}
          <span className="num-refused">{slot.reason}</span>
        </div>
      );
    case 'interrupted':
      return (
        <div className="na" role="note" data-testid={`${what}-interrupted`}>
          <strong>{slot.label}: interrupted</strong> at <code>{slot.where}</code>; no result. The
          engine rolled any half-done move back.
        </div>
      );
    case 'request-refused':
      return <RefusedView what={slot.label} refused={slot.refused} />;
    case 'done':
      return (
        <div className="faint" data-testid={`${what}-done`}>
          <code>{slot.label}</code>
          {slot.partial && (
            <span data-testid={`${what}-unresolved`} title={slot.partial}>
              {' '}
              · partly unresolved: {slot.partial}
            </span>
          )}
        </div>
      );
  }
}
