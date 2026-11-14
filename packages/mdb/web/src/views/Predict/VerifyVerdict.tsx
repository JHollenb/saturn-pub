// A verified forecast (the `verify` op, M29): the verdict is decided by the measured residual
// |native - predicted| (P20) against the floor (a few dtype spacings at the consumer's
// magnitude): `confirmed` (residual <= floor), `not confirmed` (residual > floor) or
// `unresolved` (no label for the native value; the reason says why). `edited` and `open` are
// the ledger's other states. The worst-case propagated bound is shown only when the op gives
// one (finite) and is information, never the verdict.
import { OpValue } from '../../chrome/OpValue';
import { str } from '../../chrome/labelled';

export const VERDICTS = ['confirmed', 'not confirmed', 'unresolved', 'edited', 'open'] as const;

/** The verdict chip: the op's verdict string (or the forecast's status), with its reason. */
export function VerdictChip({ v, status }: { v: Record<string, unknown> | null; status: string }) {
  const verdict = str(v?.verdict) ?? status;
  const reason = str(v?.reason);
  return (
    <>
      <span
        className={`chip verdict-chip${verdict === 'not confirmed' ? ' verdict-not' : ''}`}
        data-verdict={verdict}
        title={str(v?.criterion) ?? reason ?? undefined}
      >
        {verdict}
      </span>
      {reason && <span className="faint"> {reason}</span>}
    </>
  );
}

/** `residual ≤ floor` (confirmed) or `residual > floor` (not confirmed), then the worst-case
 *  bound as information when the op gave one. Nothing when the op measured no residual. */
export function ResidualVsFloor({ v }: { v: Record<string, unknown> }) {
  if (v.residual_max_abs === undefined) return null;
  const verdict = str(v.verdict);
  const within = v.within_bound === true ? 'holds' : v.within_bound === false ? 'exceeded' : null;
  return (
    <span className="residual-floor" data-testid="residual-floor">
      <OpValue value={v.residual_max_abs} what="residual" compact />
      {v.floor !== undefined && (
        <>
          {' '}
          {verdict === 'confirmed' ? '≤' : verdict === 'not confirmed' ? '>' : 'vs'}{' '}
          <span className="faint">floor</span> <OpValue value={v.floor} what="floor" compact />
          {str(v.dtype) && <span className="faint"> ({str(v.dtype)})</span>}
        </>
      )}
      {v.bound_max !== undefined && (
        <span
          className="faint"
          title="the worst-case propagated bound: information, not the verdict"
        >
          {' '}
          · bound <OpValue value={v.bound_max} what="bound" compact />
          {within ? ` ${within}` : ''}
        </span>
      )}
    </span>
  );
}
